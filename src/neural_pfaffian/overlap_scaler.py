from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from chex import Numeric
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Float, Int
from optax import Schedule

from neural_pfaffian.nn.ops import softplus
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils import EMA, Modules
from neural_pfaffian.utils.jax_utils import jit
from neural_pfaffian.utils.schedule import ScheduleConfig, get_schedule
from neural_pfaffian.utils.segment_utils import unsegment_axis
from neural_pfaffian.utils.summary_stats import weighted_mean, weighted_std

if TYPE_CHECKING:
    from neural_pfaffian.vmc import LocalEnergy

_LOCAL_ENERGY_EMA_KEY = 'local_energy_ema'
_ENERGY_STD_EMA_KEY = 'energy_std_ema'

Overlap = Float[Array, 'n_unique_mols n_states n_states']
OverlapWeights = Overlap


S = TypeVar('S', bound=Systems)


class OverlapScaler(Protocol):
    def init_systems(self, systems: S) -> S:
        """Adds the necessary mol_data to the systems."""
        ...

    def __call__(
        self,
        systems: Systems,
        local_energy: 'LocalEnergy',
        step: Int[Array, ''],
    ) -> tuple[OverlapWeights, Systems]:
        r"""Computes the overlap gradient scaling factor :math:`\alpha`.
        Uses systems to store moving averages of the local energy for each molecule."""
        ...


class NoScaler(OverlapScaler, PyTreeNode):
    triangular_mask: bool = field(default=True, pytree_node=False)

    def init_systems(self, systems: Systems) -> Systems:
        return systems

    @jit
    def __call__(
        self,
        systems: Systems,
        local_energy: 'LocalEnergy',
        step: Int[Array, ''],
    ) -> tuple[OverlapWeights, Systems]:
        """Does not scale the overlap gradient. Applies a mask such that
        gradients of lower lying state overlaps wrt higher lying states are zero.
        **Note**: Should not be used with PES wave functions since it implies an ordering of states."""
        overlap_weights = jnp.ones(
            (systems.n_unique_mols, systems.max_num_states, systems.max_num_states),
            dtype=local_energy.dtype,
        )
        if self.triangular_mask:
            overlap_weights = jnp.tril(overlap_weights, k=-1)
        return overlap_weights, systems


class EnergyDiffScaler(OverlapScaler, PyTreeNode):
    """Scales the overlap gradients by the difference in local energies of the states.
    In case of noisy energy estimates, the overlap gradients are scaled by the standard deviation
    of the local energies."""

    decay_schedule: Schedule = field(pytree_node=False)

    min_scale_factor: Schedule = field(
        default_factory=lambda: get_schedule(0.001),
        pytree_node=False,
    )
    max_scale_factor: float = 5.0
    asym_strategy: Literal['none', 'softplus', 'sigmoid', 'step'] = field(
        pytree_node=False,
        default='none',
    )
    r"""Different strategies to asymmetrize the overlap weighting.
    - `none`: No asymmetrization, i.e., ordered states and triangular alpha
    - `softplus`: Using a softplus on energy differences to create an asymmetric alpha
    - `sigmoid`: :math:`\sigma((E_i - E_j) * \text{asym_scale}) \max(|E_i - E_j|, \operatorname{std})`
    """
    asym_scale: float = 1e3
    r"""Determines how sharp the asymmetrization is"""

    @classmethod
    def create(
        cls,
        decay_schedule: dict[str, Any],
        min_scale_factor: ScheduleConfig,
        **kwargs,
    ) -> 'EnergyDiffScaler':
        return cls(
            decay_schedule=get_schedule(decay_schedule),
            min_scale_factor=get_schedule(min_scale_factor),
            **kwargs,
        )

    def init_systems(self, systems):
        if _LOCAL_ENERGY_EMA_KEY not in systems.mol_data:
            local_energy_ema = EMA.init(data=jnp.zeros((), dtype=jnp.float32))
            local_energy_ema = jax.tree.map(
                lambda x: jnp.stack([x] * systems.n_mols, axis=0),
                local_energy_ema,
            )
            systems = systems.set_mol_data(_LOCAL_ENERGY_EMA_KEY, local_energy_ema)
        if _ENERGY_STD_EMA_KEY not in systems.mol_data:
            energy_std_ema = EMA.init(data=jnp.zeros((), dtype=jnp.float32))
            energy_std_ema = jax.tree.map(
                lambda x: jnp.stack([x] * systems.n_mols, axis=0),
                energy_std_ema,
            )
            systems = systems.set_mol_data(_ENERGY_STD_EMA_KEY, energy_std_ema)
        return systems

    @jit
    def __call__(
        self,
        systems: Systems,
        local_energy: 'LocalEnergy',
        step: Int[Array, ''],
    ) -> tuple[OverlapWeights, Systems]:
        decay = self.decay_schedule(step)

        alpha_diff, systems, signs = self._energy_diff_scaler(
            systems,
            local_energy,
            decay,
        )
        alpha_std, systems = self._energy_std_scaler(
            systems,
            local_energy,
            decay,
        )

        alpha = self._asymmetrize(alpha_diff, alpha_std, signs)

        # Apply the min and max scale factors
        min_scale = self.min_scale_factor(step)
        alpha = jnp.clip(
            jnp.nan_to_num(alpha, nan=1.0),
            jnp.where(alpha > alpha.mT, min_scale, 0.0),
            self.max_scale_factor,
        )

        # Ensure that the diagonal is zero (no self-overlaps)
        alpha = alpha.at[:, *jnp.diag_indices(alpha.shape[-1])].set(0.0)

        return alpha, systems

    def _energy_std_scaler(
        self,
        systems: Systems,
        local_energy: 'LocalEnergy',
        decay: Numeric,
    ):
        # Update the moving average of the local energy
        energy_std_ema = systems.get_mol_data(_ENERGY_STD_EMA_KEY)
        E_std_per_mol = weighted_std(local_energy, None, None)
        energy_std_ema = energy_std_ema.update(E_std_per_mol, decay)
        systems = systems.set_mol_data(_ENERGY_STD_EMA_KEY, energy_std_ema)

        # Compute the overlap scaling factor
        energy_std_ema = energy_std_ema.value()
        energy_std_ema = unsegment_axis(
            energy_std_ema,
            np.array(systems.mol_id_groups),
            axis=0,
            indices_are_grouped=True,
            num_segments=systems.n_unique_mols,
        )
        # (n_unique_mol, states)

        alpha = (
            jnp.ones((*energy_std_ema.shape, energy_std_ema.shape[-1]))
            - jnp.eye(energy_std_ema.shape[-1])[None]
        )
        alpha *= energy_std_ema[:, :, None]

        return alpha, systems

    def _energy_diff_scaler(
        self,
        systems: Systems,
        local_energy: 'LocalEnergy',
        decay: Numeric,
    ):
        # Update the moving average of the local energy
        E_ema = systems.get_mol_data(_LOCAL_ENERGY_EMA_KEY)
        E_per_mol = weighted_mean(local_energy, None, None)
        E_ema = E_ema.update(E_per_mol, decay)
        systems = systems.set_mol_data(_LOCAL_ENERGY_EMA_KEY, E_ema)

        # Compute the overlap scaling factor
        E_per_mol_ema = E_ema.value()
        E_per_mol_ema = unsegment_axis(
            E_per_mol_ema,
            np.array(systems.mol_id_groups),
            axis=0,
            indices_are_grouped=True,
            num_segments=systems.n_unique_mols,
        )
        # (n_unique_mol, states)
        energy_diffs = E_per_mol_ema[:, :, None] - E_per_mol_ema[:, None]
        signs = jnp.sign(energy_diffs)
        return energy_diffs, systems, signs

    def _asymmetrize(self, alpha_diff, alpha_std, signs):
        match self.asym_strategy:
            case 'none':
                return self._no_asymmetrization(alpha_diff, alpha_std, signs)
            case 'softplus':
                return self._softplus_asymmetrization(alpha_diff, alpha_std, signs)
            case 'sigmoid':
                return self._sigmoid_asymmetrization(alpha_diff, alpha_std, signs)
            case 'step':
                return self._step_asymmetrization(alpha_diff, alpha_std, signs)

    def _no_asymmetrization(self, alpha_diff, alpha_std, signs):
        alpha = jnp.maximum(alpha_diff, alpha_std)
        return jnp.tril(alpha, k=-1)

    def _softplus_asymmetrization(self, alpha_diff, alpha_std, signs):
        asym_diff = softplus(alpha_diff, beta=self.asym_scale)
        asym_std = softplus(signs * alpha_std, beta=self.asym_scale)
        return jnp.maximum(asym_diff, asym_std)

    def _sigmoid_asymmetrization(self, alpha_diff, alpha_std, signs):
        scaler = jax.nn.sigmoid(alpha_diff * self.asym_scale)
        return scaler * jnp.maximum(jnp.abs(alpha_diff), alpha_std)

    def _step_asymmetrization(self, alpha_diff, alpha_std, signs):
        scaler = jnp.heaviside(alpha_diff, alpha_diff)
        return scaler * jnp.maximum(jnp.abs(alpha_diff), alpha_std)


OVERLAP_SCALER = Modules[OverlapScaler](
    {
        'none': NoScaler,
        'diff-std': EnergyDiffScaler,
    },
)
