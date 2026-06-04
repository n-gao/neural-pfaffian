from typing import Generic, TypeVar

import folx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Float, Integer

from neural_pfaffian.clipping import Clipping
from neural_pfaffian.hamiltonian import KineticEnergyOp, make_local_energy
from neural_pfaffian.mcmc import MetropolisHastings
from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    WaveFunctionParameters,
)
from neural_pfaffian.overlap import OverlapPenalty, OverlapState
from neural_pfaffian.preconditioner import Preconditioner
from neural_pfaffian.spin_operator import SpinPenalty
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils import RollingAverage
from neural_pfaffian.utils.jax_utils import (
    REPLICATE_SPEC,
    SerializeablePyTree,
    distribute_keys,
    jit,
    psum,
    shmap,
)
from neural_pfaffian.utils.segment_utils import unsegment_axis
from neural_pfaffian.utils.summary_stats import (
    weighted_centering,
    weighted_mean,
    weighted_std,
)
from neural_pfaffian.utils.tree_utils import (
    tree_add,
    tree_squared_norm,
)

LocalEnergy = Float[Array, 'batch_size n_mols']
S = TypeVar('S', bound=Systems)
SMOOTH_DATA_KEY = 'smooth_data'


O = TypeVar('O')
COrb = TypeVar('COrb')
OS = TypeVar('OS')
PS = TypeVar('PS')


class VMCState(Generic[PS], SerializeablePyTree):
    params: WaveFunctionParameters
    optimizer: optax.OptState
    preconditioner: PS
    overlap: OverlapState | None
    step: Integer[Array, '']
    epoch: Integer[Array, '']


class SmoothData(PyTreeNode):
    energy: Float[Array, ' n_mols']
    std: Float[Array, ' n_mols']

    @classmethod
    def init_systems(cls, systems: S) -> S:
        if SMOOTH_DATA_KEY not in systems.mol_data:
            smooth_data = cls(
                energy=jnp.zeros(systems.n_mols, dtype=jnp.float32),
                std=jnp.zeros(systems.n_mols, dtype=jnp.float32),
            )
            smooth_data = RollingAverage.init(smooth_data)
            systems = systems.set_mol_data(SMOOTH_DATA_KEY, smooth_data)
        return systems


class VMC(Generic[PS, O, COrb, OS, S], PyTreeNode):
    wave_function: GeneralizedWaveFunction[O, COrb, OS, S] = field(pytree_node=False)
    preconditioner: Preconditioner[PS] = field(pytree_node=False)
    optimizer: optax.GradientTransformation = field(pytree_node=False)
    sampler: MetropolisHastings = field(pytree_node=False)
    clipping: Clipping = field(pytree_node=False)
    overlap_penalty: OverlapPenalty[O, COrb, OS, S] | None = field(
        pytree_node=False,
        default=None,
    )
    spin_penalty: SpinPenalty[O, COrb, OS, S] | None = field(
        pytree_node=False,
        default=None,
    )
    reweight_overlap_mean: bool = field(pytree_node=False, default=False)

    def init(self, key: Array, systems: Systems):
        key, subkey = jax.random.split(key)
        params = self.wave_function.init(key, systems.example_input)
        overlap_state = self.overlap_penalty.init() if self.overlap_penalty else None
        assert self.overlap_penalty or not self.reweight_overlap_mean, (
            'Sample reweighting is only supported with an overlap penalty.'
        )
        return VMCState(
            params=params,
            optimizer=self.optimizer.init(params),  # type: ignore
            preconditioner=self.preconditioner.init(subkey, params, systems),
            overlap=overlap_state,
            step=jnp.zeros((), dtype=jnp.int32),
            epoch=jnp.ones((), dtype=jnp.int32),
        )

    def init_systems(self, key: Array, systems: S) -> S:
        @shmap(
            in_specs=(REPLICATE_SPEC, systems.partition_spec),
            out_specs=systems.partition_spec,
        )
        def init(key: Array, systems: S):
            key = distribute_keys(key)
            key, subkey = jax.random.split(key)
            systems = self.sampler.init_systems(subkey, systems)
            systems = SmoothData.init_systems(systems)
            if self.overlap_penalty is not None:
                systems = self.overlap_penalty.init_systems(systems)
            if self.spin_penalty is not None:
                systems = self.spin_penalty.init_systems(systems)
            return systems

        return init(key, systems)

    @jit
    def local_energy(
        self,
        state: VMCState,
        systems: Systems,
        key: Array,
    ) -> tuple[LocalEnergy, dict[str, Array]]:
        local_energy_fn = make_local_energy(
            self.wave_function,
            KineticEnergyOp.FORWARD,
        )
        batch_size = systems.electrons.shape[0]
        keys = jax.random.split(key, batch_size)
        memory_scaling_factor = (
            max(systems.n_nuc_by_mol) * max(systems.n_elec_by_mol) ** 2
        )
        folx_batch_size = max(
            1,
            # largest dense jac should not exceed ~3Gb
            360_000 // memory_scaling_factor,
        )
        local_energy_fn = folx.batched_vmap(
            local_energy_fn,
            max_batch_size=folx_batch_size,
            in_axes=(None, systems.electron_vmap, None, 0),
        )

        e_l, e_aux = local_energy_fn(
            state.params,
            systems,
            self.wave_function.reparams(state.params, systems),
            keys,
        )
        # e_l: (batch_size, n_mols)
        return e_l, e_aux

    @jit
    def mcmc_step(self, key: Array, state: VMCState[PS], systems: Systems):
        @shmap(
            in_specs=(REPLICATE_SPEC, state.partition_spec, systems.partition_spec),
            out_specs=(systems.partition_spec, REPLICATE_SPEC),
        )
        def _mcmc_step(key: Array, state: VMCState[PS], systems: Systems):
            key = distribute_keys(key)
            # Sampling
            key, subkey = jax.random.split(key)
            systems, aux_data = self.sampler(subkey, state.params, systems)
            return systems, aux_data

        return _mcmc_step(key, state, systems)

    @jit(donate_argnames=('state', 'systems'))
    def step(self, key: Array, state: VMCState[PS], systems: Systems):
        @shmap(
            in_specs=(REPLICATE_SPEC, state.partition_spec, systems.partition_spec),
            out_specs=(
                state.partition_spec,
                systems.partition_spec,
                REPLICATE_SPEC,
            ),
            check_vma=False,
        )
        def _step(key: Array, state: VMCState[PS], systems: Systems):
            key = distribute_keys(key)
            aux_data = {}

            # Sampling
            key, subkey = jax.random.split(key)
            systems, mcmc_aux = self.sampler(subkey, state.params, systems)
            aux_data |= {f'mcmc/{k}': v for k, v in mcmc_aux.items()}

            key, subkey = jax.random.split(key)
            raw_e_l, energy_aux = self.local_energy(state, systems, subkey)

            # Local energy. The energy is never reweighted, so clipping uses the
            # unmasked/unweighted defaults.
            clipped_e_l = self.clipping(raw_e_l)

            # Adding up penalty terms
            auxiliary_grads = jax.tree.map(jnp.zeros_like, state.params)
            dL_dlogpsi = jnp.zeros_like(raw_e_l)

            # Overlap penalty
            overlap_state = state.overlap
            if self.overlap_penalty is not None:
                assert overlap_state is not None
                (
                    (dOverlap_dlogpsi, overlap_aux),
                    systems,
                    overlap_state,
                ) = self.overlap_penalty(
                    state.params,
                    systems,
                    clipped_e_l,
                    overlap_state,
                    state.step,
                    reweight_overlap_mean=self.reweight_overlap_mean,
                )

                dL_dlogpsi += dOverlap_dlogpsi
                aux_data |= {f'overlap/{k}': v for k, v in overlap_aux.items()}

            if self.spin_penalty is not None:
                (spin_grads, spin_aux), systems = self.spin_penalty(
                    state.params,
                    systems,
                    state.step,
                )
                auxiliary_grads = tree_add(auxiliary_grads, spin_grads)

                spin_per_mol_metrics = [
                    (k, v) for k, v in spin_aux.items() if v.ndim == 1
                ]
                aux_data |= {k: v for k, v in spin_aux.items() if v.ndim == 0}
                aux_data |= {k: jnp.mean(v) for k, v in spin_per_mol_metrics}
            else:
                spin_per_mol_metrics = []

            # Energy penalty
            dE_dlogpsi = weighted_centering(clipped_e_l)
            aux_data |= {'dE_dlogpsi': psum(jnp.sum(dE_dlogpsi**2)) ** 0.5}
            dL_dlogpsi += dE_dlogpsi

            # Preconditioning
            gradient, preconditioner_state, precond_aux = self.preconditioner.apply(
                state.params,
                systems,
                dL_dlogpsi,
                state.preconditioner,
                auxiliary_grads,
            )
            aux_data |= {f'preconditioner/{k}': v for k, v in precond_aux.items()}

            # Gradient norm for logging
            grad_norm = tree_squared_norm(gradient) ** 0.5

            # Apply update
            updates, opt_state = self.optimizer.update(gradient, state.optimizer)  # type: ignore
            params = optax.apply_updates(state.params, updates)  # type: ignore

            # Logging
            aux_grad_norm = tree_squared_norm(auxiliary_grads) ** 0.5
            aux_data |= {'aux_grad_norm': aux_grad_norm}
            n_unique_mols = systems.n_unique_mols
            mol_ids = np.asarray(systems.mol_ids)

            E_per_mol = weighted_mean(clipped_e_l)
            E = E_per_mol.mean()
            E_std_per_mol = weighted_std(clipped_e_l)
            E_std = E_std_per_mol.mean()

            E_components_per_mol = {
                key: weighted_mean(val) for key, val in energy_aux.items()
            }
            E_components = {key: val.mean() for key, val in E_components_per_mol.items()}
            aux_data |= E_components

            smooth_data = systems.get_mol_data(SMOOTH_DATA_KEY)
            smooth_data = smooth_data.update(SmoothData(E_per_mol, E_std_per_mol))
            systems = systems.set_mol_data(SMOOTH_DATA_KEY, smooth_data)
            smooth_data = smooth_data.value()
            E_per_mol_smooth = smooth_data.energy
            E_std_per_mol_smooth = smooth_data.std
            E_smooth = E_per_mol_smooth.mean()
            E_std_smooth = E_std_per_mol_smooth.mean()

            ground_state_energy = jax.ops.segment_min(
                E_per_mol,
                mol_ids,
                num_segments=n_unique_mols,
            )
            excitation_energy_per_mol = E_per_mol - ground_state_energy[mol_ids]
            ground_state_energy_smooth = jax.ops.segment_min(
                E_per_mol_smooth,
                mol_ids,
                num_segments=n_unique_mols,
            )
            excitation_energy_per_mol_smooth = (
                E_per_mol_smooth - ground_state_energy_smooth[mol_ids]
            )

            aux_data |= {
                'E': E,
                'E_std': E_std,
                'E_smooth': E_smooth,
                'E_std_smooth': E_std_smooth,
                'grad_norm': grad_norm,
            }

            # Log per mol data
            metrics = [
                ('E', E_per_mol),
                ('E_std', E_std_per_mol),
                ('E_smooth', E_per_mol_smooth),
                ('E_std_smooth', E_std_per_mol_smooth),
                ('excitation_energy', excitation_energy_per_mol),
                ('excitation_energy_smooth', excitation_energy_per_mol_smooth),
                *spin_per_mol_metrics,
            ]

            for key, array in metrics:
                unsegmented = unsegment_axis(array, mol_ids)
                # mean only for safety; values should be scalar
                aux_data |= {
                    f'{key}/structure_{i}/state_{j}': unsegmented[i, j].mean()
                    for i in np.unique(mol_ids)
                    for j in range(unsegmented.shape[1])
                }

            return (
                state.replace(
                    params=params,
                    optimizer=opt_state,
                    preconditioner=preconditioner_state,
                    overlap=overlap_state,
                ),
                systems,
                aux_data,
            )

        return _step(key, state, systems)
