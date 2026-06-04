from typing import TYPE_CHECKING, Generic, TypeVar

import einops
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode, field
from jaxtyping import Array, DTypeLike, Float, Int

from neural_pfaffian.clipping import Clipping, Masking, NoneMasking
from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    WaveFunctionParameters,
)
from neural_pfaffian.overlap_scaler import Overlap, OverlapScaler, OverlapWeights
from neural_pfaffian.sample_reweighting import (
    LOG_NORMALIZER_CONSTANTS_KEY,
    ReweightingFactor,
    SampleMask,
    compute_logpsi,
    compute_reweighting_factor,
    compute_sample_mask,
    convert_reweighted_tensor_to_unreweighted,
    get_normalizing_constant_ratios,
    unpack_reweighted_tensor,
    update_normalizing_constant_ratios,
)
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils.jax_utils import (
    jit,
    pmax_if_pmap,
    pmean_if_pmap,
    pmin_if_pmap,
    psum_if_pmap,
)
from neural_pfaffian.utils.segment_utils import unsegment_axis
from neural_pfaffian.utils.summary_stats import weighted_mean
from neural_pfaffian.utils.tree_utils import tree_to_dtype

if TYPE_CHECKING:
    from neural_pfaffian.vmc import LocalEnergy

MeanRatio = Float[Array, 'n_unique_mols denominator numerator']
r"""Represents :math:`\tilde{R}_{ij} = \Bbb{E}_{i}\left[\frac{\psi_j}{\psi_i}\right]`"""
O = TypeVar('O')
"""Orbital type"""
COrb = TypeVar('COrb')
"""Core orbital type"""
OS = TypeVar('OS')
"""Orbital state type"""
S = TypeVar('S', bound=Systems)


class OverlapState(PyTreeNode): ...


class OverlapPenalty(Generic[O, COrb, OS, S], PyTreeNode):
    wave_function: GeneralizedWaveFunction[O, COrb, OS, S] = field(pytree_node=False)
    clipping: Clipping = field(pytree_node=False)
    overlap_scaler: OverlapScaler = field(pytree_node=False)

    penalty_scale: float = 4.0
    dtype: DTypeLike | None = field(pytree_node=False, default=None)
    masking: Masking = field(pytree_node=False, default=NoneMasking())

    def init(self) -> OverlapState:
        return OverlapState()

    def init_systems(self, systems: S) -> S:
        systems = self.overlap_scaler.init_systems(systems)
        systems = systems.set_mol_data(
            LOG_NORMALIZER_CONSTANTS_KEY,
            jnp.zeros((systems.n_mols,), dtype=jnp.float64),
        )
        return systems

    @jit(static_argnames=('reweight_overlap_mean',))
    def __call__(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        local_energy: 'LocalEnergy',
        state: OverlapState,
        step: Int[Array, ''],
        reweight_overlap_mean: bool,
    ) -> tuple[
        tuple[
            'LocalEnergy',
            dict[str, Float[Array, '...']],
        ],
        Systems,
        OverlapState,
    ]:
        aux_data = {}
        if reweight_overlap_mean:
            systems, normalizer_aux = update_normalizing_constant_ratios(
                self.wave_function,
                params,
                systems,
            )
            aux_data |= normalizer_aux
            reweighting_factor = compute_reweighting_factor(
                self.wave_function,
                params,
                systems,
            )
            sample_mask, sample_mask_aux = compute_sample_mask(
                self.wave_function,
                params,
                systems,
                self.masking,
                reweighting_factor,
                reweight_overlap_mean,
            )
            aux_data |= sample_mask_aux
            aux_data |= self._effective_sample_size(
                sample_mask, reweighting_factor, systems
            )
            aux_data |= {
                'reweighting_factor_mean': pmean_if_pmap(reweighting_factor.mean()),
                'reweighting_factor_max': pmax_if_pmap(reweighting_factor.max()),
            }
        else:
            sample_mask = None
            reweighting_factor = None
        out_dtypes = jax.tree.map(lambda x: x.dtype, systems)
        if self.dtype is not None:
            params, systems = tree_to_dtype((params, systems), self.dtype)

        overlap_weights, systems = self.overlap_scaler(
            systems,
            local_energy,
            step,
        )
        overlap_weights *= self.penalty_scale

        grad_contributions, mean_ratio, mean_ratio_aux = self._mean_ratio(
            params,
            systems,
            reweighting_factor,
            reweight_overlap_mean=reweight_overlap_mean,
            sample_mask=sample_mask,
        )

        grad_contributions *= overlap_weights

        cotangent = (
            jnp.sum(grad_contributions, axis=-1) * 2
        )  # (total_samples, mol, denominator)
        cotangent = cotangent[:, systems.mol_id_groups, systems.excitations]

        if reweight_overlap_mean:
            assert sample_mask is not None and reweighting_factor is not None
            cotangent = self.clipping(
                cotangent,
                mask=sample_mask,
                reweighting_factor=reweighting_factor,
                data_is_reweighted=True,
            )

        overlap = _compute_mean_overlap(mean_ratio)
        loss = _compute_overlap_loss(overlap, overlap_weights)

        systems = jax.tree.map(jax.lax.convert_element_type, systems, out_dtypes)
        cotangent = jax.lax.convert_element_type(cotangent, local_energy.dtype)

        # Logging
        aux_data |= {
            'loss': loss,
            'max_pair_overlap': (jnp.tril(overlap, k=-1)).max(),
            'max_mean_ratio': jnp.abs(
                mean_ratio * (jnp.ones_like(mean_ratio) - jnp.eye(mean_ratio.shape[-1])),
            ).max(),
            'avg_mean_ratio': jnp.abs(
                mean_ratio * (jnp.ones_like(mean_ratio) - jnp.eye(mean_ratio.shape[-1])),
            ).mean(),
            'mean_overlap_weights': overlap_weights.mean(),
            'dL_dlogpsi_norm': psum_if_pmap(jnp.sum(cotangent**2)) ** 0.5,
        }
        aux_data |= {
            f'overlap_weights/state_{i}': overlap_weights[:, i, :].mean()
            for i in range(systems.max_num_states)
        }
        aux_data |= mean_ratio_aux

        if reweight_overlap_mean:
            assert reweighting_factor is not None
            cotangent = jnp.where(
                reweighting_factor > 0,
                cotangent / reweighting_factor,
                0.0,
            )
            cotangent = convert_reweighted_tensor_to_unreweighted(cotangent, systems)

        return (cotangent, aux_data), systems, state

    def _mean_ratio(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        reweighting_factor: ReweightingFactor | None = None,
        sample_mask: SampleMask | None = None,
        *,
        reweight_overlap_mean: bool,
    ) -> tuple[
        Float[Array, '(sample_src*walker) n_mol_ids state state']
        | Float[Array, 'walker n_mol_ids state state'],
        MeanRatio,
        dict,
    ]:
        if reweight_overlap_mean:
            assert reweighting_factor is not None, (
                'Reweighting factor must be provided for reweighted overlap computation.'
            )
            return self._reweighted_mean_ratio(
                params,
                systems,
                reweighting_factor,
                sample_mask,
            )

        return self._unreweighted_mean_ratio(params, systems)

    def _reweighted_mean_ratio(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        reweighting_factor: ReweightingFactor,
        sample_mask: SampleMask | None = None,
    ) -> tuple[
        Float[Array, '(sample_src*walker) n_mol_ids state state'],
        MeanRatio,
        dict,
    ]:
        logpsis, signs = compute_logpsi(self.wave_function, params, systems)
        normalizer_ratios = get_normalizing_constant_ratios(systems)
        n_states = logpsis.shape[-1]

        # Expectation of psi ratios
        logpsis = einops.rearrange(
            logpsis,
            'walker mol sample density -> (sample walker) mol density',
        )
        signs = einops.rearrange(
            signs,
            'walker mol sample density -> (sample walker) mol density',
        )

        # numerical stability
        stability_constant = jnp.max(logpsis, axis=-1, keepdims=True)
        logpsis = logpsis - stability_constant

        # normalizer_ratio[..., i, s] = c_i / c_s
        # psis[..., s] = :math:`\log(\psi_s(r \sim \psi_{mix}))`
        # We want to sum over c_s and \psi_s
        log_normalizer = -jax.nn.logsumexp(
            (2 * logpsis)[..., None, :] + normalizer_ratios,  # insert i dim
            axis=-1,
        )
        # normalizer now has shape (total_samples, mol, state)
        # where i is the last axis

        log_ratio = logpsis[..., None] + logpsis[..., None, :] + log_normalizer[..., None]
        sign_ratio = signs[..., None] * signs[..., None, :]
        # add axis for j in the normalizer
        psi_ratio = jnp.exp(log_ratio) * sign_ratio
        # (total_samples, mol, state, state)
        if sample_mask is None:
            sample_mask = jnp.ones_like(psi_ratio, dtype=bool)
        else:
            sample_mask = unpack_reweighted_tensor(sample_mask, systems)
            sample_mask = einops.rearrange(
                sample_mask,
                'walker n_mol_ids sample_src density -> (sample_src walker) n_mol_ids density 1',
            )

        reweighting_factor = unpack_reweighted_tensor(reweighting_factor, systems)
        reweighting_factor = einops.rearrange(
            reweighting_factor,
            'walker n_mol_ids sample_src density -> (sample_src walker) n_mol_ids density 1',
        )

        psi_ratio = self.clipping(
            psi_ratio,
            sample_mask,
            reweighting_factor,
            data_is_reweighted=True,
        )

        mean_ratio = n_states * weighted_mean(
            psi_ratio,
            sample_mask,
            reweighting_factor,
            data_is_reweighted=True,
        )
        overlap = mean_ratio * mean_ratio.mT

        grad_contributions = (psi_ratio * mean_ratio.mT) - jnp.exp(
            2.0 * logpsis + log_normalizer,
        )[..., None] * overlap
        grad_contributions *= n_states

        return (
            grad_contributions,
            mean_ratio,
            {
                'mean_normalizer': pmean_if_pmap(jnp.exp(log_normalizer).mean()),
                'max_normalizer': pmax_if_pmap(jnp.exp(log_normalizer).max()),
                'median_normalizer': pmean_if_pmap(jnp.median(jnp.exp(log_normalizer))),
                'stability_constant': pmean_if_pmap(stability_constant.mean()),
                'max_stability_constant': pmax_if_pmap(stability_constant.max()),
                'min_stability_constant': pmin_if_pmap(stability_constant.min()),
            },
        )

    def _unreweighted_mean_ratio(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
    ) -> tuple[Float[Array, 'walker n_mol_ids state state'], MeanRatio, dict]:
        logpsis, signs = compute_logpsi(self.wave_function, params, systems)
        # logpsis[..., i, j] = logpsi_j(r ~ i)

        # Compute psi ratios
        # shifted_logpis = logpsis - pmean_if_pmap(jnp.mean(logpsis, axis=0, keepdims=True))
        log_ratio = logpsis - jnp.diagonal(logpsis, axis1=-2, axis2=-1)[..., None]
        # log_ratio[..., i, j] = logpsi_j(r ~ i) - logpsi_i(r ~ i)
        sign_ratio = signs * jnp.diagonal(signs, axis1=-2, axis2=-1)[..., None]
        psi_ratio = sign_ratio * jnp.exp(log_ratio)
        psi_ratio = self.clipping(psi_ratio)

        mean_ratio = weighted_mean(psi_ratio)
        grad_contributions = (psi_ratio - mean_ratio) * mean_ratio.mT

        return grad_contributions, mean_ratio, {}

    def unweighted_loss(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
    ) -> Float[Array, '']:
        """Computes the unweighted loss of the overlap penalty.
        Only for debugging purposes."""

        dummy_weights = jnp.tril(
            jnp.ones(
                (systems.n_unique_mols, systems.max_num_states, systems.max_num_states),
                dtype=jnp.float32,
            ),
            k=-1,
        )
        overlap = self.pairwise_overlap(
            params,
            systems,
        )

        return _compute_overlap_loss(overlap, dummy_weights)

    def pairwise_overlap(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
    ) -> Overlap:
        mean_ratio = self._mean_ratio(
            params,
            systems,
            reweight_overlap_mean=False,
        )[1]
        overlap = _compute_mean_overlap(mean_ratio)
        return overlap

    def _effective_sample_size(
        self,
        sample_mask: SampleMask,
        reweighting_factor: ReweightingFactor,
        systems: Systems,
    ) -> dict[str, Float[Array, '']]:
        masked_weights = jnp.where(sample_mask, reweighting_factor, 0.0)
        weights_sum = jnp.asarray(psum_if_pmap(masked_weights.sum(0)))
        weights_sq_sum = jnp.asarray(psum_if_pmap((masked_weights**2).sum(0)))
        ess_per_mol = jnp.where(
            weights_sq_sum > 0,
            (weights_sum**2) / weights_sq_sum,
            jnp.zeros_like(weights_sq_sum),
        )
        total_samples = psum_if_pmap(
            jnp.asarray(masked_weights.shape[0], dtype=jnp.float32)
        )
        ess_per_mol = ess_per_mol / jnp.maximum(total_samples, 1.0)

        mol_ids = np.asarray(systems.mol_ids)
        ess_unsegmented = unsegment_axis(ess_per_mol, mol_ids)
        aux: dict[str, Float[Array, '']] = {'ESS': ess_per_mol.mean()}
        aux |= {
            f'ESS/structure_{i}/state_{j}': ess_unsegmented[i, j].mean()
            for i in np.unique(mol_ids)
            for j in range(ess_unsegmented.shape[1])
        }
        return aux


def _compute_overlap_loss(
    overlap: Overlap,
    overlap_weights: OverlapWeights,
) -> Float[Array, '']:
    loss = overlap**2 * overlap_weights
    # Sum over states, mean over molecules
    # Since we only want the sum over i < j, we can either sum the upper triangle or divide by 2
    # (notice that the diagonal is zero and overlap is symmetric)
    return jnp.sum(loss, axis=(1, 2)).mean()


def _compute_mean_overlap(
    mean_ratio: MeanRatio,
) -> Overlap:
    # Symmetrize the overlap
    # Since we are only using MCMC samples for the computation, we can't guarantee the resulting overlap to be
    # in [0, 1]. Thus we clip the values.
    return jnp.sign(mean_ratio) * jnp.sqrt(
        jnp.clip(mean_ratio * jnp.transpose(mean_ratio, (0, 2, 1)), 0.0),
    )
