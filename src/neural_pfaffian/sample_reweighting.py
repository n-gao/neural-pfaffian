import einops
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from neural_pfaffian.clipping import Masking
from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    WaveFunctionParameters,
)
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils.jax_utils import (
    jit,
    pmax_if_pmap,
    pmean_if_pmap,
    pmin_if_pmap,
    psum_if_pmap,
    vmap,
)
from neural_pfaffian.utils.segment_utils import unsegment_axis
from neural_pfaffian.utils.tree_utils import tree_to_dtype

Signs = Float[Array, 'n_walker_per_mol n_unique_mols sample_src density']
LogPsis = Float[Array, 'n_walker_per_mol n_unique_mols sample_src density']
NormalizerRatio = Float[Array, 'n_unique_mols numerator denominator']
r"""Represents :math:`C_{ij} = \log \frac{c_i}{c_j} = \int\psi_i^2 - {\int\psi_j^2}`"""
ReweightingFactor = Float[Array, '(sample_src*walker) n_mols']
SampleMask = Bool[Array, '(sample_src*walker) n_mols']

LOG_NORMALIZER_CONSTANTS_KEY = 'log_norm_constants'


@jit
def update_normalizing_constant_ratios(
    wf: GeneralizedWaveFunction,
    params: WaveFunctionParameters,
    systems: Systems,
    *,
    iterations: int = 10,
    max_update_factor: float = 2,
) -> tuple[Systems, dict]:
    """Given logpsis, computes the ratios of the normalizing constants of the wave functions.

    For details on the derivation see:
    https://www.jstor.org/stable/24306045

    Returns:
        Systems: The systems with updated normalizing constant ratios.
    """

    out_dtypes = jax.tree.map(lambda x: x.dtype, systems)
    systems = tree_to_dtype(systems, jnp.float64)

    logpsis, _ = compute_logpsi(wf, params, systems)
    dtype = logpsis.dtype

    # Rearrange to (mol, sample, density, walker) and work in log-densities directly
    local_logdens = 2.0 * einops.rearrange(
        logpsis,
        'walker mol sample density -> mol sample density walker',
    )

    # Per-device walker count and global average helpers
    local_n_walker = local_logdens.shape[-1]
    global_n_walker = psum_if_pmap(jnp.array(local_n_walker))

    n_states = local_logdens.shape[2]
    n_unique_mol = local_logdens.shape[0]

    eps = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)

    def iteration(carry, _):
        r, *_ = carry
        # r: (mol, density), positive ratios
        log_r = jnp.log(jnp.maximum(r, eps))  # (mol, density)
        log_r_b = log_r[:, None, :, None]  # (mol, sample, density, walker)

        # log_qmix(s,w) = log sum_j exp( log r_j + 2 logpsi_{s,j,w} )
        log_qmix = jax.nn.logsumexp(
            log_r_b + local_logdens,
            axis=2,
        )  # (mol, sample, walker)
        q_mix_min = jnp.min(log_qmix)
        q_mix_max = jnp.max(log_qmix)

        # Responsibilities: density_j / qmix, computed stably
        R = jnp.exp(
            local_logdens - log_qmix[:, :, None, :],
        )  # (mol, sample, density, walker)

        # Off-diagonals of B: -E_w[R]  (mean over walkers across devices)
        local_B = -R.sum(axis=3)  # (mol, sample, density)
        B = psum_if_pmap(local_B) / global_n_walker  # (mol, sample, density)

        # Diagonal correction: bii_i = sum_{s != i} E_w[ R_{s,i} ]
        sum_over_sample = R.sum(axis=1)  # (mol, density, walker)
        diag_R = R[
            :,
            jnp.arange(n_states),
            jnp.arange(n_states),
            :,
        ]  # (mol, density, walker)
        local_bii = (sum_over_sample - diag_R).sum(axis=2)  # (mol, density)
        bii = psum_if_pmap(local_bii) / global_n_walker  # (mol, density)

        # Place diagonal
        B = B.at[:, jnp.arange(n_states), jnp.arange(n_states)].set(
            bii,
        )  # (mol, sample, density)

        # Solve for r with r_0 = 1 convention
        b = -B[:, 1:, :1]  # (mol, n-1, 1)
        B_sub = B[:, 1:, 1:]  # (mol, n-1, n-1)

        # Small ridge for numerical stability
        ridge = jnp.asarray(1e-8 if dtype == jnp.float64 else 1e-5, dtype=dtype)
        I = jnp.eye(n_states - 1, dtype=dtype)[None, :, :]
        B_sub = B_sub + ridge * I

        r_tail = jnp.linalg.solve(B_sub, b)[..., 0]  # (mol, n-1)
        r_new = jnp.concatenate(
            [jnp.ones((*r_tail.shape[:-1], 1), dtype=dtype), r_tail],
            axis=-1,
        )
        r_new = jnp.maximum(r_new, eps)  # enforce positivity

        delta_r = jnp.linalg.norm(r - r_new)

        return (r_new, B_sub, q_mix_min, q_mix_max), delta_r

    # Initialize r: (mol, density)
    log_r_old = systems.get_mol_data(LOG_NORMALIZER_CONSTANTS_KEY)  # (n_mols,)
    log_r_old = unsegment_axis(
        log_r_old,
        np.array(systems.mol_id_groups),
        axis=0,
        indices_are_grouped=True,
        num_segments=systems.n_unique_mols,
    )  # (n_unique_mols, states)
    r0 = jnp.exp(log_r_old)
    B_sub0 = jnp.zeros((n_unique_mol, n_states - 1, n_states - 1), dtype=dtype)
    q_mix0 = jnp.zeros((), dtype=dtype)
    (r_new, B_sub, q_mix_min, q_mix_max), delta_r_norm = jax.lax.scan(
        iteration,
        (r0, B_sub0, q_mix0, q_mix0),
        None,
        length=iterations,
    )
    log_r_new = jnp.log(jnp.maximum(r_new, eps))

    # Clipped update
    log_r_new = log_r_old + jnp.clip(
        log_r_new - log_r_old,
        -jnp.log(max_update_factor),
        jnp.log(max_update_factor),
    )

    # Ensure r_0 = 1
    log_r_new = log_r_new - log_r_new[:, :1]

    log_r_new = log_r_new[systems.mol_id_groups, systems.excitations]
    systems = systems.set_mol_data(LOG_NORMALIZER_CONSTANTS_KEY, log_r_new)

    # Logging
    s = jnp.linalg.svd(B_sub, compute_uv=False)
    s_min = s[..., -1]
    s_max = s[..., 0]
    cond = s_max / s_min
    aux = {
        'normalizing_constants/log_r_min': pmin_if_pmap(jnp.min(log_r_new)),
        'normalizing_constants/log_r_max': pmax_if_pmap(jnp.max(log_r_new)),
        'normalizing_constants/B_sub_s_min': jnp.min(s_min),
        'normalizing_constants/B_sub_s_max': jnp.max(s_max),
        # condition number of the linear system; if ill-conditioned may lead to wrong ratios
        'normalizing_constants/B_sub_cond': jnp.max(cond),
        'normalizing_constants/B_sub_min_norm': jnp.min(
            jnp.linalg.norm(B_sub, axis=(-2, -1)),
        ),
        'normalizing_constants/B_sub_max_norm': jnp.max(
            jnp.linalg.norm(B_sub, axis=(-2, -1)),
        ),
        # indicates whether all psi are tiny for a sample; when it drops the denom in reweighting explodes
        'normalizing_constants/log_q_mix_min': pmin_if_pmap(q_mix_min),
        'normalizing_constants/log_q_mix_max': pmax_if_pmap(q_mix_max),
    }

    aux |= {
        f'normalizing_constants/delta_r_norm_it_{i}': delta
        for i, delta in enumerate(delta_r_norm)
    }
    own_logpsis = pmean_if_pmap(
        jnp.mean(jnp.diagonal(logpsis, axis1=-2, axis2=-1), axis=0),
    )
    aux |= {
        f'normalizing_constants/mean_logpsi/structure_{i}/state_{j}': own_logpsis[i, j]
        for i in np.unique(systems.mol_ids)
        for j in range(own_logpsis.shape[1])
    }

    systems = jax.tree.map(jax.lax.convert_element_type, systems, out_dtypes)

    return systems, aux


def get_normalizing_constant_ratios(systems: Systems) -> NormalizerRatio:
    log_r = systems.get_mol_data(LOG_NORMALIZER_CONSTANTS_KEY)  # (n_mols,)
    log_r = unsegment_axis(
        log_r,
        np.array(systems.mol_id_groups),
        axis=0,
        indices_are_grouped=True,
        num_segments=systems.n_unique_mols,
    )  # (n_unique_mols, states)
    log_pairwise = log_r[..., None, :] - log_r[..., :, None]
    return log_pairwise


@jit
def compute_logpsi(
    wf: GeneralizedWaveFunction,
    params: WaveFunctionParameters,
    systems: Systems,
) -> tuple[LogPsis, Signs]:
    """
    Computes the logarithm of the wave function values and their signs for all state and electron set combinations.
    """

    @vmap(in_axes=(systems.electron_vmap,))
    def _compute_single_logpsi(systems: Systems):
        # Fix the reparameterization of the wave function
        wf_fixed = wf.fix_structure(params, systems)

        # Cache excitation independent quantities
        embeddings = wf_fixed.embedding(params, systems)
        core_orbitals = wf_fixed.core_orbitals(params, systems, embeddings)

        @vmap(in_axes=0, out_axes=-1)
        def _compute_state(i):
            excited_systems = systems.set_global_excitation(i)
            orbitals = wf_fixed.apply_excitation(params, excited_systems, core_orbitals)
            sign, logpsi = wf_fixed.signed(params, excited_systems, orbitals, embeddings)
            return sign, logpsi

        states = jnp.arange(systems.max_num_states)
        signs, logpsis = _compute_state(states)
        # (n_mols, wf_state) n_mols are segmented by systems.mol_ids

        logpsis = unsegment_axis(
            logpsis,
            np.array(systems.mol_id_groups),
            axis=0,
            indices_are_grouped=True,
            num_segments=systems.n_unique_mols,
        )
        signs = unsegment_axis(
            signs,
            np.array(systems.mol_id_groups),
            axis=0,
            indices_are_grouped=True,
            num_segments=systems.n_unique_mols,
        )
        # (n_unique_mols, sample_src, wf_state)

        return logpsis, signs

    return _compute_single_logpsi(systems)


@jit
def compute_reweighting_factor(
    wf: GeneralizedWaveFunction,
    params: WaveFunctionParameters,
    systems: Systems,
) -> ReweightingFactor:
    r"""
    Computes the reweighting factor.
    :math:`W_i = N_s \frac{|\psi_i|^2}{\sum_j |\psi_j|^2 \frac{c_i}{c_j}}`
    """
    logpsis, _ = compute_logpsi(wf, params, systems)
    logdens = 2.0 * logpsis

    logdens = einops.rearrange(
        logdens,
        'walker mol sample_src density -> (sample_src walker) mol density',
    )

    log_ratios = get_normalizing_constant_ratios(systems)  # (mol, i, j)

    log_denom = -jax.nn.logsumexp(
        logdens[..., None, :] + log_ratios,
        axis=-1,
    )  # (sample_src*walker, mol, i)
    log_factor = logdens + log_denom
    return (jnp.exp(log_factor) * systems.max_num_states)[
        :,
        systems.mol_id_groups,
        systems.excitations,
    ]


def compute_sample_mask(
    wf: GeneralizedWaveFunction,
    params: WaveFunctionParameters,
    systems: Systems,
    masking: Masking,
    reweighting_factor: ReweightingFactor,
    reweight_overlap_mean: bool,
) -> tuple[SampleMask, dict]:
    if reweight_overlap_mean:
        # We need a full sample mask
        logpsis, signs = compute_logpsi(wf, params, systems)
        mask_values = signs * jnp.exp(-(logpsis - pmin_if_pmap(logpsis.min())))
        mask_values = pack_reweighted_tensor(mask_values, systems)
        mask = masking(mask_values, reweighting_factor)
        diagonal_overwrite = jnp.eye(systems.max_num_states, dtype=bool)
        diagonal_overwrite = jnp.broadcast_to(diagonal_overwrite, logpsis.shape)
        diagonal_overwrite = pack_reweighted_tensor(diagonal_overwrite, systems)
        mask = diagonal_overwrite | mask
    else:
        mask = jnp.ones_like(reweighting_factor, dtype=bool)
    return mask, {'effective_samples': psum_if_pmap(mask.sum())}


def unpack_reweighted_tensor(
    x: Float[Array, '(sample_src*walker) n_mols'],
    systems: Systems,
) -> Float[Array, 'walker n_mol_ids sample_src wf_state']:
    """Unpacks a reweighted tensor, where the sample_src is segmented in the batch dimension
    and the density is segmented in the n_mols dimension in a tensor of shape
    (walker, n_mol_ids, sample_src, wf_state)."""
    x = unsegment_axis(
        x,
        np.array(systems.mol_id_groups),
        axis=1,
        indices_are_grouped=True,
        num_segments=systems.n_unique_mols,
    )
    x = einops.rearrange(
        x,
        '(sample_src walker) n_mol_ids wf_state -> walker n_mol_ids sample_src wf_state',
        sample_src=systems.max_num_states,
    )
    return x


def pack_reweighted_tensor(
    x: Float[Array, 'walker n_mol_ids sample_src wf_state'],
    systems: Systems,
) -> Float[Array, '(sample_src*walker) n_mols']:
    """Packs a tensor of shape (walker, n_mol_ids, sample_src, wf_state) into a reweighted
    tensor, where the sample_src is segmented in the batch dimension and the density is
    segmented in the n_mols dimension."""
    x = einops.rearrange(
        x,
        'walker n_mol_ids sample_src wf_state -> (sample_src walker) n_mol_ids wf_state',
    )
    return x[:, systems.mol_id_groups, systems.excitations]


def convert_reweighted_tensor_to_unreweighted(
    x: Float[Array, '(sample_src*walker) n_mols'],
    systems: Systems,
) -> Float[Array, 'walker n_mols']:
    """Takes a reweighted tensor with the sample_src segmented in the batch dimension
    and the density segmented in the n_mols dimension and strips away all entries where
    sample_src and wf_state are not aligned, i.e., returns a tensor as if we didn't have
    reweighting."""
    x = unpack_reweighted_tensor(x, systems)
    x = jnp.diagonal(x, axis1=-2, axis2=-1)
    return x[:, systems.mol_id_groups, systems.excitations]
