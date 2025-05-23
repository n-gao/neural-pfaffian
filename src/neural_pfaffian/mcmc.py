from typing import Callable, NamedTuple, Protocol, Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode, field
from jaxtyping import Array, ArrayLike, Float, Integer

from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    LogAmplitude,
    WaveFunctionParameters,
)
from neural_pfaffian.systems import Electrons, Systems, chunk_nuclei
from neural_pfaffian.utils.jax_utils import pmean_if_pmap, pvary

PMove = Float[Array, 'n_mols']
Width = Float[ArrayLike, 'n_mols']
ProposalRatio = Float[Array, '... n_mols']
type LogDensity = LogAmplitude

_WIDTH_KEY = 'mcmc_width'


class ProposalFn(Protocol):
    def __call__(
        self, i: Integer[Array, ''], key: jax.Array, electrons: Electrons
    ) -> tuple[Electrons, ProposalRatio]: ...


class WidthSchedulerState(NamedTuple):
    width: Width
    pmoves: Float[PMove, 'n_mols steps']
    i: Integer[Array, ' n_mols']


class InitWidthState(Protocol):
    def __call__(self, n_mols: int) -> WidthSchedulerState: ...


class UpdateWidthState(Protocol):
    def __call__(
        self, state: WidthSchedulerState, pmove: PMove
    ) -> WidthSchedulerState: ...


class WidthScheduler(NamedTuple):
    init: InitWidthState
    update: UpdateWidthState


def make_width_scheduler(
    init_width: Width,
    window_size: int = 20,
    target_pmove: float = 0.525,
    error: float = 0.025,
):
    init_width = jnp.asarray(init_width, dtype=jnp.float32)

    def init(n_mols: int) -> WidthSchedulerState:
        return WidthSchedulerState(
            width=jnp.full((n_mols,), init_width, jnp.float32),
            pmoves=jnp.zeros((n_mols, window_size, *init_width.shape), dtype=jnp.float32),
            i=jnp.zeros((n_mols,), dtype=jnp.int32),
        )

    @jax.jit
    @jax.vmap
    def update(state: WidthSchedulerState, pmove: PMove) -> WidthSchedulerState:
        pmoves = state.pmoves.at[jnp.mod(state.i, window_size)].set(pmove)
        pm_mean = state.pmoves.mean()
        i = state.i + 1

        upd_width = jnp.where(
            pm_mean < target_pmove - error, state.width / 1.1, state.width
        )
        upd_width = jnp.where(pm_mean > target_pmove + error, upd_width * 1.1, upd_width)
        width = jnp.where(
            jnp.mod(i, window_size) == 0,
            upd_width,
            state.width,
        )
        return WidthSchedulerState(width=width, pmoves=pmoves, i=i)

    return WidthScheduler(init, update)


def make_block_update_proposal(width: Width, systems: Systems, blocks: int) -> ProposalFn:
    elec_per_mol = systems.n_elec_by_mol
    updates_per_mol = np.ceil(np.array(elec_per_mol) / blocks).astype(int)
    n_updates = sum(updates_per_mol)
    update_mask = np.arange(len(elec_per_mol)).repeat(updates_per_mol)
    update_widths = jnp.asarray(width)[update_mask][:, None]
    # Create index tensor where by default all indices are out of bounds
    out_of_bounds_idx = sum(elec_per_mol)
    update_idx = np.full((blocks, n_updates), dtype=int, fill_value=out_of_bounds_idx)
    # Fill in the indices for each block
    for block in range(blocks):
        for elec_offset, elecs, update_offset, updates in zip(
            np.cumulative_sum(elec_per_mol, include_initial=True),
            elec_per_mol,
            np.cumulative_sum(updates_per_mol, include_initial=True),
            updates_per_mol,
            strict=False,  # the last elements of the cumulative sum are not used
        ):
            idx = np.arange(block * updates, block * updates + updates)
            idx = np.where(idx < elecs, idx, out_of_bounds_idx)
            update_idx[block, update_offset : update_offset + updates] = elec_offset + idx
    update_idx = jnp.asarray(update_idx)

    def block_update_proposal(
        i: Integer[Array, ''], key: jax.Array, electrons: Electrons
    ):
        block_idx = i % blocks
        eps = jax.random.normal(
            key, (*electrons.shape[:-2], n_updates, 3), electrons.dtype
        )
        eps *= update_widths
        ratio = jnp.ones((*electrons.shape[:-2], systems.n_mols), dtype=electrons.dtype)
        return electrons.at[..., update_idx[block_idx], :].add(eps, mode='drop'), ratio

    return block_update_proposal


def make_nonlocal_update_proposal(
    systems: Systems, nonlocal_step_width: float
) -> ProposalFn:
    n_mols = systems.n_mols
    n_elec_per_mol = np.array(systems.n_elec_by_mol, dtype=int)
    n_nuc_per_mol = np.array(systems.n_nuc_by_mol, dtype=int)
    e_offsets = np.cumulative_sum(n_elec_per_mol, include_initial=True)[:-1]
    n_offsets = np.cumulative_sum(n_nuc_per_mol, include_initial=True)[:-1]
    e_least_common_multiple = np.lcm.reduce(n_elec_per_mol)

    p = jnp.asarray(systems.flat_charges, dtype=jnp.float32)
    p /= jax.ops.segment_sum(p, systems.nuclei_molecule_mask, systems.n_mols)[
        systems.nuclei_molecule_mask
    ]
    pdf = jnp.vectorize(
        jax.scipy.stats.multivariate_normal.pdf, signature='(n),(n),(n,n)->()'
    )
    segment_sum = jax.vmap(jax.ops.segment_sum, in_axes=(0, None, None))

    def nonlocal_update_proposal(
        i: Integer[Array, ''], key: jax.Array, electrons: Electrons
    ):
        key, key_el, key_nuc = jax.random.split(key, 3)
        idx_el = jax.random.randint(key_el, (n_mols,), 0, e_least_common_multiple)
        idx_el = e_offsets + idx_el % n_elec_per_mol

        idx_nuc = []
        for (_, nuclei, __), prob in zip(
            systems.iter_grouped_molecules(), systems.group(p, chunk_nuclei)
        ):
            n_sub, n_nucs = nuclei.shape[:2]
            key_nuc, subkey = jax.random.split(key_nuc)
            idx_nuc.append(
                jax.random.choice(subkey, jnp.arange(n_nucs), (n_sub,), p=prob[0])
            )
        idx_nuc = n_offsets + jnp.concatenate(idx_nuc)[systems.inverse_unique_indices]

        eps = jax.random.normal(key, (*electrons.shape[:-2], n_mols, 3), electrons.dtype)
        eps *= nonlocal_step_width

        r_old = electrons[..., idx_el, :]
        r_new = systems.nuclei[..., idx_nuc, :] + eps
        new_electrons = electrons.at[..., idx_el, :].set(r_new)
        mask = systems.nuclei_molecule_mask

        sigma = nonlocal_step_width**2 * jnp.eye(3, dtype=electrons.dtype)
        new_pdfs = pdf(r_new[..., mask, :], systems.nuclei, sigma)
        old_pdfs = pdf(r_old[..., mask, :], systems.nuclei, sigma)
        p_fwd = segment_sum(new_pdfs * p, mask, systems.n_mols)
        p_bwd = segment_sum(old_pdfs * p, mask, systems.n_mols)
        ratio = p_bwd / p_fwd
        return new_electrons, ratio

    return nonlocal_update_proposal


def make_mh_update(
    logprob_fn: Callable[[Electrons], LogDensity],
    proposal: ProposalFn,
    elec_per_mol: Sequence[int],
):
    mol_to_elecs = np.asarray(elec_per_mol)

    def mh_update(
        i: Integer[Array, ''],
        key: jax.Array,
        electrons: Electrons,
        log_prob: LogDensity,
        num_accepts: Integer[Array, ''],
    ):
        key, subkey = jax.random.split(key)
        new_electrons, ratio = proposal(i, key, electrons)
        log_prob_new = logprob_fn(new_electrons)
        ratio = log_prob_new - log_prob + jnp.log(ratio)

        key, subkey = jax.random.split(key)
        alpha = jnp.log(jax.random.uniform(subkey, log_prob_new.shape))
        cond = ratio > alpha
        new_electrons = jnp.where(
            jnp.repeat(cond, mol_to_elecs, axis=-1)[..., None], new_electrons, electrons
        )
        log_prob = jnp.where(cond, log_prob_new, log_prob)
        num_accepts += cond
        return key, new_electrons, log_prob, num_accepts

    return mh_update


S = TypeVar('S', bound=Systems)


class MetropolisHastings(PyTreeNode):
    wave_function: GeneralizedWaveFunction = field(pytree_node=False)
    steps: int = field(pytree_node=False)
    init_width: Width
    window_size: int = field(pytree_node=False)
    target_pmove: float
    error: float
    blocks: int = field(pytree_node=False)
    nonlocal_steps: int = field(pytree_node=False)
    nonlocal_step_width: float = field(pytree_node=False)

    def init_systems(self, key: Array, systems: S) -> S:
        if _WIDTH_KEY not in systems.mol_data:
            return systems.set_mol_data(
                _WIDTH_KEY, self.width_scheduler.init(systems.n_mols)
            )
        return systems

    def __call__(
        self, key: Array, params: WaveFunctionParameters, systems: S
    ) -> tuple[S, dict[str, Float[Array, '']]]:
        # Fix the per molecule parameters and do not recompute them
        wf_fixed = self.wave_function.fix_structure(params, systems)
        # Get current width
        width_state = systems.get_mol_data(_WIDTH_KEY)
        batch_size = systems.electrons.shape[0]

        @jax.vmap
        def logprob_fn(electrons: Electrons) -> LogDensity:
            return 2 * wf_fixed.apply(params, systems.replace(electrons=electrons))

        log_probs = logprob_fn(systems.electrons)
        aux_data = {}

        # Regular local MCMC moves
        assert self.blocks >= 1, 'Number of blocks must be at least 1'
        n_local_steps = self.steps * self.blocks
        if n_local_steps > 0:
            proposal = make_block_update_proposal(width_state.width, systems, self.blocks)
            mh_update = make_mh_update(logprob_fn, proposal, systems.n_elec_by_mol)
            num_accepts = jnp.zeros(log_probs.shape, dtype=jnp.int32)
            num_accepts = pvary(num_accepts)

            key, electrons, log_probs, num_accepts = jax.lax.scan(
                lambda x, i: (mh_update(i, *x), None),
                (key, systems.electrons, log_probs, num_accepts),
                jnp.arange(n_local_steps),
            )[0]

            # Update local update width
            pmove = jnp.sum(num_accepts, axis=0) / (n_local_steps * batch_size)
            pmove = pmean_if_pmap(pmove)
            width_state = self.width_scheduler.update(width_state, pmove)
            aux_data['pmove'] = pmove.mean()
            aux_data['width'] = jnp.mean(width_state.width)

        # Nonlocal moves
        if self.nonlocal_steps > 0:
            proposal = make_nonlocal_update_proposal(systems, self.nonlocal_step_width)
            mh_update = make_mh_update(logprob_fn, proposal, systems.n_elec_by_mol)
            num_accepts = jnp.zeros(log_probs.shape, dtype=jnp.int32)
            key, electrons, _, num_accepts = jax.lax.scan(
                lambda x, i: (mh_update(i, *x), None),
                (key, electrons, log_probs, num_accepts),
                jnp.arange(self.nonlocal_steps),
            )[0]
            pmove = jnp.sum(num_accepts, axis=0) / (self.nonlocal_steps * batch_size)
            aux_data['pmove_nonlocal'] = pmean_if_pmap(pmove.mean())

        return systems.replace(electrons=electrons).set_mol_data(
            _WIDTH_KEY, width_state
        ), aux_data

    @property
    def width_scheduler(self):
        return make_width_scheduler(
            self.init_width, self.window_size, self.target_pmove, self.error
        )
