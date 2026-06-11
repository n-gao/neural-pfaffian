from collections.abc import Callable, Sequence
from typing import NamedTuple, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode, field
from jaxtyping import Array, ArrayLike, Float, Integer

from neural_pfaffian.nn.wave_function import (
    GeneralizedLogAmplitude,
    LogAmplitude,
    WaveFunctionParameters,
)
from neural_pfaffian.systems import Electrons, Systems, chunk_nuclei
from neural_pfaffian.utils.jax_utils import pmean_if_pmap, pvary
from neural_pfaffian.utils.segment_utils import segment_argmin, segment_sum

PMove = Float[Array, 'n_mols']
Width = Float[ArrayLike, 'n_mols']
ProposalRatio = Float[Array, '... n_mols']
type LogDensity = LogAmplitude

_WIDTH_KEY = 'mcmc_width'
_WIDTH_KEY_LANGEVIN = _WIDTH_KEY + '_langevin'


class ProposalFn(Protocol):
    def __call__(
        self,
        i: Integer[Array, ''],
        key: jax.Array,
        electrons: Electrons,
    ) -> tuple[Electrons, ProposalRatio]: ...


class WidthSchedulerState(NamedTuple):
    width: Width
    pmoves: Float[PMove, 'n_mols steps']
    i: Integer[Array, ' n_mols']


class InitWidthState(Protocol):
    def __call__(self, n_mols: int) -> WidthSchedulerState: ...


class UpdateWidthState(Protocol):
    def __call__(
        self,
        state: WidthSchedulerState,
        pmove: PMove,
    ) -> WidthSchedulerState: ...


class WidthScheduler(NamedTuple):
    init: InitWidthState
    update: UpdateWidthState


def make_width_scheduler(
    init_width: Width,
    window_size: int = 20,
    target_pmove: float = 0.525,
    error: float = 0.025,
) -> WidthScheduler:
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
            pm_mean < target_pmove - error,
            state.width / 1.1,
            state.width,
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
        i: Integer[Array, ''],
        key: jax.Array,
        electrons: Electrons,
    ):
        block_idx = i % blocks
        eps = jax.random.normal(
            key,
            (*electrons.shape[:-2], n_updates, 3),
            electrons.dtype,
        )
        eps *= update_widths
        ratio = jnp.ones((*electrons.shape[:-2], systems.n_mols), dtype=electrons.dtype)
        return electrons.at[..., update_idx[block_idx], :].add(eps, mode='drop'), ratio

    return block_update_proposal


def make_nonlocal_update_proposal(
    systems: Systems,
    nonlocal_step_width: float,
) -> ProposalFn:
    n_mols = systems.n_mols
    n_elec_per_mol = np.array(systems.n_elec_by_mol, dtype=int)
    n_nuc_per_mol = np.array(systems.n_nuc_by_mol, dtype=int)
    e_offsets = np.cumulative_sum(n_elec_per_mol, include_initial=True)[:-1]
    n_offsets = np.cumulative_sum(n_nuc_per_mol, include_initial=True)[:-1]
    e_least_common_multiple = np.lcm.reduce(n_elec_per_mol)

    p = jnp.asarray(systems.flat_charges, dtype=jnp.float32)
    p /= segment_sum(p, systems.nuclei_molecule_mask, systems.n_mols)[
        systems.nuclei_molecule_mask
    ]
    pdf = jnp.vectorize(
        jax.scipy.stats.multivariate_normal.pdf,
        signature='(n),(n),(n,n)->()',
    )
    batch_segment_sum = jax.vmap(segment_sum, in_axes=(0, None, None))

    def nonlocal_update_proposal(
        i: Integer[Array, ''],
        key: jax.Array,
        electrons: Electrons,
    ):
        key, key_el, key_nuc = jax.random.split(key, 3)
        idx_el = jax.random.randint(key_el, (n_mols,), 0, e_least_common_multiple)
        idx_el = e_offsets + idx_el % n_elec_per_mol

        idx_nuc = []
        for (_, nuclei, __), prob in zip(
            systems.iter_grouped_molecules(),
            systems.group(p, chunk_nuclei),
            strict=False,
        ):
            n_sub, n_nucs = nuclei.shape[:2]
            key_nuc, subkey = jax.random.split(key_nuc)
            idx_nuc.append(
                jax.random.choice(subkey, jnp.arange(n_nucs), (n_sub,), p=prob[0]),
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
        p_fwd = batch_segment_sum(new_pdfs * p, mask, systems.n_mols)
        p_bwd = batch_segment_sum(old_pdfs * p, mask, systems.n_mols)
        ratio = p_bwd / p_fwd
        return new_electrons, ratio

    return nonlocal_update_proposal


def make_langevin_update_proposal(
    systems: Systems,
    logprob_fn: Callable[[Electrons], LogDensity],
    width: Width,
) -> ProposalFn:
    @jax.vmap
    @jax.grad
    def quantum_force_closure(electrons: Electrons):
        return jnp.sum(0.5 * logprob_fn(electrons))

    update_width = jnp.repeat(width, np.array(systems.n_elec_by_mol))[None, :, None]

    def clean_force(systems: Systems, electrons: Electrons):
        # Find closest nuclei for each electron
        dist = systems.elec_nuc_dists  # (..., elec_nuc, 4)
        closest_pair_idx = jax.vmap(segment_argmin, in_axes=(0, None, None))(
            dist[..., -1],
            systems.elec_nuc_idx[0],
            systems.n_elec,
        )  # (..., n_elec)
        nuc_pair_idx = systems.elec_nuc_idx[1]  # (elec_nuc,)
        closest_nuc_idx = jnp.take(nuc_pair_idx, closest_pair_idx)  # (..., n_elec)
        closest_charge = jnp.array(systems.flat_charges)[closest_nuc_idx]  # (..., n_elec)
        idx = jnp.broadcast_to(
            closest_pair_idx[..., :, None],
            (*closest_pair_idx.shape, 4),
        )
        closest_dist = jnp.take_along_axis(dist, idx, axis=-2)  # (..., n_elec, 4)

        # Compute the force
        qf = quantum_force_closure(electrons)  # Shape of electrons

        # Compute crossover parameter
        eps = jnp.finfo(qf.dtype).eps
        z = closest_dist[..., :3]
        z_unit = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
        qf_unit = qf / jnp.clip(jnp.linalg.norm(qf, axis=-1, keepdims=True), eps, None)
        Z2z2 = closest_charge**2 * closest_dist[..., -1] ** 2
        crossover = (1 + jnp.sum(qf_unit * z_unit, axis=-1)) / 2 + Z2z2 / (
            10 * (4 + Z2z2)
        )

        av2tau = crossover * jnp.sum(qf**2, axis=-1) * update_width[..., 0]
        factor = 2 / (jnp.sqrt(1 + 2 * av2tau) + 1)
        qf = factor[..., None] * qf
        norm_factor = jnp.minimum(
            1.0,
            jnp.sqrt(closest_dist[..., -1])
            / (
                update_width[..., 0]
                * jnp.clip(
                    jnp.linalg.norm(qf, axis=-1),
                    eps,
                    None,
                )
            ),
        )
        qf = qf * norm_factor[..., None]

        return qf

    def proposal(
        i: Integer[Array, ''],
        key: jax.Array,
        electrons: Electrons,
    ) -> tuple[Electrons, ProposalRatio]:
        # Compute the quantum force
        qf = clean_force(systems, electrons)  # Cleaned quantum force

        # Sample a random noise vector
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, electrons.shape, electrons.dtype)
        proposal = electrons + update_width * qf + jnp.sqrt(update_width) * noise

        qf_prop = clean_force(systems, proposal)

        log_ratio = jnp.sum(
            (qf + qf_prop) * ((electrons - proposal) + update_width / 2 * (qf - qf_prop)),
            axis=-1,
        )
        log_ratio = jax.vmap(segment_sum, in_axes=(0, None, None))(
            log_ratio,
            systems.electron_molecule_mask,
            systems.n_mols,
        )
        ratio = jnp.exp(log_ratio)

        return proposal, ratio

    return proposal


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
            jnp.repeat(cond, mol_to_elecs, axis=-1)[..., None],
            new_electrons,
            electrons,
        )
        log_prob = jnp.where(cond, log_prob_new, log_prob)
        num_accepts += cond
        return key, new_electrons, log_prob, num_accepts

    return mh_update


S = TypeVar('S', bound=Systems)


class MetropolisHastings(PyTreeNode):
    wave_function: GeneralizedLogAmplitude = field(pytree_node=False)
    steps: int = field(pytree_node=False)
    init_width: Width
    window_size: int = field(pytree_node=False)
    target_pmove: float
    error: float
    blocks: int = field(pytree_node=False)
    nonlocal_steps: int = field(pytree_node=False)
    nonlocal_step_width: float = field(pytree_node=False)

    # Langevin parameters
    langevin_steps: int = field(pytree_node=False, default=0)
    langevin_init_width: Width = field(pytree_node=False, default=1.0)

    def init_systems(self, key: Array, systems: S) -> S:
        if _WIDTH_KEY not in systems.mol_data:
            systems = systems.set_mol_data(
                _WIDTH_KEY,
                self.width_scheduler.init(systems.n_mols),
            )
        if _WIDTH_KEY_LANGEVIN not in systems.mol_data:
            systems = systems.set_mol_data(
                _WIDTH_KEY_LANGEVIN,
                self.langevin_width_scheduler.init(systems.n_mols),
            )
        return systems

    def __call__(
        self,
        key: Array,
        params: WaveFunctionParameters,
        systems: S,
    ) -> tuple[S, dict[str, Float[Array, '']]]:
        # Fix the per molecule parameters and do not recompute them
        wf_fixed = self.wave_function.fix_structure(params, systems)
        batch_size = systems.electrons.shape[0]

        def logprob_fn(electrons: Electrons) -> LogDensity:
            return 2 * wf_fixed.apply(params, systems.replace(electrons=electrons))

        batch_logprob_fn = jax.vmap(logprob_fn)

        log_probs = batch_logprob_fn(systems.electrons)
        aux_data = {}

        # Regular local MCMC moves
        assert self.blocks >= 1, 'Number of blocks must be at least 1'
        n_local_steps = self.steps * self.blocks
        assert any(
            (n_local_steps > 0, self.nonlocal_steps > 0, self.langevin_steps > 0),
        ), 'No MCMC steps specified'

        electrons = systems.electrons
        if n_local_steps > 0:
            width_state = systems.get_mol_data(_WIDTH_KEY)
            proposal = make_block_update_proposal(width_state.width, systems, self.blocks)
            mh_update = make_mh_update(batch_logprob_fn, proposal, systems.n_elec_by_mol)
            num_accepts = jnp.zeros(log_probs.shape, dtype=jnp.int32)
            num_accepts = pvary(num_accepts)

            key, electrons, log_probs, num_accepts = jax.lax.scan(
                lambda x, i: (mh_update(i, *x), None),
                (key, electrons, log_probs, num_accepts),
                jnp.arange(n_local_steps),
            )[0]

            # Update local update width
            pmove = jnp.sum(num_accepts, axis=0) / (n_local_steps * batch_size)
            pmove = pmean_if_pmap(pmove)
            width_state = self.width_scheduler.update(width_state, pmove)
            systems = systems.set_mol_data(_WIDTH_KEY, width_state)
            aux_data['pmove'] = pmove.mean()
            aux_data['width'] = jnp.mean(width_state.width)

        # Langevin moves
        if self.langevin_steps > 0:
            width_state = systems.get_mol_data(_WIDTH_KEY_LANGEVIN)
            proposal = make_langevin_update_proposal(
                systems,
                logprob_fn,
                width_state.width,
            )
            mh_update = make_mh_update(batch_logprob_fn, proposal, systems.n_elec_by_mol)
            num_accepts = jnp.zeros(log_probs.shape, dtype=jnp.int32)
            num_accepts = pvary(num_accepts)

            key, electrons, log_probs, num_accepts = jax.lax.scan(
                lambda x, i: (mh_update(i, *x), None),
                (key, electrons, log_probs, num_accepts),
                jnp.arange(self.langevin_steps),
            )[0]
            # Update Langevin update width
            pmove = jnp.sum(num_accepts, axis=0) / (self.langevin_steps * batch_size)
            pmove = pmean_if_pmap(pmove)
            width_state = self.langevin_width_scheduler.update(width_state, pmove)
            systems = systems.set_mol_data(_WIDTH_KEY_LANGEVIN, width_state)
            aux_data['pmove_langevin'] = pmove.mean()
            aux_data['width_langevin'] = jnp.mean(width_state.width)

        # Nonlocal moves
        if self.nonlocal_steps > 0:
            proposal = make_nonlocal_update_proposal(systems, self.nonlocal_step_width)
            mh_update = make_mh_update(batch_logprob_fn, proposal, systems.n_elec_by_mol)
            num_accepts = jnp.zeros(log_probs.shape, dtype=jnp.int32)
            num_accepts = pvary(num_accepts)
            key, electrons, _, num_accepts = jax.lax.scan(
                lambda x, i: (mh_update(i, *x), None),
                (key, electrons, log_probs, num_accepts),
                jnp.arange(self.nonlocal_steps),
            )[0]
            pmove = jnp.sum(num_accepts, axis=0) / (self.nonlocal_steps * batch_size)
            aux_data['pmove_nonlocal'] = pmean_if_pmap(pmove.mean())

        return systems.replace(electrons=electrons), aux_data

    @property
    def width_scheduler(self):
        return make_width_scheduler(
            self.init_width,
            self.window_size,
            self.target_pmove,
            self.error,
        )

    @property
    def langevin_width_scheduler(self):
        return make_width_scheduler(
            self.langevin_init_width,
            self.window_size,
            self.target_pmove,
            self.error,
        )
