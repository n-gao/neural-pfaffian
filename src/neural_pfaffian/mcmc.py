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
from neural_pfaffian.systems import Electrons, Systems
from neural_pfaffian.utils.jax_utils import pmean_if_pmap

PMove = Float[Array, 'n_mols']
Width = Float[ArrayLike, 'n_mols']
type LogDensity = LogAmplitude

_WIDTH_KEY = 'mcmc_width'


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


def make_mh_update(
    logprob_fn: Callable[[Electrons], LogDensity],
    width: Width,
    elec_per_mol: Sequence[int],
    blocks: int,
):
    mol_to_elecs = np.asarray(elec_per_mol)
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

    def mh_update(
        i: Integer[Array, ''],
        key: jax.Array,
        electrons: Electrons,
        log_prob: LogDensity,
        num_accepts: Integer[Array, ''],
    ):
        block_idx = i % blocks
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(
            subkey, (*electrons.shape[:-2], n_updates, 3), electrons.dtype
        )
        eps *= update_widths
        new_electrons = electrons.at[..., update_idx[block_idx], :].add(eps, mode='drop')
        log_prob_new = logprob_fn(new_electrons)
        ratio = log_prob_new - log_prob

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
    wave_function: GeneralizedWaveFunction
    steps: int = field(pytree_node=False)
    init_width: Width
    window_size: int = field(pytree_node=False)
    target_pmove: float
    error: float
    blocks: int

    def init_systems(self, key: Array, systems: S) -> S:
        if _WIDTH_KEY not in systems.mol_data:
            return systems.set_mol_data(
                _WIDTH_KEY, self.width_scheduler.init(systems.n_mols)
            )
        return systems

    def __call__(self, key: Array, params: WaveFunctionParameters, systems: S) -> S:
        # Fix the per molecule parameters and do not recompute them
        wf_fixed = self.wave_function.fix_structure(params, systems)
        # Get current width
        width_state = systems.get_mol_data(_WIDTH_KEY)
        width = width_state.width
        batch_size = systems.electrons.shape[0]

        @jax.vmap
        def logprob_fn(electrons: Electrons) -> LogDensity:
            return 2 * wf_fixed.apply(params, systems.replace(electrons=electrons))

        mh_update = make_mh_update(logprob_fn, width, systems.n_elec_by_mol, self.blocks)
        log_probs = logprob_fn(systems.electrons)
        num_accepts = jnp.zeros(log_probs.shape, dtype=jnp.int32)

        key, electrons, _, num_accepts = jax.lax.scan(
            lambda x, i: (mh_update(i, *x), None),
            (key, systems.electrons, log_probs, num_accepts),
            jnp.arange(self.steps * self.blocks),
        )[0]

        pmove = jnp.sum(num_accepts, axis=0) / (self.steps * batch_size)
        pmove = pmean_if_pmap(pmove)
        width_state = self.width_scheduler.update(width_state, pmove)
        return systems.replace(electrons=electrons).set_mol_data(_WIDTH_KEY, width_state)

    @property
    def width_scheduler(self):
        return make_width_scheduler(
            self.init_width, self.window_size, self.target_pmove, self.error
        )
