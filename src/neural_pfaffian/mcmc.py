from typing import Callable, NamedTuple, Protocol, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Integer

from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    LogAmplitude,
    WaveFunctionParameters,
)
from neural_pfaffian.systems import Electrons, Systems
from neural_pfaffian.utils.jax_utils import jit, pmean_if_pmap

PMove = Float[Array, 'n_mols']
Width = Float[Array, 'n_mols']
type LogDensity = LogAmplitude


class WidthSchedulerState(NamedTuple):
    width: Width
    pmoves: Float[PMove, ' steps']
    i: Integer[Array, ' n_mols']


class InitWidthState(Protocol):
    def __call__(self, init_width: Width) -> WidthSchedulerState: ...


class UpdateWidthState(Protocol):
    def __call__(
        self, state: WidthSchedulerState, pmove: PMove
    ) -> WidthSchedulerState: ...


class WidthScheduler(NamedTuple):
    init: InitWidthState
    update: UpdateWidthState


def make_width_scheduler(
    window_size: int = 20,
    target_pmove: float = 0.525,
    error: float = 0.025,
):
    @jax.jit
    def init(init_width: Width) -> WidthSchedulerState:
        return WidthSchedulerState(
            width=jnp.array(init_width, dtype=jnp.float32),
            pmoves=jnp.zeros((window_size, *init_width.shape), dtype=jnp.float32),
            i=jnp.zeros((), dtype=jnp.int32),
        )

    @jax.jit
    def update(state: WidthSchedulerState, pmove: PMove) -> WidthSchedulerState:
        pmoves = state.pmoves.at[jnp.mod(state.i, window_size)].set(pmove)
        pm_mean = state.pmoves.mean()
        i = state.i + 1

        upd_width = jnp.where(
            pm_mean < target_pmove - error, state.width / 1.1, state.width
        )
        upd_width = jnp.where(pm_mean > target_pmove + error, upd_width * 1.1, upd_width)
        width = jnp.where(
            jnp.mod(state.i, window_size) == 0,
            upd_width,
            state.width,
        )
        return WidthSchedulerState(width=width, pmoves=pmoves, i=i)

    return WidthScheduler(init, update)


def make_mh_update(
    logprob_fn: Callable[[Electrons], LogDensity],
    width: Width,
    elec_per_mol: Sequence[int],
):
    mol_to_elecs = np.asarray(elec_per_mol)

    def mh_update(
        key: jax.Array,
        electrons: Electrons,
        log_prob: LogDensity,
        num_accepts: Integer[Array, ''],
    ):
        key, subkey = jax.random.split(key)
        eps = (
            jax.random.normal(subkey, electrons.shape, electrons.dtype)
            * jnp.repeat(width, mol_to_elecs)[:, None]
        )
        new_electrons = electrons + eps
        log_prob_new = logprob_fn(new_electrons)
        ratio = log_prob_new - log_prob

        key, subkey = jax.random.split(key)
        alpha = jnp.log(jax.random.uniform(key, dtype=electrons.dtype))
        cond = ratio > alpha
        new_electrons = jnp.where(
            jnp.repeat(cond, mol_to_elecs)[:, None], new_electrons, electrons
        )
        log_prob = jnp.where(cond, log_prob_new, log_prob)
        num_accepts += cond
        return key, new_electrons, log_prob, num_accepts

    return mh_update


def make_mcmc(wf: GeneralizedWaveFunction, steps: int, width_scheduler: WidthScheduler):
    @jit
    def mcmc(
        key: jax.Array,
        params: WaveFunctionParameters,
        systems: Systems,
        width_state: WidthSchedulerState,
    ):
        # Fix the per molecule parameters and do not recompute them
        wf_fixed = wf.fix_structure(params, systems)
        # Get current width
        width = width_state.width
        batch_size = systems.electrons.shape[0]

        @jax.vmap
        def logprob_fn(electrons: Electrons) -> LogDensity:
            return 2 * wf_fixed.apply(params, systems.replace(electrons=electrons))

        mh_update = make_mh_update(logprob_fn, width, systems.n_elec_by_mol)
        log_probs = logprob_fn(systems.electrons)
        num_accepts = jnp.zeros(log_probs.shape, dtype=jnp.int32)

        key, electrons, _, num_accepts = jax.lax.scan(
            lambda x, _: (mh_update(*x), None),
            (key, systems.electrons, log_probs, num_accepts),
            jnp.arange(steps),
        )[0]
        systems = systems.replace(electrons=electrons)

        pmove = jnp.sum(num_accepts, axis=0) / (steps * batch_size)
        pmove = pmean_if_pmap(pmove)
        width_state = jax.vmap(width_scheduler.update)(width_state, pmove)
        return systems, width_state

    return mcmc
