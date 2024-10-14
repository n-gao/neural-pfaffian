import jax
import jax.numpy as jnp
import numpy as np
from fixtures import *  # noqa: F403

from neural_pfaffian.mcmc import make_mcmc, make_width_scheduler


def test_mcmc(generalized_wf, generalized_wf_params, batched_systems):
    scheduler = make_width_scheduler()
    state = jax.vmap(scheduler.init)(jnp.ones((batched_systems.n_mols,)))
    mcmc = make_mcmc(generalized_wf, 5, scheduler)
    result_systems, result_state = mcmc(
        jax.random.key(2), generalized_wf_params, batched_systems, state
    )
    assert result_systems.electrons.shape == batched_systems.electrons.shape
    assert result_systems.electrons.dtype == batched_systems.electrons.dtype
    assert result_state.width.shape == state.width.shape
    assert result_state.width.dtype == state.width.dtype
    assert result_state.i.shape == state.i.shape
    assert result_state.i.dtype == state.i.dtype
    assert result_state.pmoves.shape == state.pmoves.shape
    assert result_state.pmoves.dtype == state.pmoves.dtype
    assert np.all(np.isfinite(result_systems.electrons))
    assert np.all(np.isfinite(result_state.width))
    assert np.all(np.isfinite(result_state.i))
    assert np.all(np.isfinite(result_state.pmoves))
