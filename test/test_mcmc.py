import jax
import jax.numpy as jnp
from fixtures import *  # noqa: F403
from utils import assert_finite, assert_shape_and_dtype

from neural_pfaffian.mcmc import make_langevin_update_proposal


def test_mcmc(mcmc, neural_pfaffian_params, batched_systems):
    batched_systems = mcmc.init_systems(jax.random.key(0), batched_systems)
    result_systems, _ = mcmc(jax.random.key(55), neural_pfaffian_params, batched_systems)
    assert_shape_and_dtype(result_systems, batched_systems)
    assert_finite(result_systems)


def test_langevin_sampling(batched_systems):
    def dummy_log_prob_fn(x):
        return x.mean()

    proposal_fn = make_langevin_update_proposal(
        batched_systems,
        dummy_log_prob_fn,
        jnp.ones(batched_systems.n_mols),
    )

    proposal_fn(jnp.ones(()), jax.random.key(0), batched_systems.electrons)
