import jax
from fixtures import *  # noqa: F403
from utils import assert_finite, assert_shape_and_dtype


def test_mcmc(mcmc, neural_pfaffian_params, batched_systems):
    batched_systems = mcmc.init_systems(jax.random.key(0), batched_systems)
    result_systems = mcmc(jax.random.key(55), neural_pfaffian_params, batched_systems)
    assert_shape_and_dtype(result_systems, batched_systems)
    assert_finite(result_systems)
