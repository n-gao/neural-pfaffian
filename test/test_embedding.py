import jax
import jax.numpy as jnp
import numpy as np
from fixtures import *  # noqa: F403
from numpy.testing import assert_allclose


def test_fwd_and_bwd(embedding_fwdpass, embedding_params, systems):
    params = embedding_params
    learnable_parameters = params['params']

    @jax.jit
    @jax.value_and_grad
    def fwd_sum(p, systems):
        return embedding_fwdpass({**params, 'params': p}, systems).sum()

    emb_sum, grad = fwd_sum(learnable_parameters, systems)
    assert isinstance(emb_sum, jax.Array)
    assert np.isfinite(emb_sum).all()

    def is_finite(path, x):
        assert np.isfinite(x).all(), f'{path} is not finite'

    jax.tree_util.tree_map_with_path(is_finite, grad)


def test_equivariance(embedding_fwdpass, embedding_params, systems):
    emb = embedding_fwdpass(embedding_params, systems)
    assert isinstance(emb, jax.Array)
    assert np.isfinite(emb).all()
    assert emb.shape[0] == systems.n_elec
    assert len(emb.shape) == 2
    permutation = np.array([0, 1])
    permuted_system = systems.replace(
        electrons=systems.electrons.at[permutation].set(
            systems.electrons[permutation[::-1]]
        )
    )
    permuted_emb = embedding_fwdpass(embedding_params, permuted_system)
    permuted_emb = np.asarray(permuted_emb)
    emb = emb.at[permutation].set(emb[permutation[::-1]])
    emb = np.asarray(emb)
    assert_allclose(emb, permuted_emb, atol=1e-6)


def test_param_dtype(embedding_params):
    def assert_non_float64(path, x):
        if isinstance(x, jax.Array):
            assert x.dtype != jnp.float64, f'{path} is float64'

    jax.tree_util.tree_map_with_path(assert_non_float64, embedding_params)


def test_fwd_dtype(embedding_fwdpass, embedding_params, systems):
    emb = jax.eval_shape(embedding_fwdpass, embedding_params, systems)
    assert emb.dtype == jnp.float32
    assert emb.shape[0] == systems.n_elec
    assert len(emb.shape) == 2
