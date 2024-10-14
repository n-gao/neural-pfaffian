import jax
import numpy as np
from fixtures import *  # noqa: F403
from numpy.testing import assert_allclose


def test_fwd_and_bwd(systems_embedding_params):
    systems, fwd_pass, params = systems_embedding_params
    learnable_parameters = params['params']

    def fwd_sum(p, systems):
        return fwd_pass({**params, 'params': p}, systems).sum()

    emb_sum, grad = jax.value_and_grad(fwd_sum, argnums=0)(learnable_parameters, systems)
    assert isinstance(emb_sum, jax.Array)
    assert np.isfinite(emb_sum).all()

    def is_finite(path, x):
        assert np.isfinite(x).all(), f'{path} is not finite'

    jax.tree_util.tree_map_with_path(is_finite, grad)


def test_equivariance(systems_embedding_params):
    systems, fwd_pass, params = systems_embedding_params
    emb = fwd_pass(params, systems)
    assert isinstance(emb, jax.Array)
    assert np.isfinite(emb).all()
    assert emb.shape[0] == systems.n_elec
    permutation = np.array([0, 1])
    permuted_system = systems.replace(
        electrons=systems.electrons.at[permutation].set(
            systems.electrons[permutation[::-1]]
        )
    )
    permuted_emb = fwd_pass(params, permuted_system)
    permuted_emb = np.asarray(permuted_emb)
    emb = emb.at[permutation].set(emb[permutation[::-1]])
    emb = np.asarray(emb)
    assert_allclose(emb, permuted_emb, atol=1e-6)
