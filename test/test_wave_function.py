import jax
from fixtures import *  # noqa: F403


def test_fwd_and_bwd(systems_wf_params):
    systems, fwd_pass, params = systems_wf_params
    learnable_parameters = params['params']

    def fwd_sum(p, systems):
        return fwd_pass({**params, 'params': p}, systems).sum()

    emb_sum, grad = jax.value_and_grad(fwd_sum, argnums=0)(learnable_parameters, systems)
    assert isinstance(emb_sum, jax.Array)
    assert jax.numpy.isfinite(emb_sum).all()

    def is_finite(path, x):
        assert jax.numpy.isfinite(x).all(), f'{path} is not finite'

    jax.tree_util.tree_map_with_path(is_finite, grad)
