import jax
import jax.numpy as jnp
from fixtures import *  # noqa: F403
from numpy.testing import assert_allclose
from utils import assert_finite, assert_not_float64

from neural_pfaffian.nn.module import ParamTypes


def test_param_dtypes(meta_gnn, one_system):
    key = jax.random.PRNGKey(0)
    params = meta_gnn.init(key, one_system)
    assert_not_float64(params)


def test_shape_and_dtype(out_meta, meta_gnn, one_system, systems):
    key = jax.random.PRNGKey(0)
    meta_gnn = meta_gnn.copy(out_structure=out_meta)
    params = meta_gnn.lazy_init(key, one_system)
    out = jax.eval_shape(meta_gnn.apply, params, systems)
    assert out.dtype == out_meta.shape_and_dtype.dtype
    match out_meta.param_type:
        case ParamTypes.GLOBAL:
            leading_dim = systems.n_mols
        case ParamTypes.NUCLEI:
            leading_dim = systems.n_nuc
        case ParamTypes.NUCLEI_NUCLEI:
            leading_dim = systems.n_nn
        case _:
            raise ValueError(f'Unknown param type {out_meta.param_type}')
    assert out.shape == (leading_dim, *out_meta.shape_and_dtype.shape)


def test_fwd_and_bwd(out_meta, meta_gnn, systems):
    key = jax.random.PRNGKey(0)
    meta_gnn = meta_gnn.copy(out_structure=out_meta)
    params = meta_gnn.lazy_init(key, systems)

    @jax.jit
    @jax.value_and_grad
    def fwd_sum(p, systems):
        return meta_gnn.apply({**params, 'params': p}, systems).sum()

    emb_sum, grad = fwd_sum(params['params'], systems)
    assert isinstance(emb_sum, jax.Array)
    assert jnp.isfinite(emb_sum).all()
    assert_finite(grad)


def test_independence(out_meta, meta_gnn, systems_float64):
    key = jax.random.PRNGKey(0)
    meta_gnn = meta_gnn.copy(out_structure=out_meta)
    params = meta_gnn.lazy_init(key, systems_float64)
    targets = meta_gnn.apply(params, systems_float64)

    offset = 0
    for s in systems_float64:
        chunk_size = out_meta.param_type.value.chunk_fn(s.spins[0], s.charges[0])
        target = targets[offset : offset + chunk_size]
        indep = meta_gnn.apply(params, s)
        assert_allclose(indep, target, atol=1e-6)
        offset += chunk_size
