import folx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neural_pfaffian.linalg import (
    slog_pfaffian_bordered_quadratic,
    slog_pfaffian_skewsymmetric_quadratic,
)

N_ELEC, N_ORB, N_COORD = 4, 6, 12


@pytest.fixture(scope='module')
def skew_matrix():
    A = jax.random.normal(jax.random.key(0), (N_ORB, N_ORB), dtype=jnp.float64)
    return (A - A.mT) / 2


@pytest.fixture(scope='module')
def coords():
    return jax.random.normal(jax.random.key(1), (N_COORD,), dtype=jnp.float64)


def orbitals_dense(r):
    # every orbital entry depends on all coordinates
    w = jax.random.normal(jax.random.key(2), (N_COORD, N_ELEC * N_ORB), jnp.float64)
    return jnp.tanh(r @ w + jnp.sin(r).sum()).reshape(N_ELEC, N_ORB)


def orbitals_sparse(r):
    # depends only on the first half of the coordinates
    w = jax.random.normal(jax.random.key(2), (N_COORD // 2, N_ELEC * N_ORB), jnp.float64)
    return jnp.tanh(r[: N_COORD // 2] @ w).reshape(N_ELEC, N_ORB)


def orbitals_batched(r):
    x = orbitals_dense(r)
    return jnp.stack([x, jnp.sin(x)])[None]  # (mols=1, det=2, elec, orb)


@pytest.mark.parametrize(
    'orbital_fn', [orbitals_dense, orbitals_sparse, orbitals_batched]
)
def test_fused_forward_laplacian(coords, skew_matrix, orbital_fn):
    def log_pf(r):
        return slog_pfaffian_skewsymmetric_quadratic(orbital_fn(r), skew_matrix)[1]

    result = folx.forward_laplacian(log_pf, sparsity_threshold=64)(coords)
    jac_ref = jax.jacobian(log_pf)(coords)
    lapl_ref = jnp.trace(jax.hessian(log_pf)(coords), axis1=-2, axis2=-1)
    np.testing.assert_allclose(result.x, log_pf(coords), rtol=1e-12)
    # sparse jacobians only span the coordinates they depend on
    jac = result.jacobian.dense_array
    pad = [(0, N_COORD - jac.shape[0])] + [(0, 0)] * (jac.ndim - 1)
    np.testing.assert_allclose(
        jnp.pad(jac, pad), jnp.moveaxis(jac_ref, -1, 0), atol=1e-10
    )
    np.testing.assert_allclose(result.laplacian, lapl_ref, atol=1e-9)


def test_fused_forward_laplacian_sign(coords, skew_matrix):
    def slog_pf(r):
        return slog_pfaffian_skewsymmetric_quadratic(orbitals_dense(r), skew_matrix)

    sign, log_pf = folx.forward_laplacian(slog_pf)(coords)
    sign_ref, _ = slog_pf(coords)
    sign = sign.x if isinstance(sign, folx.api.FwdLaplArray) else sign
    np.testing.assert_allclose(sign, sign_ref)


def border_dense(r):
    w = jax.random.normal(jax.random.key(3), (N_COORD, N_ELEC - 1), jnp.float64)
    return jnp.sin(r @ w)


@pytest.mark.parametrize('batched', [False, True])
def test_fused_bordered_forward_laplacian(coords, skew_matrix, batched):
    def log_pf(r):
        x, c = orbitals_dense(r)[: N_ELEC - 1], border_dense(r)  # odd n
        if batched:
            x, c = jnp.stack([x, jnp.sin(x)])[None], jnp.stack([c, jnp.cos(c)])[None]
        return slog_pfaffian_bordered_quadratic(x, c, skew_matrix)[1]

    result = folx.forward_laplacian(log_pf)(coords)
    jac_ref = jax.jacobian(log_pf)(coords)
    lapl_ref = jnp.trace(jax.hessian(log_pf)(coords), axis1=-2, axis2=-1)
    np.testing.assert_allclose(result.x, log_pf(coords), rtol=1e-12)
    np.testing.assert_allclose(
        result.jacobian.dense_array, jnp.moveaxis(jac_ref, -1, 0), atol=1e-10
    )
    np.testing.assert_allclose(result.laplacian, lapl_ref, atol=1e-9)


def test_bordered_gradients(coords, skew_matrix):
    # gradients w.r.t. x, c and A through the custom JVP
    x = orbitals_dense(coords)[: N_ELEC - 1]
    c = border_dense(coords)

    def f(x, c, A):
        return slog_pfaffian_bordered_quadratic(x, c, A)[1]

    def f_ref(x, c, A):
        n = x.shape[-2]
        y = x @ A @ x.mT
        Y = jnp.zeros((n + 1, n + 1), x.dtype)
        Y = Y.at[:n, :n].set(y).at[:n, n].set(c).at[n, :n].set(-c)
        return jnp.linalg.slogdet(Y)[1] / 2

    for i in range(3):
        g = jax.grad(f, argnums=i)(x, c, skew_matrix)
        g_ref = jax.grad(f_ref, argnums=i)(x, c, skew_matrix)
        if i == 2:  # slogdet spreads the A gradient over both triangles
            g_ref = (g_ref - g_ref.mT) / 2
            g = (g - g.mT) / 2
        np.testing.assert_allclose(g, g_ref, atol=1e-10)


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='pallas kernel is GPU-only')
@pytest.mark.parametrize('bordered', [False, True])
def test_fused_pallas_matches_float64(coords, skew_matrix, bordered):
    def log_pf(r, dtype):
        x = orbitals_batched(r).astype(dtype)
        A = skew_matrix.astype(dtype)
        if bordered:
            c = jnp.sin(x[..., 0]).astype(dtype)
            return slog_pfaffian_bordered_quadratic(x[..., :-1, :], c[..., :-1], A)[1]
        return slog_pfaffian_skewsymmetric_quadratic(x, A)[1]

    # float32 dispatches to the pallas kernel, float64 to the einsum fallback
    out32 = folx.forward_laplacian(lambda r: log_pf(r, jnp.float32))(coords)
    out64 = folx.forward_laplacian(lambda r: log_pf(r, jnp.float64))(coords)
    np.testing.assert_allclose(out32.x, out64.x, rtol=1e-4)
    np.testing.assert_allclose(
        out32.jacobian.dense_array, out64.jacobian.dense_array, rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(out32.laplacian, out64.laplacian, rtol=1e-3, atol=1e-3)


def test_fwd_lapl_config():
    from neural_pfaffian.kernels import pfaffian_fwd_lapl_config

    small = jax.ShapeDtypeStruct((60, 1, 20, 40), jnp.float32)
    huge = jax.ShapeDtypeStruct((768, 1, 256, 512), jnp.float32)
    if jax.default_backend() == 'gpu':
        assert pfaffian_fwd_lapl_config(small, bordered=False) is not None
    else:
        assert pfaffian_fwd_lapl_config(small, bordered=False) is None
    assert pfaffian_fwd_lapl_config(huge, bordered=False) is None


def test_fused_matches_decomposed(coords, skew_matrix):
    def fused(r):
        return slog_pfaffian_skewsymmetric_quadratic(orbitals_dense(r), skew_matrix)[1]

    def decomposed(r):
        return slog_pfaffian_skewsymmetric_quadratic.fun(
            orbitals_dense(r), skew_matrix
        )[1]

    out_fused = folx.forward_laplacian(fused)(coords)
    out_ref = folx.forward_laplacian(decomposed)(coords)
    np.testing.assert_allclose(out_fused.x, out_ref.x, rtol=1e-12)
    np.testing.assert_allclose(
        out_fused.jacobian.dense_array, out_ref.jacobian.dense_array, atol=1e-10
    )
    np.testing.assert_allclose(out_fused.laplacian, out_ref.laplacian, atol=1e-9)
