import folx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from neural_pfaffian.linalg import (
    antisymmetric_block_diagonal,
    skewsymmetric_quadratic,
    slog_pfaffian,
    slog_pfaffian_with_updates,
)

from utils import assert_finite


@pytest.fixture(scope='module', params=[2, 3, 4, 8])
def num_orbitals(request):
    return request.param


@pytest.fixture(scope='module', params=[2, 8])
def num_electrons(request):
    return request.param


@pytest.fixture(scope='module', params=[2, 4, 6])
def rank(request):
    return request.param


@pytest.fixture(scope='module')
def antisymmetric_mat(num_orbitals):
    n = num_orbitals
    A = jax.random.normal(jax.random.PRNGKey(0), (n, n))
    return A - A.T


@pytest.fixture(scope='module')
def orbitals(num_orbitals, num_electrons):
    n = num_orbitals
    m = num_electrons
    if m > n:
        pytest.skip('Number of electrons cannot exceed number of orbitals.')
    Phi = jax.random.normal(jax.random.PRNGKey(1), (m, n))
    return Phi


@pytest.fixture(scope='module')
def updates(num_electrons, rank):
    updates = jax.random.normal(jax.random.PRNGKey(2), (num_electrons, rank))
    return updates


def log_pfaffian(x):
    return jnp.linalg.slogdet(x)[1] / 2


def test_slog_pfaffian(antisymmetric_mat):
    sign, log_pf = slog_pfaffian(antisymmetric_mat)
    target = log_pfaffian(antisymmetric_mat)
    if antisymmetric_mat.shape[0] % 2 != 0:
        assert np.isneginf(log_pf).all()
    else:
        assert_finite((sign, log_pf))
        npt.assert_allclose(log_pf, target)


def test_slog_pfaffian_jvp(antisymmetric_mat):
    log_pf = jax.jacobian(lambda x: slog_pfaffian(x)[1])(antisymmetric_mat)
    target = jax.jacobian(log_pfaffian)(antisymmetric_mat)
    if antisymmetric_mat.shape[0] % 2 != 0:
        assert np.isnan(log_pf).all()
    else:
        assert_finite(log_pf)
        npt.assert_allclose(log_pf, target, atol=1e-8)


def test_slog_pfaffian_folx(num_orbitals):
    W = jax.random.normal(jax.random.key(4), (num_orbitals, 12, num_orbitals))

    def f(x):
        A = x @ W
        A = jnp.tanh(A - A.mT)
        return slog_pfaffian(A)[1]

    inp = jax.random.normal(jax.random.PRNGKey(0), (12,))
    lapl, jac = folx.ForwardLaplacianOperator(0)(f)(inp)
    if num_orbitals % 2 != 0:
        assert np.isnan(jac).all()
        assert np.isnan(lapl).all()
        return
    assert_finite((lapl, jac))
    t_lapl, t_jac = folx.LoopLaplacianOperator()(f)(inp)
    t_lapl = t_lapl.sum()
    npt.assert_allclose(lapl, t_lapl, atol=1e-8)
    npt.assert_allclose(jac, t_jac, atol=1e-8)


def sk_sq(x, A):
    return x @ A @ x.mT


def test_skewsymmetric_quadratic(orbitals, antisymmetric_mat):
    out = skewsymmetric_quadratic(orbitals, antisymmetric_mat)
    target = sk_sq(orbitals, antisymmetric_mat)
    assert_finite(out)
    npt.assert_allclose(out, target, atol=1e-8)


def test_skewsymmetric_quadratic_jvp_lhs(orbitals, antisymmetric_mat):
    out = jax.jacobian(lambda x: skewsymmetric_quadratic(x, antisymmetric_mat))(orbitals)
    target = jax.jacobian(lambda x: sk_sq(x, antisymmetric_mat))(orbitals)
    assert_finite(out)
    npt.assert_allclose(out, target, atol=1e-8)


def test_skewsymmetric_quadratic_jvp_rhs(orbitals, antisymmetric_mat):
    out = jax.jacobian(lambda x: skewsymmetric_quadratic(orbitals, x))(antisymmetric_mat)
    target = jax.jacobian(lambda x: sk_sq(orbitals, x))(antisymmetric_mat)
    assert_finite(out)
    npt.assert_allclose(out, target, atol=1e-8)


def test_skewsymmetric_quadratic_folx():
    elecs = jax.random.normal(jax.random.PRNGKey(0), (12,))
    W = jax.random.normal(jax.random.PRNGKey(1), (8, 12, 8))
    W_A = jax.random.normal(jax.random.PRNGKey(2), (8, 12, 8))

    def f(x):
        orb = jnp.tanh(x @ W)
        A = jnp.tanh(x @ W_A).reshape(8, 8)
        A = A - A.T
        return jnp.abs(skewsymmetric_quadratic(orb, A)).sum()

    lapl, jac = folx.ForwardLaplacianOperator(0)(f)(elecs)
    lapl_t, jac_t = folx.LoopLaplacianOperator()(f)(elecs)
    lapl_t = lapl_t.sum()
    assert_finite((lapl, jac))
    npt.assert_allclose(jac, jac_t, atol=1e-8)
    npt.assert_allclose(lapl, lapl_t, atol=1e-8)


def test_slog_pfaffian_with_updates(orbitals, antisymmetric_mat, updates):
    xAx = skewsymmetric_quadratic(orbitals, antisymmetric_mat)
    sign, log = slog_pfaffian_with_updates(xAx, updates)

    assert_finite((sign, log))

    C = antisymmetric_block_diagonal(updates.shape[-1] // 2, updates.dtype)
    xAx_ = skewsymmetric_quadratic(updates, C) + xAx
    sign_t, log_t = slog_pfaffian(xAx_)

    npt.assert_allclose(sign, sign_t, atol=1e-8)
    npt.assert_allclose(log, log_t, atol=1e-8)
