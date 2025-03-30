import folx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from neural_pfaffian.linalg import skewsymmetric_quadratic, slog_pfaffian

from utils import assert_finite


@pytest.fixture(scope='module', params=[2, 3, 4, 32])
def num_orbitals(request):
    return request.param


@pytest.fixture(scope='module', params=[2, 8])
def num_electrons(request):
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


def test_slog_pfaffian_hessian(antisymmetric_mat):
    hess = jax.hessian(lambda x: slog_pfaffian(x)[1])(antisymmetric_mat)
    target = jax.hessian(log_pfaffian)(antisymmetric_mat)
    if antisymmetric_mat.shape[0] % 2 != 0:
        assert np.isnan(hess).all()
    else:
        assert_finite(hess)
        npt.assert_allclose(hess, target, atol=1e-8)


def test_slog_pfaffian_folx(antisymmetric_mat):
    def f(x):
        return slog_pfaffian(x)[1]

    lapl, jac = folx.ForwardLaplacianOperator(0)(f)(antisymmetric_mat)
    if antisymmetric_mat.shape[0] % 2 != 0:
        assert np.isnan(jac).all()
        assert np.isnan(lapl).all()
        return
    assert_finite((lapl, jac))
    t_lapl, t_jac = folx.LoopLaplacianOperator()(f)(antisymmetric_mat)
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
    elecs = jax.random.normal(jax.random.PRNGKey(0), (4, 3))
    W = jax.random.normal(jax.random.PRNGKey(1), (3, 8))
    W_A = jax.random.normal(jax.random.PRNGKey(2), (3, 8 * 8))

    def f(x):
        orb = jnp.tanh(x @ W)
        A = jnp.tanh(x @ W_A).sum(0).reshape(8, 8)
        A = A - A.T
        return jnp.abs(skewsymmetric_quadratic(orb, A)).sum()

    lapl, jac = folx.ForwardLaplacianOperator(0)(f)(elecs)
    lapl_t, jac_t = folx.LoopLaplacianOperator()(f)(elecs)
    lapl_t = lapl_t.sum()
    assert_finite((lapl, jac))
    npt.assert_allclose(jac, jac_t, atol=1e-8)
    npt.assert_allclose(lapl, lapl_t, atol=1e-8)
