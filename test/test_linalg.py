import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from neural_pfaffian.linalg import slog_pfaffian

from utils import assert_finite


@pytest.fixture(scope='module', params=[2, 4, 8, 16, 32])
def antisymmetric_mat(request):
    n = request.param
    A = jax.random.normal(jax.random.PRNGKey(0), (n, n))
    return A - A.T


def test_slog_pfaffian(antisymmetric_mat):
    sign, log_pf = slog_pfaffian(antisymmetric_mat)
    _, log_det = jnp.linalg.slogdet(antisymmetric_mat)
    assert_finite((sign, log_pf))
    npt.assert_allclose(2 * log_pf, log_det)
