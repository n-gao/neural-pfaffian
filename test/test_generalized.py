import jax
import pytest
from fixtures import *  # noqa: F403
from numpy.testing import assert_allclose
from utils import assert_finite, assert_not_float64


def test_param_dtype(generalized_wf_params):
    assert_not_float64(generalized_wf_params)


def test_fwd_and_bwd(generalized_wf, generalized_wf_params, two_systems):
    @jax.jit
    @jax.value_and_grad
    def fwd_sum(p, systems):
        return generalized_wf.apply(p, systems).sum()

    emb_sum, grad = fwd_sum(generalized_wf_params, two_systems)
    assert isinstance(emb_sum, jax.Array)
    assert jax.numpy.isfinite(emb_sum).all()
    assert_finite(grad)


def test_fixed_structure(generalized_wf, generalized_wf_params, one_system, two_systems):
    params = generalized_wf_params
    fixed = generalized_wf.fix_structure(generalized_wf_params, two_systems)
    assert_allclose(
        fixed.apply(params, two_systems), generalized_wf.apply(params, two_systems)
    )
    with pytest.raises(ValueError):
        fixed.apply(params, one_system)

    fixed = generalized_wf.fix_structure(generalized_wf_params, one_system)
    assert_allclose(
        fixed.apply(params, one_system), generalized_wf.apply(params, one_system)
    )
    with pytest.raises(ValueError):
        fixed.apply(params, two_systems)
