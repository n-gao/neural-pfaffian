import jax
import jax.numpy as jnp
import numpy as np
from fixtures import *  # noqa: F403
from numpy.testing import assert_allclose
from utils import assert_finite, assert_not_float64, assert_shape_and_dtype  # noqa: F403


def test_fwd_and_bwd(wf_apply, wf_params, systems):
    learnable_parameters = wf_params['params']

    @jax.jit
    @jax.value_and_grad
    def fwd_sum(p, systems):
        return wf_apply({**wf_params, 'params': p}, systems).sum()

    emb_sum, grad = fwd_sum(learnable_parameters, systems)
    assert isinstance(emb_sum, jax.Array)
    assert jax.numpy.isfinite(emb_sum).all()
    assert_shape_and_dtype(learnable_parameters, grad)
    assert_finite(grad)


def test_sign(wf_signed, wf_params, systems):
    sign, _ = wf_signed(wf_params, systems)
    assert sign.shape == (systems.n_mols,)
    assert np.isfinite(sign).all()
    assert np.isin(sign, [-1, 1]).all()


# Run this one in float64
def test_antisymmetry(wf_signed, wf_params, systems_float64):
    systems = systems_float64
    sign, logpsi = wf_signed(wf_params, systems)
    assert sign.shape == (systems.n_mols,)
    assert logpsi.shape == (systems.n_mols,)
    assert np.isfinite(sign).all()
    assert np.isfinite(logpsi).all()
    permutation = np.array([0, 1])
    permuted_system = systems.replace(
        electrons=systems.electrons.at[permutation].set(
            systems.electrons[permutation[::-1]]
        )
    )
    permuted_sign, permuted_logpsi = wf_signed(wf_params, permuted_system)
    assert_allclose(sign[0], -permuted_sign[0])
    assert_allclose(logpsi, permuted_logpsi, atol=1e-8)


def test_param_dtype(wf_params):
    assert_not_float64(wf_params)


def test_out_dtype(wave_function, wf_params, systems):
    logpsi = jax.eval_shape(wave_function.apply, wf_params, systems)
    assert logpsi.dtype == jnp.float32
    assert logpsi.shape == (systems.n_mols,)
