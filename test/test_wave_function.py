import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fixtures import *  # noqa: F403
from numpy.testing import assert_allclose
from utils import assert_finite, assert_not_float64, assert_shape_and_dtype


@pytest.mark.parametrize('systems', ['two_systems', 'one_system'], indirect=True)
def test_fwd_and_bwd(wf_apply, wf_params, systems):
    learnable_parameters = wf_params['params']

    @jax.jit
    @jax.value_and_grad
    def fwd_sum(p, systems):
        return wf_apply({**wf_params, 'params': p}, systems).sum()

    emb_sum, grad = fwd_sum(learnable_parameters, systems)
    assert isinstance(emb_sum, jax.Array)
    assert jax.numpy.isfinite(emb_sum).all()
    assert_finite(grad)
    assert_shape_and_dtype(learnable_parameters, grad)


def test_sign(wf_signed, wf_params, systems):
    sign, _ = wf_signed(wf_params, systems)
    assert sign.shape == (systems.n_mols,)
    assert np.isfinite(sign).all()
    assert np.isin(sign, [-1, 1]).all()


# Run this one in float64
@pytest.mark.parametrize('systems', ['two_systems', 'one_system'], indirect=True)
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
            systems.electrons[permutation[::-1]],
        ),
    )
    permuted_sign, permuted_logpsi = wf_signed(wf_params, permuted_system)
    assert_allclose(sign[0], -permuted_sign[0])
    assert_allclose(logpsi, permuted_logpsi, atol=1e-8)


@pytest.mark.parametrize('systems', ['excited_systems'], indirect=True)
def test_excitation_dependence(wf_signed, wf_params, systems):
    # Make sure the wave function spits out different results for different excitations
    _sign, logpsi = wf_signed(wf_params, systems)
    # We need to compare with an excited system at the same index
    # since the embeddings for identical stacked systems are not the same
    systems = systems.replace(excitations=systems.excitations[::-1])
    _sign_swap, logpsi_swap = wf_signed(wf_params, systems)

    systems = systems.set_global_excitation(0)
    _sign_ground, logpsi_ground = wf_signed(wf_params, systems)

    systems = systems.set_global_excitation(1)
    _sign_excited, logpsi_excited = wf_signed(wf_params, systems)

    assert not np.allclose(logpsi, logpsi_swap)
    assert np.allclose(logpsi_ground, [logpsi[0], logpsi_swap[1]])
    assert np.allclose(logpsi_excited, [logpsi_swap[0], logpsi[1]])


@pytest.mark.parametrize('systems', ['two_systems', 'one_system'], indirect=True)
def test_param_dtype(wf_params):
    assert_not_float64(wf_params)


@pytest.mark.parametrize('systems', ['two_systems', 'one_system'], indirect=True)
def test_out_dtype(wave_function, wf_params, systems):
    logpsi = jax.eval_shape(wave_function.apply, wf_params, systems)
    assert logpsi.dtype == jnp.float32
    assert logpsi.shape == (systems.n_mols,)
