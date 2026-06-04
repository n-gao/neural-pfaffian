import jax.numpy as jnp
import numpy as np
import pytest
from fixtures import *  # noqa: F403
from orthogonal_wavefunction import *  # noqa: F403

from neural_pfaffian.sample_reweighting import (
    LOG_NORMALIZER_CONSTANTS_KEY,
    compute_logpsi,
    compute_reweighting_factor,
    get_normalizing_constant_ratios,
    unpack_reweighted_tensor,
    update_normalizing_constant_ratios,
)


@pytest.fixture(scope='module')
def reweighting_systems(perfect_sample_systems):
    """Perfect samples with 4 walkers and initialized normalizer constants."""
    systems = perfect_sample_systems.replace(
        electrons=jnp.stack([perfect_sample_systems.electrons] * 4),
    )
    return systems.set_mol_data(
        LOG_NORMALIZER_CONSTANTS_KEY,
        jnp.zeros((systems.n_mols,), dtype=jnp.float64),
    )


def test_compute_logpsi_shape_dtype(
    generalized_wf,
    generalized_wf_params,
    batched_excited_systems,
):
    logpsis, signs = compute_logpsi(
        generalized_wf,
        generalized_wf_params,
        batched_excited_systems,
    )

    assert logpsis.shape == (2, batched_excited_systems.n_unique_mols, 2, 2)
    assert signs.shape == logpsis.shape
    assert logpsis.dtype == batched_excited_systems.electrons.dtype
    assert signs.dtype == batched_excited_systems.electrons.dtype
    assert np.isfinite(np.asarray(logpsis)).all()


def test_update_normalizing_constant_ratios_orthogonal_states(
    mock_orthogonal_wf,
    mock_wf_params,
    reweighting_systems,
):
    systems, aux_data = update_normalizing_constant_ratios(
        mock_orthogonal_wf,
        mock_wf_params,
        reweighting_systems,
    )
    log_ratios = get_normalizing_constant_ratios(systems)

    assert np.isfinite(np.asarray(log_ratios)).all()
    assert np.isfinite(np.asarray(aux_data['normalizing_constants/log_r_min']))
    assert np.allclose(np.asarray(log_ratios), 0.0, atol=1e-2)


def test_compute_reweighting_factor_normalization(
    generalized_wf,
    generalized_wf_params,
    batched_excited_systems,
):
    factor = compute_reweighting_factor(
        generalized_wf,
        generalized_wf_params,
        batched_excited_systems,
    )
    unpacked = unpack_reweighted_tensor(factor, batched_excited_systems)

    assert factor.shape == (
        batched_excited_systems.electrons.shape[0]
        * batched_excited_systems.max_num_states,
        2,
    )
    assert np.isfinite(np.asarray(factor)).all()
    assert np.allclose(
        np.asarray(unpacked.sum(axis=-1)),
        batched_excited_systems.max_num_states,
        atol=1e-5,
    )
