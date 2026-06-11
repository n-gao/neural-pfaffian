import jax
import jax.numpy as jnp
import pytest
from fixtures import *  # noqa: F403
from orthogonal_wavefunction import *  # noqa: F403
from utils import assert_finite, to_numpy_dict

from neural_pfaffian.clipping import NoneClipping, QuantileMasking
from neural_pfaffian.overlap import OverlapPenalty, _compute_mean_overlap
from neural_pfaffian.overlap_scaler import NoScaler
from neural_pfaffian.sample_reweighting import compute_logpsi


@pytest.fixture(scope='module')
def overlap_penalty(mock_orthogonal_wf):
    return OverlapPenalty(
        wave_function=mock_orthogonal_wf,
        clipping=NoneClipping(),
        overlap_scaler=NoScaler(triangular_mask=False),
        penalty_scale=1.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mock_orthogonal_wf(mock_orthogonal_wf, mock_wf_params, excited_systems):
    result = mock_orthogonal_wf.apply(mock_wf_params, excited_systems)
    assert_finite(result)


def test_mock_with_perfect_samples(
    mock_orthogonal_wf,
    mock_wf_params,
    perfect_sample_systems,
):
    diagonal = mock_orthogonal_wf.apply(mock_wf_params, perfect_sample_systems)

    electrons = jnp.zeros_like(perfect_sample_systems.electrons)
    electrons = electrons.at[:4].set(3.0)

    off_diagonal = mock_orthogonal_wf.apply(
        mock_wf_params,
        perfect_sample_systems.replace(electrons=electrons),
    )

    assert jnp.allclose(diagonal, 0.0)
    assert jnp.allclose(off_diagonal, -3.0 * jnp.sqrt(3))


def test_compute_logpsi(mock_orthogonal_wf, mock_wf_params, batched_perfect_samples):
    logpsis, signs = compute_logpsi(
        mock_orthogonal_wf,
        mock_wf_params,
        batched_perfect_samples,
    )
    assert logpsis.shape == (2, 1, 2, 2)
    assert signs.shape == (2, 1, 2, 2)
    assert jnp.allclose(logpsis[0, 0], (jnp.eye(2) - 1.0) * 3.0 * jnp.sqrt(3))


def test_unreweighted_overlap_is_near_identity(
    mock_wf_params,
    batched_perfect_random_samples,
    overlap_penalty,
):
    overlap = overlap_penalty.pairwise_overlap(
        mock_wf_params,
        batched_perfect_random_samples,
    )

    assert overlap.shape == (batched_perfect_random_samples.n_unique_mols, 2, 2)
    assert jnp.allclose(overlap, jnp.eye(2)[None], atol=1e-1)


def test_overlap_penalty_returns_finite_cotangents(
    mock_wf_params,
    batched_perfect_random_samples,
    overlap_penalty,
):
    systems = batched_perfect_random_samples
    local_energy = jax.random.normal(
        jax.random.PRNGKey(0),
        (systems.electrons.shape[0], systems.n_mols),
        dtype=systems.electrons.dtype,
    )

    (cotangent, aux_data), new_systems, state = overlap_penalty(
        mock_wf_params,
        systems,
        local_energy,
        overlap_penalty.init(),
        jnp.array(0, dtype=jnp.int32),
        False,
    )

    assert state is not None
    assert_finite(cotangent)
    assert_finite(aux_data)
    assert cotangent.shape == local_energy.shape
    assert new_systems.electrons.shape == systems.electrons.shape


def test_overlap_exact_for_perfect_samples(
    overlap_penalty,
    mock_wf_params,
    batched_perfect_samples,
):
    """Perfect samples sit exactly on each state's centroid, so off-diagonal
    ratios are exp(-3*sqrt(3)) and the diagonal is 1."""
    overlap = overlap_penalty.pairwise_overlap(mock_wf_params, batched_perfect_samples)
    off_diag = jnp.exp(-3.0 * jnp.sqrt(jnp.array(3.0, dtype=jnp.float32)))
    expected = jnp.array([[[1.0, off_diag], [off_diag, 1.0]]], dtype=jnp.float32)
    assert jnp.allclose(overlap, expected, atol=1e-6)


def test_cotangent_vanishes_for_noiseless_samples(
    overlap_penalty,
    mock_wf_params,
    batched_perfect_samples,
):
    """With all walkers at identical positions the MC variance is zero, so
    psi_ratio == mean_ratio for every sample and the cotangent is exactly 0."""
    systems = batched_perfect_samples
    local_energy = jnp.zeros(
        (systems.electrons.shape[0], systems.n_mols),
        dtype=systems.electrons.dtype,
    )

    (cotangent, _), _, _ = overlap_penalty(
        mock_wf_params,
        systems,
        local_energy,
        overlap_penalty.init(),
        jnp.array(0, dtype=jnp.int32),
        False,
    )

    assert jnp.allclose(cotangent, 0.0)


def test_overlap_penalty_regression(
    ndarrays_regression,
    generalized_wf,
    generalized_wf_params,
    batched_excited_systems,
):
    penalty = OverlapPenalty(
        wave_function=generalized_wf,
        clipping=NoneClipping(),
        overlap_scaler=NoScaler(triangular_mask=False),
        penalty_scale=1.0,
    )
    local_energy = jnp.array(
        [[-1.0, -0.75], [-1.1, -0.65]],
        dtype=batched_excited_systems.electrons.dtype,
    )
    (cotangent, aux), *_ = penalty(
        generalized_wf_params,
        batched_excited_systems,
        local_energy,
        penalty.init(),
        jnp.array(0, dtype=jnp.int32),
        False,
    )
    pairwise = penalty.pairwise_overlap(generalized_wf_params, batched_excited_systems)
    ndarrays_regression.check(
        to_numpy_dict(
            {
                'pairwise': pairwise,
                'cotangent': cotangent,
                'loss': aux['loss'],
            },
        ),
        default_tolerance={'rtol': 1e-5, 'atol': 1e-5},
    )


def test_compute_mean_overlap_symmetrizes_ratio_matrix():
    mean_ratio = jnp.array(
        [
            [
                [1.0, 0.25, -2.0],
                [0.16, 1.0, 0.5],
                [-0.5, 0.18, 1.0],
            ],
        ],
        dtype=jnp.float32,
    )

    overlap = _compute_mean_overlap(mean_ratio)

    assert overlap.shape == mean_ratio.shape
    assert jnp.allclose(jnp.diagonal(overlap, axis1=-2, axis2=-1), 1.0)
    assert jnp.allclose(overlap, jnp.swapaxes(overlap, -1, -2))


def test_effective_sample_size(overlap_penalty, excited_systems):
    """Directly exercise the relocated ESS computation (new in the overlap
    penalty). ``excited_systems`` has mol_ids=(0, 0) -> 1 structure, 2 states."""
    n_samples, n_mols = 4, excited_systems.n_mols  # n_mols == 2
    reweighting_factor = jnp.ones((n_samples, n_mols), dtype=jnp.float32)

    # All-True mask + unit weights -> ESS fraction is exactly 1.0 everywhere.
    full_mask = jnp.ones((n_samples, n_mols), dtype=bool)
    aux = overlap_penalty._effective_sample_size(
        full_mask, reweighting_factor, excited_systems
    )
    assert set(aux) == {'ESS', 'ESS/structure_0/state_0', 'ESS/structure_0/state_1'}
    assert jnp.allclose(aux['ESS'], 1.0)
    assert jnp.allclose(aux['ESS/structure_0/state_0'], 1.0)

    # Mask out half the samples of state 0 -> its ESS fraction drops to 0.5.
    half_mask = full_mask.at[:2, 0].set(False)
    aux_half = overlap_penalty._effective_sample_size(
        half_mask, reweighting_factor, excited_systems
    )
    assert jnp.allclose(aux_half['ESS/structure_0/state_0'], 0.5)
    assert jnp.allclose(aux_half['ESS/structure_0/state_1'], 1.0)
    assert jnp.allclose(aux_half['ESS'], 0.75)


def test_overlap_penalty_reweighted_path_runs(
    generalized_wf,
    generalized_wf_params,
    batched_excited_systems,
):
    """Smoke-test the reweighted path: the only branch that runs the relocated
    normalizer-update + reweighting-factor + sample-mask + ESS + cotangent
    un-reweighting code (no other test sets reweight_overlap_mean=True)."""
    penalty = OverlapPenalty(
        wave_function=generalized_wf,
        clipping=NoneClipping(),
        overlap_scaler=NoScaler(triangular_mask=False),
        penalty_scale=1.0,
        masking=QuantileMasking(max_deviation=10.0, quantile=0.95),
    )
    local_energy = jnp.array(
        [[-1.0, -0.75], [-1.1, -0.65]],
        dtype=batched_excited_systems.electrons.dtype,
    )
    (cotangent, aux), *_ = penalty(
        generalized_wf_params,
        batched_excited_systems,
        local_energy,
        penalty.init(),
        jnp.array(0, dtype=jnp.int32),
        True,  # reweight_overlap_mean
    )
    assert_finite(cotangent)
    # Penalty now returns the final per-walker (walker, n_mols) cotangent.
    assert cotangent.shape == local_energy.shape
    assert 'ESS' in aux
    assert 'effective_samples' in aux
    assert 'reweighting_factor_mean' in aux
