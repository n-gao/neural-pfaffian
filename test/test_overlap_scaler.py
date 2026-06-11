from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from fixtures import *  # noqa: F403
from utils import to_numpy_dict

from neural_pfaffian.overlap_scaler import (
    _ENERGY_STD_EMA_KEY,
    _LOCAL_ENERGY_EMA_KEY,
    EnergyDiffScaler,
)
from neural_pfaffian.utils.schedule import get_schedule


def _run_scaler(
    systems,
    local_energy,
    *,
    asym_strategy: Literal['none', 'softplus', 'step', 'sigmoid'] = 'none',
):
    scaler = EnergyDiffScaler(
        decay_schedule=get_schedule(0.0),
        min_scale_factor=get_schedule(0.0),
        max_scale_factor=100.0,
        asym_strategy=asym_strategy,
        asym_scale=5.0,
    )
    systems = scaler.init_systems(systems)
    alpha, systems = scaler(
        systems,
        local_energy,
        jnp.array(0, dtype=jnp.int32),
    )
    return alpha, systems


@pytest.fixture
def ordered_three_state(three_state_system):
    """Three-state system with 4 walkers and energies monotonically ordered: [0, 1, 3]."""
    systems = three_state_system.replace(
        electrons=jnp.stack([three_state_system.electrons] * 4),
    )
    local_energy = jnp.broadcast_to(
        jnp.array([0.0, 1.0, 3.0], dtype=jnp.float32),
        (4, 3),
    )
    return systems, local_energy


@pytest.fixture
def unordered_three_state(three_state_system):
    """Three-state system with 4 walkers where state 1 is highest in energy: [0, 3, 1]."""
    systems = three_state_system.replace(
        electrons=jnp.stack([three_state_system.electrons] * 4),
    )
    local_energy = jnp.broadcast_to(
        jnp.array([0.0, 3.0, 1.0], dtype=jnp.float32),
        (4, 3),
    )
    return systems, local_energy


@pytest.mark.parametrize(
    'systems',
    ['excited_systems', 'two_excited_systems'],
    indirect=True,
)
def test_shapes_and_initialization(systems):
    local_energy = jnp.ones((4, systems.n_mols), dtype=systems.electrons.dtype)
    alpha, out_systems = _run_scaler(systems, local_energy)

    assert out_systems.get_mol_data(_ENERGY_STD_EMA_KEY) is not None
    assert out_systems.get_mol_data(_LOCAL_ENERGY_EMA_KEY) is not None
    assert alpha.shape == (
        systems.n_unique_mols,
        systems.max_num_states,
        systems.max_num_states,
    )
    assert jnp.allclose(jnp.diagonal(alpha, axis1=-2, axis2=-1), 0.0)


def test_step_asym_ordered_states_is_lower_triangular(ordered_three_state):
    """For energies E0 < E1 < E2, step strategy produces a lower-triangular alpha."""
    systems, local_energy = ordered_three_state
    alpha, *_ = _run_scaler(systems, local_energy, asym_strategy='step')
    alpha = np.asarray(alpha[0])

    assert np.allclose(np.triu(alpha, k=1), 0.0)
    assert alpha[1, 0] > 0
    assert alpha[2, 0] > 0
    assert alpha[2, 1] > 0


def test_step_asym_monotone_in_energy_gap(ordered_three_state):
    """Larger energy gap between states produces a larger overlap weight."""
    systems, local_energy = ordered_three_state
    alpha, *_ = _run_scaler(systems, local_energy, asym_strategy='step')
    alpha = np.asarray(alpha[0])

    # E2-E0=3 > E2-E1=2 > E1-E0=1
    assert alpha[2, 0] > alpha[2, 1] > alpha[1, 0] > 0


def test_step_asym_unordered_energies_zeros_downhill_overlaps(unordered_three_state):
    """alpha[i,j]=0 whenever E_i <= E_j; nonzero only when state i is above state j."""
    systems, local_energy = unordered_three_state
    alpha, *_ = _run_scaler(systems, local_energy, asym_strategy='step')
    alpha = np.asarray(alpha[0])

    # Energies: E0=0, E1=3, E2=1
    # alpha[i,j] nonzero iff E_i > E_j
    assert alpha[1, 0] > 0  # E1=3 > E0=0
    assert alpha[1, 2] > 0  # E1=3 > E2=1
    assert alpha[2, 0] > 0  # E2=1 > E0=0
    assert np.isclose(alpha[0, 1], 0.0)  # E0=0 < E1=3
    assert np.isclose(alpha[0, 2], 0.0)  # E0=0 < E2=1
    assert np.isclose(alpha[2, 1], 0.0)  # E2=1 < E1=3


def test_step_asym_differs_from_none_for_unordered_energies(unordered_three_state):
    """'step' produces non-lower-triangular alpha for unordered energies, unlike 'none'."""
    systems, local_energy = unordered_three_state
    alpha_none, *_ = _run_scaler(systems, local_energy, asym_strategy='none')
    alpha_step, *_ = _run_scaler(systems, local_energy, asym_strategy='step')

    alpha_none = np.asarray(alpha_none[0])
    alpha_step = np.asarray(alpha_step[0])

    # 'none' is always lower-triangular; 'step' has alpha[1,2]>0 since E1=3 > E2=1
    assert np.isclose(alpha_none[1, 2], 0.0)
    assert alpha_step[1, 2] > 0


# ---------------------------------------------------------------------------
# Regression tests — pin exact numerics to catch silent changes
# ---------------------------------------------------------------------------


def test_energy_diff_scaler_regression(
    ndarrays_regression,
    ordered_three_state,
    unordered_three_state,
):
    """Pins the numerical output of EnergyDiffScaler across all asymmetrization strategies.

    Regenerate after deliberate changes:
        uv run pytest -n0 test/test_overlap_scaler.py::test_energy_diff_scaler_regression --regen-all
    """
    systems_ord, energy_ord = ordered_three_state
    systems_unord, energy_unord = unordered_three_state

    ndarrays_regression.check(
        to_numpy_dict(
            {
                # Ordered energies [0, 1, 3]: all three strategies
                'alpha_none': _run_scaler(systems_ord, energy_ord, asym_strategy='none')[
                    0
                ],
                'alpha_step': _run_scaler(systems_ord, energy_ord, asym_strategy='step')[
                    0
                ],
                'alpha_sigmoid': _run_scaler(
                    systems_ord,
                    energy_ord,
                    asym_strategy='sigmoid',
                )[0],
                # Unordered energies [0, 3, 1]: pin the non-trivial step case
                'alpha_step_unordered': _run_scaler(
                    systems_unord,
                    energy_unord,
                    asym_strategy='step',
                )[0],
            },
        ),
        default_tolerance={'rtol': 1e-5, 'atol': 1e-5},
    )
