import jax
from fixtures import *  # noqa: F403
from utils import assert_finite, assert_shape_and_dtype

from neural_pfaffian.pretraining import Pretraining, PretrainingState
from neural_pfaffian.systems import SystemsWithHF


def test_step(
    pretrainer: Pretraining,
    pretrainer_state: PretrainingState,
    pretraining_systems: SystemsWithHF,
):
    # Test one step
    new_state, new_systems, aux_data = pretrainer.step(
        jax.random.key(8), pretrainer_state, pretraining_systems
    )

    assert_finite(new_state)
    assert_finite(new_systems)
    assert_finite(aux_data)
    assert_shape_and_dtype(new_state, pretrainer_state)
    assert_shape_and_dtype(new_systems, pretraining_systems)

    # Test a second step
    new_state, new_systems, aux_data = pretrainer.step(
        jax.random.key(9), new_state, new_systems
    )

    assert_finite(new_state)
    assert_finite(new_systems)
    assert_finite(aux_data)
    assert_shape_and_dtype(new_state, pretrainer_state)
    assert_shape_and_dtype(new_systems, pretraining_systems)


def test_hf(systems_with_hf):
    orbitals = systems_with_hf.hf_orbitals
    for orbitals, system in zip(orbitals, systems_with_hf):
        n_up, n_down = system.spins[0]
        assert orbitals[0].dtype == system.electrons.dtype
        assert orbitals[1].dtype == system.electrons.dtype
        assert orbitals[0].shape == (*system.electrons.shape[:-2], n_up, n_up)
        assert orbitals[1].shape == (*system.electrons.shape[:-2], n_down, n_down)
