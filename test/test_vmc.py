import jax
from fixtures import *  # noqa: F403
from utils import assert_finite, assert_not_float64, assert_shape_and_dtype

from neural_pfaffian.systems import Systems
from neural_pfaffian.vmc import VMC, VMCState


def test_dtypes(vmc_state: VMCState, vmc_systems: Systems):
    assert_not_float64(vmc_state.params)
    assert_not_float64(vmc_systems)
    assert_finite(vmc_state)
    assert_finite(vmc_systems)


def test_step(vmc: VMC, vmc_state: VMCState, vmc_systems: Systems):
    # Test one step
    new_state, new_systems, aux_data = vmc.step(jax.random.key(8), vmc_state, vmc_systems)

    assert_finite(new_state)
    assert_finite(new_systems)
    assert_finite(aux_data)
    assert_shape_and_dtype(new_state, vmc_state)
    assert_shape_and_dtype(new_systems, vmc_systems)

    # Test a second step
    new_state, new_systems, aux_data = vmc.step(jax.random.key(9), new_state, new_systems)

    assert_finite(new_state)
    assert_finite(new_systems)
    assert_finite(aux_data)
    assert_shape_and_dtype(new_state, vmc_state)
    assert_shape_and_dtype(new_systems, vmc_systems)
