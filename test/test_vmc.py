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


def test_mcmc(vmc: VMC, vmc_state: VMCState, vmc_systems: Systems):
    # Test one step
    new_systems, aux_data = vmc.mcmc_step(
        jax.random.key(8), vmc_state.sharded, vmc_systems.sharded
    )

    assert_finite(new_systems)
    assert_finite(aux_data)
    assert_shape_and_dtype(new_systems, vmc_systems)

    # Test a second step
    new_systems, aux_data = vmc.mcmc_step(
        jax.random.key(9), vmc_state.sharded, new_systems.sharded
    )

    assert_finite(new_systems)
    assert_finite(aux_data)
    assert_shape_and_dtype(new_systems, vmc_systems)


def test_energy(vmc: VMC, vmc_state: VMCState, vmc_systems: Systems):
    e_l = vmc.local_energy(vmc_state, vmc_systems)
    assert_finite(e_l)
    assert e_l.shape == (vmc_systems.electrons.shape[0], vmc_systems.n_mols)
    assert e_l.dtype == vmc_systems.electrons.dtype


def test_step(vmc: VMC, vmc_state: VMCState, vmc_systems: Systems, clear_cache_each_time):
    # Test one step
    new_state, new_systems, aux_data = vmc.step(
        jax.random.key(8), vmc_state.sharded, vmc_systems.sharded
    )

    assert_finite(new_state)
    assert_finite(new_systems)
    assert_finite(aux_data)
    assert_shape_and_dtype(new_state, vmc_state)
    assert_shape_and_dtype(new_systems, vmc_systems)

    # Test a second step
    new_state, new_systems, aux_data = vmc.step(
        jax.random.key(9), new_state.sharded, new_systems.sharded
    )

    assert_finite(new_state)
    assert_finite(new_systems)
    assert_finite(aux_data)
    assert_shape_and_dtype(new_state, vmc_state)
    assert_shape_and_dtype(new_systems, vmc_systems)
