import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fixtures import *  # noqa: F403

from neural_pfaffian.hamiltonian import (
    KineticEnergyOp,
    make_kinetic_energy,
    potential_energy,
)
from neural_pfaffian.nn.wave_function import GeneralizedWaveFunction, WaveFunction
from neural_pfaffian.systems import Systems


def test_potential_energy(systems):
    pot = potential_energy(systems)
    assert pot.shape == (systems.n_mols,)
    assert pot.dtype == systems.electrons.dtype
    assert np.isfinite(pot).all()


@pytest.mark.parametrize('operator', [KineticEnergyOp.LOOP, KineticEnergyOp.FORWARD])
def test_kinetic_energy(
    neural_pfaffian, neural_pfaffian_params, systems, operator, clear_cache_each_time
):
    kin_fn = make_kinetic_energy(neural_pfaffian, operator)
    kin = kin_fn(neural_pfaffian_params, systems)
    assert kin.shape == (systems.n_mols,)
    assert kin.dtype == systems.electrons.dtype
    assert np.isfinite(kin).all()


@pytest.fixture(scope='module')
def row_sparse_system():
    # 6 electrons keep folx' sparsity threshold above the 3 coordinates an
    # orbital depends on, so the row-sparse determinant rule is used
    return Systems(
        spins=((3, 3),),
        charges=((3, 3),),
        electrons=jax.random.normal(jax.random.key(0), (6, 3), dtype=jnp.float64),
        nuclei=jax.random.normal(jax.random.key(1), (2, 3), dtype=jnp.float64),
        mol_data={},
    )


def test_row_sparse_kinetic_energy(
    row_sparse_system, fire_local, fermi_sets, cusp_jastrow, clear_cache_each_time
):
    wf = GeneralizedWaveFunction.create(
        WaveFunction(fire_local, fermi_sets, cusp_jastrow), None, row_sparse_system
    )
    params = wf.init(jax.random.key(42), row_sparse_system)
    forward = make_kinetic_energy(wf, KineticEnergyOp.FORWARD)(params, row_sparse_system)
    loop = make_kinetic_energy(wf, KineticEnergyOp.LOOP)(params, row_sparse_system)
    np.testing.assert_allclose(forward, loop, rtol=1e-10)
