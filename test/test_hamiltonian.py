import numpy as np
import pytest
from fixtures import *  # noqa: F403

from neural_pfaffian.hamiltonian import (
    KineticEnergyOp,
    make_kinetic_energy,
    potential_energy,
)


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
