import jax
import numpy as np
from fixtures import *  # noqa: F403

from neural_pfaffian.hamiltonian import (
    local_pp_energy,
    nonlocal_pp_energy,
    potential_energy,
)


def test_attach_pseudopotentials_reduces_to_valence_system(li_pseudopotential_system):
    systems = li_pseudopotential_system

    assert systems.charges == ((3,),)
    assert systems.effective_charges == ((1,),)
    assert systems.spins == ((1, 0),)
    assert systems.electrons.shape == (1, 3)
    assert systems.has_pseudopotentials
    assert systems.pp_data[0].v_loc.shape[0] == 1
    assert systems.pp_data[0].v_nonloc.shape[0] == 1
    assert systems.pp_data[0].v_nonloc.shape[1] > 0


def test_pseudopotential_energies_are_finite_and_different(
    li_all_electron_system,
    li_pseudopotential_system,
):
    all_electron = li_all_electron_system
    valence = li_pseudopotential_system

    all_electron_potential = potential_energy(all_electron)
    valence_coulomb = potential_energy(valence)
    valence_local_pp = local_pp_energy(valence)

    assert np.isfinite(np.asarray(all_electron_potential)).all()
    assert np.isfinite(np.asarray(valence_coulomb)).all()
    assert np.isfinite(np.asarray(valence_local_pp)).all()
    assert not np.allclose(
        np.asarray(all_electron_potential),
        np.asarray(valence_coulomb + valence_local_pp),
    )


def test_nonlocal_pseudopotential_energy_is_finite(
    generalized_wf,
    li_pseudopotential_system,
):
    systems = li_pseudopotential_system
    params = generalized_wf.init(jax.random.PRNGKey(0), systems)

    energy = nonlocal_pp_energy(
        generalized_wf,
        params,
        systems,
        jax.random.PRNGKey(7),
    )

    assert energy.shape == (systems.n_mols,)
    assert np.isfinite(np.asarray(energy)).all()
