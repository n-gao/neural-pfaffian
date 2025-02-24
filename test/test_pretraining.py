from chex import assert_trees_all_close
import jax
from fixtures import *  # noqa: F403
from neural_pfaffian.hf import AOImplementation, make_hf_orbitals
from utils import assert_finite, assert_shape_and_dtype

from neural_pfaffian.pretraining import Pretraining, PretrainingState
from neural_pfaffian.systems import SystemsWithHF, chunk_electron


def test_jax_orbitals(systems):
    mols = systems.pyscf_molecules('aug-ccpvtz')
    for m, elecs in zip(mols, systems.group(systems.electrons, chunk_electron)):
        pyscf_fn = make_hf_orbitals(m, AOImplementation.PYSCF)
        jax_fn = make_hf_orbitals(m, AOImplementation.JAX)
        pyscf_up, pyscf_down = pyscf_fn(elecs)
        jax_up, jax_down = jax_fn(elecs)
        assert_finite(jax_up)
        assert_finite(jax_down)
        assert_finite(pyscf_up)
        assert_finite(pyscf_down)
        assert_shape_and_dtype(jax_up, pyscf_up)
        assert_shape_and_dtype(jax_down, pyscf_down)
        assert_trees_all_close(jax_up, pyscf_up, rtol=1e-5)
        assert_trees_all_close(jax_down, pyscf_down, rtol=1e-5)


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
