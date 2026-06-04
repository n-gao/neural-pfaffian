import jax
import pytest
import yaml

from neural_pfaffian.dataset import create_systems


@pytest.fixture
def single_mol_sample():
    cfg = """
    molecules:
        - ["diatomic", { charge1: 7, charge2: 7, distance: 2.02858 }]

    num_walker_per_mol: 4096
    """
    return yaml.safe_load(cfg)


@pytest.fixture
def excited_mol_sample():
    cfg = """
    molecules:
      - - excited
        - total_spins: 0
          n_states: 4
          molecules:
            - [diatomic, { charge1: 7, charge2: 7, distance: 2.02858 }]
    num_walker_per_mol: 4096
    """
    return yaml.safe_load(cfg)


@pytest.fixture
def multiple_mol_sample():
    cfg = """
    molecules:
        - ["diatomic", { charge1: 7, charge2: 7, distance: 1.60151 }]
        - ["diatomic", { charge1: 7, charge2: 7, distance: 2.02858 }]

    num_walker_per_mol: 4096
    """
    return yaml.safe_load(cfg)


@pytest.fixture
def excited_multiple_mol_sample():
    cfg = """
    molecules:
        - - excited
          - total_spins: 0
            n_states: 4
            molecules:
                - ["diatomic", { charge1: 7, charge2: 7, distance: 1.60151 }]
                - ["diatomic", { charge1: 7, charge2: 7, distance: 2.02858 }]
    num_walker_per_mol: 4096
    """
    return yaml.safe_load(cfg)


def test_create_systems(
    single_mol_sample,
    excited_mol_sample,
    multiple_mol_sample,
    excited_multiple_mol_sample,
):
    systems = create_systems(jax.random.PRNGKey(0), **single_mol_sample)
    assert len(systems) == 1
    assert systems.excitations == (0,)

    systems = create_systems(jax.random.PRNGKey(0), **excited_mol_sample)
    assert len(systems) == 4
    assert systems.excitations == (0, 1, 2, 3)
    assert systems.mol_ids == (0,) * 4

    systems = create_systems(jax.random.PRNGKey(0), **multiple_mol_sample)
    assert len(systems) == 2
    assert systems.excitations == (0, 0)
    assert systems.mol_ids == (0, 1)

    systems = create_systems(jax.random.PRNGKey(0), **excited_multiple_mol_sample)
    assert len(systems) == 8
    assert systems.excitations == (0, 1, 2, 3, 0, 1, 2, 3)
    assert systems.mol_ids == (0,) * 4 + (1,) * 4
