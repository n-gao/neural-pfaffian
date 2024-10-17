import jax
from fixtures import *  # noqa: F403

from neural_pfaffian.systems import Systems


def test_splitting(systems: Systems):
    static = systems.static_args
    data = systems.data_args
    # Static
    assert static.spins == systems.spins
    assert static.charges == systems.charges
    assert static.electrons is None
    assert static.nuclei is None
    assert static.mol_data is None
    # Data
    assert data.spins is None
    assert data.charges is None
    assert (data.electrons == systems.electrons).all()
    assert (data.nuclei == systems.nuclei).all()
    assert data.mol_data is not None
    # Merging
    merged = systems.merge_static_and_data(static, data)

    def assert_equal(a, b):
        assert (a == b).all()

    jax.tree_util.tree_map(assert_equal, systems, merged)
