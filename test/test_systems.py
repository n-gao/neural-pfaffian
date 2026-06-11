import pytest
from fixtures import *  # noqa: F403
from utils import assert_shape_and_dtype

from neural_pfaffian.systems import Systems


def test_splitting(systems: Systems):
    sub_configs = systems.sub_configs
    merged = Systems.merge(sub_configs)

    assert merged.spins == systems.spins
    assert merged.charges == systems.charges
    assert merged.excitations == systems.excitations

    assert_shape_and_dtype(merged, systems)


def test_safe_batch(excited_systems: Systems):
    batches = Systems.safe_batch(excited_systems, 2)
    assert len(batches) == 1
    assert batches[0] == excited_systems

    with pytest.raises(ValueError):
        Systems.safe_batch(excited_systems, 1)
