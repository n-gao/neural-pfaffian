from fixtures import *  # noqa: F403
from utils import assert_shape_and_dtype

from neural_pfaffian.systems import Systems


def test_splitting(systems: Systems):
    sub_configs = systems.sub_configs
    merged = Systems.merge(sub_configs)

    assert merged.spins == systems.spins
    assert merged.charges == systems.charges

    assert_shape_and_dtype(merged, systems)
