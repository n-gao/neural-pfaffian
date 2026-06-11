import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope='module')
def _clear_jax_caches():
    jax.clear_caches()
    gc.collect()
    yield
    jax.clear_caches()
    gc.collect()


@pytest.fixture
def _clear_jax_caches_per_test():
    """Opt-in per-function cache clearing for tests that need full isolation."""
    jax.clear_caches()
    gc.collect()
    yield
    jax.clear_caches()
    gc.collect()
