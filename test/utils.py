import jax
import jax.numpy as jnp


def assert_not_float64(x):
    def assert_non_float64(path, x):
        if isinstance(x, jax.Array):
            assert x.dtype != jnp.float64, f'{path} is float64'

    jax.tree_util.tree_map_with_path(assert_non_float64, x)


def assert_finite(x):
    def assert_finite(path, x):
        if isinstance(x, jax.Array):
            assert jnp.isfinite(x).all(), f'{path} is not finite'

    jax.tree_util.tree_map_with_path(assert_finite, x)


def assert_shape_and_dtype(x, y):
    def assert_shape(path, x, y):
        if isinstance(x, jax.Array):
            assert x.shape == y.shape, f'{path} shape mismatch {x.shape} != {y.shape}'
            assert x.dtype == y.dtype, f'{path} dtype mismatch {x.dtype} != {y.dtype}'

    jax.tree_util.tree_map_with_path(assert_shape, x, y)
