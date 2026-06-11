import jax
import jax.numpy as jnp
import numpy as np

from neural_pfaffian.utils import EMA, RollingAverage


def to_numpy_dict(data: dict[str, jax.Array]) -> dict[str, np.ndarray]:
    return {key: np.asarray(jax.device_get(value)) for key, value in data.items()}


def assert_not_float64(x):
    def assert_non_float64(path, x):
        if isinstance(x, jax.Array):
            assert x.dtype != jnp.float64, f'{path} is float64'

    jax.tree_util.tree_map_with_path(assert_non_float64, x)


def assert_finite(x):
    # EMA and RollingAverage own buffers that are intentionally NaN-initialized
    # (see neural_pfaffian.utils), so we skip them rather than flag expected NaNs.
    def is_nan_by_design(node):
        return isinstance(node, (EMA, RollingAverage))

    def assert_finite(path, x):
        if is_nan_by_design(x):
            return
        if isinstance(x, jax.Array):
            assert np.isfinite(x).all(), f'{path} is not finite'

    jax.tree_util.tree_map_with_path(assert_finite, x, is_leaf=is_nan_by_design)


def assert_shape_and_dtype(x, y):
    def assert_shape(path, x, y):
        if isinstance(x, jax.Array):
            assert x.shape == y.shape, f'{path} shape mismatch {x.shape} != {y.shape}'
            assert x.dtype == y.dtype, f'{path} dtype mismatch {x.dtype} != {y.dtype}'

    jax.tree_util.tree_map_with_path(assert_shape, x, y)
