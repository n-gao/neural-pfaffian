import jax.numpy as jnp
from numpy.testing import assert_allclose
from utils import assert_shape_and_dtype

from neural_pfaffian.clipping import (
    MeanClipping,
    MedianClipping,
    NoneClipping,
    QuantileClipping,
)


def test_mean_clipping():
    clipping = MeanClipping(1.0)
    x = jnp.array([-1, 0, 1], dtype=jnp.float32)[:, None]
    y = clipping(x)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, x * 2 / 3)


def test_median_clipping():
    clipping = MedianClipping(1.0)
    x = jnp.array([-1, 0, 1], dtype=jnp.float32)[:, None]
    y = clipping(x)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, x * 2 / 3)


def test_no_clipping():
    clipping = NoneClipping()
    x = jnp.array([-1, 0, 1], dtype=jnp.float32)[:, None]
    y = clipping(x)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, x)


def test_quantile_clipping():
    clipping = QuantileClipping(0.5, 1.0)
    x = jnp.array([-1, 0, 2], dtype=jnp.float32)[:, None]
    y = clipping(x)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, jnp.array([-1, 0, 1])[:, None])
