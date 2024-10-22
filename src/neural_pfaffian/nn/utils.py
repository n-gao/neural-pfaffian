from typing import Callable, TypeVar

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from neural_pfaffian.utils.jax_utils import vectorize


def residual(x: Array, y: Array):
    if x.shape == y.shape:
        return (x + y) / jnp.sqrt(2)
    return y


def log1p_rescale(x: Float[Array, '... 3+1']):
    return x / x[..., -1:] * jnp.log1p(x[..., -1:])


T = TypeVar('T', bound=Array)

Activation = Callable[[T], T]


class GatedLinearUnit(nn.Module):
    dim: int
    activation: Activation
    hidden_dim: int | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        hidden_dim = self.dim if self.hidden_dim is None else self.hidden_dim
        x = nn.LayerNorm()(x)
        return nn.Dense(self.dim, use_bias=False)(
            self.activation(nn.Dense(hidden_dim, use_bias=False)(x))
            * nn.Dense(hidden_dim, use_bias=False)(x)
        )


@vectorize(signature='(a,b)->(c,d)', excluded={1, 2, 3})
def pad_block_constant(
    x: Array, top_right: ArrayLike, bottom_left: ArrayLike, bottom_right: ArrayLike
):
    n, m = x.shape
    dtype = x.dtype
    tr = jnp.full((n, 1), top_right, dtype=dtype)
    bl = jnp.full((1, m), bottom_left, dtype=dtype)
    br = jnp.full((1, 1), bottom_right, dtype=dtype)
    return jnp.block([[x, tr], [bl, br]])


@vectorize(signature='(a,b),(a),(b),()->(c,d)')
def block(x: Array, top_right: Array, bottom_left: Array, bottom_right: Array):
    return jnp.block(
        [
            [x, top_right[:, None]],
            [bottom_left[None], bottom_right[None, None]],
        ]
    )
