from typing import Callable, TypeVar

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


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
