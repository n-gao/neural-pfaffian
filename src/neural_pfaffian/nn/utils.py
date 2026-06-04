import functools
from collections.abc import Callable, Sequence
from typing import TypeVar

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from neural_pfaffian.utils.jax_utils import vectorize


def residual(x: Array, y: Array):
    if x.shape == y.shape:
        return (x + y) / jnp.sqrt(2)
    return y


def log1p_rescale(x: Float[Array, '... 3+1']):
    return x / x[..., -1:] * jnp.log1p(x[..., -1:])


T = TypeVar('T', bound=Array)

ActivationOrName = Callable[[T], T]


class Activation(nn.Module):
    activation: ActivationOrName

    def __call__(self, x: Float[Array, ' ...']) -> Float[Array, ' ...']:
        if callable(self.activation):
            return self.activation(x)
        return getattr(nn, self.activation)(x)


class GatedLinearUnit(nn.Module):
    dim: int | tuple[int, ...]
    activation: ActivationOrName
    hidden_dim: int | None = None
    normalize: bool = True
    out_std: ArrayLike = 1.0

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.hidden_dim is None:
            assert isinstance(self.dim, int)
            hidden_dim = self.dim
        else:
            hidden_dim = self.hidden_dim
        if self.normalize:
            x = nn.LayerNorm()(x)
        hidden = Activation(self.activation)(
            nn.Dense(hidden_dim, use_bias=False)(x),
        ) * nn.Dense(hidden_dim, use_bias=False)(x)

        # Fold out_std into the kernel init: variance_scaling(out_std**2, ...)
        # reproduces nn.Dense's default lecun_normal at out_std == 1.0 (most
        # callers) and rescales the output otherwise (ParamOut passes meta.std).
        OutDense = functools.partial(
            nn.Dense,
            use_bias=False,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.out_std**2, 'fan_in', 'truncated_normal'
            ),
        )

        if isinstance(self.dim, int):
            return OutDense(self.dim)(hidden)

        output_shape = tuple(self.dim)

        return OutDense(int(np.prod(output_shape)))(hidden).reshape(
            *hidden.shape[:-1],
            *output_shape,
        )


@vectorize(signature='(a,b)->(c,d)', excluded={1, 2, 3})
def pad_block_constant(
    x: Array,
    top_right: ArrayLike,
    bottom_left: ArrayLike,
    bottom_right: ArrayLike,
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
        ],
    )


class MLP(nn.Module):
    dims: Sequence[int]
    activation: ActivationOrName

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        activation = Activation(self.activation)
        for dim in self.dims[:-1]:
            x = activation(nn.Dense(dim)(x))
        x = nn.Dense(self.dims[-1])(x)
        return x


def normal_init(mean: Float[ArrayLike, ''], std: Float[ArrayLike, '']):
    def init(key: jax.Array, shape: Sequence[int], dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype=dtype) * jnp.array(
            std,
            dtype=dtype,
        ) + jnp.array(mean, dtype=dtype)

    return init
