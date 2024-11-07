from typing import Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float

from neural_pfaffian.nn.module import ReparamModule
from neural_pfaffian.nn.utils import Activation, ActivationOrName
from neural_pfaffian.systems import Systems


def shifted_normal(shift: float = 0, scale: float = 1):
    return lambda key, shape: jax.random.normal(key, shape, jnp.float32) * scale + shift


class Rbf(ReparamModule):
    out_dim: int
    max_charge: int | None
    sigma_init: float = 10

    def __call__(
        self,
        systems: Systems,
        edges: Float[Array, '... 3+1'],
        center_idx: npt.NDArray[np.int64] | None,
    ) -> Float[Array, '... 3+1']:
        raise NotImplementedError


class ExponentialRbf(Rbf):
    @nn.compact
    def __call__(
        self,
        systems: Systems,
        edges: Float[Array, '... 3+1'],
        center_idx: npt.NDArray[np.int64] | None,
    ):
        sigma = self.edge_reparam(
            'sigma',
            systems,
            shifted_normal(self.sigma_init),
            (self.out_dim,),
            self.max_charge,
            center_idx,
        )
        sigma = nn.softplus(sigma)
        return jnp.exp(-jnp.abs(edges[..., -1:] / sigma))


class BesselRbf(Rbf):
    @nn.compact
    def __call__(
        self,
        systems: Systems,
        edges: Float[Array, '... 3+1'],
        center_idx: npt.NDArray[np.int64] | None,
    ):
        sigma = nn.softplus(
            self.edge_reparam(
                'sigma',
                systems,
                shifted_normal(self.sigma_init),
                (self.out_dim,),
                self.max_charge,
                center_idx,
            )
        )
        frequencies = jnp.arange(1, self.out_dim + 1) * jnp.pi

        safe_dist = edges[..., -1:] + 1e-6
        safe_dist /= 2 * sigma  # we set the cutoff to be 2 * sigma
        result = jnp.sqrt(2.0 / sigma) * jnp.sin(frequencies * safe_dist) / safe_dist
        env = jnp.exp(-safe_dist)
        return result * env


class EdgeEmbedding(ReparamModule):
    out_dim: int
    hidden_dim: int
    n_rbf: int
    activation: ActivationOrName
    rbf: Type[Rbf]
    # If set, we use charge embeddings instead of .reparam
    # If set to -1, we use the same embedding for all nuclei
    max_charge: int | None
    sigma_init: float = 10

    @nn.compact
    def __call__(
        self,
        systems: Systems,
        edges: Float[Array, '... 3+1'],
        center_idx: npt.NDArray[np.int64] | None = None,
    ):
        bias = self.edge_reparam(
            'bias',
            systems,
            jax.nn.initializers.normal(2, jnp.float32),
            (self.hidden_dim,),
            self.max_charge,
            center_idx,
        )
        kernel = self.edge_reparam(
            'kernel',
            systems,
            jax.nn.initializers.normal(1 / 2, jnp.float32),
            (4, self.hidden_dim),
            self.max_charge,
            center_idx,
            keep_distr=True,
        )

        hidden = jnp.einsum('...d,...dk->...k', edges, kernel) + bias
        hidden = Activation(self.activation)(hidden)
        env = self.rbf(self.n_rbf, self.max_charge, self.sigma_init)(
            systems, edges, center_idx
        )
        hidden = (hidden[..., None] * env[..., None, :]).reshape(
            *hidden.shape[:-1], self.hidden_dim * self.n_rbf
        )
        return nn.Dense(self.out_dim, use_bias=False)(hidden)


class NormEnvelope(ReparamModule):
    # If set, we use charge embeddings instead of .reparam
    max_charge: int | None = None
    sigma_init: float = 10

    @nn.compact
    def __call__(
        self,
        systems: Systems,
        edges: Float[Array, '... 3+1'],
        center_idx: npt.NDArray[np.int64] | None = None,
    ):
        return ExponentialRbf(1, self.max_charge, self.sigma_init)(
            systems, edges, center_idx
        ).squeeze(-1)
