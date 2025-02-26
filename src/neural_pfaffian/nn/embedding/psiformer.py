from enum import Enum
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float

from neural_pfaffian.nn.ops import segment_softmax
from neural_pfaffian.nn.utils import Activation, ActivationOrName
from neural_pfaffian.nn.wave_function import EmbeddingP
from neural_pfaffian.systems import Systems, chunk_electron

from .ferminet import FermiNetFeatures

SingleStream = Float[Array, 'n_elec single_dim']


class AttentionImplementation(Enum):
    ITERATIVE = 'iterative'
    PARALLEL = 'parallel'


def iterative_attention(
    Q: SingleStream, K: SingleStream, V: SingleStream, systems: Systems
):
    heads, dim = Q.shape[-2:]
    result = []
    for q, k, v in zip(
        systems.group(Q, chunk_electron),
        systems.group(K, chunk_electron),
        systems.group(V, chunk_electron),
    ):
        A = jnp.einsum('...ahd,...bhd->...abh', q, k) / jnp.sqrt(dim)
        A = jax.nn.softmax(A, axis=-2)
        attn = jnp.einsum('...abh,...bhd->...ahd', A, v).reshape(*v.shape[:-2], -1)
        result.extend(list(attn))
    return jnp.concatenate([result[i] for i in systems.inverse_unique_indices])


def parallel_attention(
    Q: SingleStream, K: SingleStream, V: SingleStream, systems: Systems
):
    e_e_i, e_e_j, _ = systems.elec_elec_idx
    n, heads, dim = Q.shape
    A = jnp.einsum('...ahd,...ahd->...ah', Q[e_e_i], K[e_e_j]) / jnp.sqrt(dim)
    A = segment_softmax(A, e_e_j, n)
    attn = jax.ops.segment_sum(jnp.einsum('...ah,...ahd->...ahd', A, V[e_e_j]), e_e_i, n)
    return attn.reshape(n, -1)


class Attention(nn.Module):
    dim: int
    heads: int
    activation: ActivationOrName
    attention_implementation: AttentionImplementation

    @nn.compact
    def __call__(self, systems: Systems, h_one: SingleStream):
        assert self.dim % self.heads == 0, 'dim must be divisible by heads'
        Q, K, V = jnp.split(nn.Dense(self.dim * 3, use_bias=False)(h_one), 3, axis=-1)
        Q, K, V = jtu.tree_map(
            lambda x: x.reshape(*x.shape[:-1], self.heads, -1), (Q, K, V)
        )
        match AttentionImplementation(self.attention_implementation):
            case AttentionImplementation.ITERATIVE:
                attn = iterative_attention(Q, K, V, systems)
            case AttentionImplementation.PARALLEL:
                attn = parallel_attention(Q, K, V, systems)
            case _:
                raise ValueError(
                    f'Unknown attention implementation: {self.attention_implementation}'
                )
        assert attn.shape == h_one.shape
        # this layer is technically redundant but it's in the original code
        h_one += nn.Dense(self.dim, use_bias=False)(attn)
        h_one = nn.LayerNorm()(h_one)

        mlp_out = Activation(self.activation)(nn.Dense(self.dim)(h_one))

        h_one += mlp_out
        h_one = nn.LayerNorm()(h_one)
        return h_one


class PsiFormer(nn.Module, EmbeddingP):
    embedding_dim: int
    dim: int
    n_head: int
    n_layer: int
    activation: ActivationOrName
    attention_implementation: AttentionImplementation

    @nn.compact
    def __call__(self, systems: Systems):
        assert self.dim == self.embedding_dim, 'dim must be equal to embedding_dim'
        h_one, _ = FermiNetFeatures(self.embedding_dim, False, False)(systems)
        # Spin embedding
        spin_emb = self.param(
            'spin_emb',
            jax.nn.initializers.normal(stddev=1 / jnp.sqrt(self.embedding_dim)),
            (h_one.shape[-1],),
            jnp.float32,
        )
        h_one += spin_emb * (2 * systems.spin_mask - 1)[:, None].astype(jnp.float32)
        for _ in range(self.n_layer):
            h_one = Attention(
                self.dim, self.n_head, self.activation, self.attention_implementation
            )(systems, h_one)
        return h_one
