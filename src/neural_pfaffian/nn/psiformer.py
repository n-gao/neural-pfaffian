import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float

from neural_pfaffian.nn.ferminet import FermiNetFeatures
from neural_pfaffian.nn.ops import segment_softmax
from neural_pfaffian.nn.utils import Activation, ActivationOrName
from neural_pfaffian.nn.wave_function import EmbeddingP
from neural_pfaffian.systems import Systems

SingleStream = Float[Array, 'n_elec single_dim']


class Attention(nn.Module):
    dim: int
    heads: int
    activation: ActivationOrName

    @nn.compact
    def __call__(self, systems: Systems, h_one: SingleStream):
        n = h_one.shape[0]
        e_e_i, e_e_j, _ = systems.elec_elec_idx

        Q, K, V = jnp.split(nn.Dense(self.dim * 3)(h_one), 3, axis=-1)
        Q, K, V = jtu.tree_map(
            lambda x: x.reshape(*x.shape[:-1], self.heads, -1), (Q, K, V)
        )
        A = jnp.einsum('...ahd,...ahd->...ah', Q[e_e_i], K[e_e_j]) / jnp.sqrt(
            self.dim / self.heads
        )
        A = segment_softmax(A, e_e_j, n)
        attn = jax.ops.segment_sum(
            jnp.einsum('...ah,...ahd->...ahd', A, V[e_e_j]), e_e_i, n
        )
        attn = attn.reshape(n, -1)

        h_one += attn
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

    @nn.compact
    def __call__(self, systems: Systems):
        assert self.dim == self.embedding_dim, 'dim must be equal to embedding_dim'
        h_one, _ = FermiNetFeatures(self.embedding_dim)(systems)
        # Spin embedding
        h_one += nn.Embed(2, self.embedding_dim)(systems.spin_mask)
        for _ in range(self.n_layer):
            h_one = Attention(self.dim, self.n_head, self.activation)(systems, h_one)
        return h_one
