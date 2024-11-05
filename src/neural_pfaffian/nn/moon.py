from typing import Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Float

from neural_pfaffian.nn.edges import EdgeEmbedding, ExponentialRbf, NormEnvelope
from neural_pfaffian.nn.module import ParamTypes, ReparamModule
from neural_pfaffian.nn.ops import segment_sum
from neural_pfaffian.nn.utils import Activation, log1p_rescale, residual
from neural_pfaffian.nn.wave_function import EmbeddingP
from neural_pfaffian.systems import Systems

ElecEmbedding = Float[Array, 'n_elec embedding_dim']
NucEmbeddings = tuple[
    Float[Array, 'n_nuc embedding_dim'], Float[Array, 'n_nuc embedding_dim']
]
ElecNucEdge = Float[Array, 'n_elec_nuc edge_embedding']
ElecNormalizer = Float[Array, 'n_elec']


class MoonEmbeddingElecElec(ReparamModule):
    embedding_dim: int
    hidden_dim: int
    rbf: int
    activation: Activation

    @nn.compact
    def __call__(self, systems: Systems) -> ElecEmbedding:
        r_ij = systems.elec_elec_dists
        r_ij_same, r_ij_diff = (
            r_ij[: systems.n_elec_pair_same],
            r_ij[systems.n_elec_pair_same :],
        )

        # Electron - Electron Embedding
        same_ij = EdgeEmbedding(
            self.embedding_dim // 2,
            self.hidden_dim,
            self.rbf,
            self.activation,
            ExponentialRbf,
            -1,
        )(systems, r_ij_same, None)
        diff_ij = EdgeEmbedding(
            self.embedding_dim // 2,
            self.hidden_dim,
            self.rbf,
            self.activation,
            ExponentialRbf,
            -1,
        )(systems, r_ij_diff, None)

        e_e_filter = jnp.concatenate([same_ij, diff_ij], axis=0)
        e_e_data = nn.Dense(self.embedding_dim // 2)(log1p_rescale(r_ij))
        e_e_data = self.activation(e_e_data)
        # Here we merge two segment sums for efficiency
        e_e_i = systems.elec_elec_idx[0]
        # one sum for same spin, one for different
        e_e_i[systems.n_elec_pair_same :] += systems.n_elec
        e_emb = segment_sum(e_e_filter * e_e_data, e_e_i, 2 * systems.n_elec)
        result = einops.rearrange(e_emb, '(two elec) feat -> elec (two feat)', two=2)
        return result


class MoonEmbedding(ReparamModule):
    embedding_dim: int
    activation: Activation

    edge_embedding: int
    edge_hidden_dim: int
    edge_rbf: int

    @nn.compact
    def __call__(
        self, systems: Systems
    ) -> Tuple[ElecEmbedding, NucEmbeddings, ElecNucEdge, ElecNormalizer]:
        # Electron-electron embedding
        elec_emb = MoonEmbeddingElecElec(
            self.embedding_dim, self.edge_hidden_dim, self.edge_rbf, self.activation
        )(systems)

        # Normalize by number of neighbors
        elec_nuc_dists = systems.elec_nuc_dists
        nuc_idx = systems.elec_nuc_idx[1]
        e_env = NormEnvelope()(systems, elec_nuc_dists)
        e_normalizer = segment_sum(
            e_env * systems.flat_charges[nuc_idx],
            systems.elec_nuc_idx[0],
            systems.n_elec,
        )
        e_normalizer += 1
        elec_emb /= e_normalizer[..., None]

        # Electron - nucleus embedding
        kernel = self.reparam(
            'kernel',
            jax.nn.initializers.normal(1 / 2, dtype=jnp.float32),
            (systems.n_nuc, 4, self.embedding_dim),
            param_type=ParamTypes.NUCLEI,
            keep_distr=True,
        )[0][nuc_idx]
        bias = self.reparam(
            'bias',
            jax.nn.initializers.normal(1.0, dtype=jnp.float32),
            (systems.n_nuc, self.embedding_dim),
            param_type=ParamTypes.NUCLEI,
        )[0][nuc_idx]
        elec_nuc_emb = log1p_rescale(elec_nuc_dists)
        elec_nuc_emb = jnp.einsum('...d,...dk->...k', elec_nuc_emb, kernel) + bias
        elec_nuc_emb += elec_emb[systems.elec_nuc_idx[0]]
        elec_nuc_emb = self.activation(elec_nuc_emb)

        # Electron - nuclei filter
        elec_nuc_edge = EdgeEmbedding(
            self.edge_embedding,
            self.edge_hidden_dim,
            self.edge_rbf,
            self.activation,
            ExponentialRbf,
            None,
        )(systems, elec_nuc_dists, nuc_idx)
        elec_nuc_scale = nn.Dense(2 * self.embedding_dim, use_bias=False)(elec_nuc_edge)
        elec_nuc_scale = einops.rearrange(
            elec_nuc_scale, 'edges (two feat) -> two edges feat', two=2
        )
        elec_nuc_emb = elec_nuc_emb * elec_nuc_scale

        # Aggregate to electron and nucleus embeddings
        # We merge a three segment sum for efficiency
        mask = systems.spin_mask[systems.elec_elec_idx[0]]
        e_n_i, e_n_m, _ = systems.elec_nuc_idx
        e_n_m += systems.n_elec
        e_n_m[mask] += systems.n_nuc  # aggregate separately for each spin
        aggregate_inp = segment_sum(
            elec_nuc_emb.reshape(-1, self.embedding_dim),
            np.concatenate([e_n_i, e_n_m]),
            systems.n_elec + 2 * systems.n_nuc,
        )
        elec_emb = aggregate_inp[: systems.n_elec]
        nuc_emb = (
            aggregate_inp[systems.n_elec : systems.n_elec + systems.n_nuc],
            aggregate_inp[systems.n_elec + systems.n_nuc :],
        )
        # Normalize by neighbor count
        elec_emb /= e_normalizer[..., None]

        n_neigh = segment_sum(
            NormEnvelope()(systems, systems.nuc_nuc_dists)
            * systems.flat_charges[systems.nuc_nuc_idx[1]],
            systems.nuc_nuc_idx[0],
            systems.n_nuc,
        )[..., None]
        nuc_emb = jtu.tree_map(lambda x: x / (n_neigh + 1), nuc_emb)
        return elec_emb, nuc_emb, elec_nuc_edge, e_normalizer


class Update(ReparamModule):
    out_dim: int
    activation: Activation

    @nn.compact
    def __call__(self, nuc_emb: NucEmbeddings) -> NucEmbeddings:
        n_nuc = nuc_emb[0].shape[0]
        same = nn.Dense(self.out_dim)
        diff = nn.Dense(self.out_dim)
        up_in, down_in = nuc_emb
        bias = self.reparam(
            'bias',
            jax.nn.initializers.normal(0.1, dtype=jnp.float32),
            (n_nuc, self.out_dim),
            param_type=ParamTypes.NUCLEI,
        )[0]
        return tuple(
            residual(a, self.activation((same(a) + diff(b)) / jnp.sqrt(2) + bias))
            for a, b in ((up_in, down_in), (down_in, up_in))
        )  # type: ignore


class Diffusion(ReparamModule):
    out_dim: int
    activation: Activation

    @nn.compact
    def __call__(
        self,
        systems: Systems,
        elec_emb: ElecEmbedding,
        nuc_emb: NucEmbeddings,
        elec_nuc_edge: ElecNucEdge,
        e_normalizer: Array,
    ) -> ElecEmbedding:
        out_emb = nn.Dense(self.out_dim)(elec_emb)
        elec_idx, nuc_idx, _ = systems.elec_nuc_idx
        edge_spin_mask = systems.spin_mask[elec_idx]

        up_inp, down_inp = nuc_emb
        inp = jnp.where(edge_spin_mask[:, None], up_inp[nuc_idx], down_inp[nuc_idx])

        weights = nn.Dense(self.out_dim, use_bias=False)(elec_nuc_edge)
        to_elec = inp * weights
        out_emb *= (
            segment_sum(to_elec, systems.elec_nuc_idx[0], systems.n_elec)
            / e_normalizer[..., None]
        )
        out_emb = self.activation(out_emb)
        out_emb = self.activation(nn.Dense(self.out_dim)(out_emb))
        return residual(elec_emb, out_emb)


class Moon(nn.Module, EmbeddingP):
    dim: int
    n_layer: int

    embedding_dim: int
    edge_embedding: int
    edge_hidden_dim: int
    edge_rbf: int

    activation: Activation

    @nn.compact
    def __call__(self, systems: Systems):
        elec_emb, nuc_emb, elec_nuc_edge, e_normalizer = MoonEmbedding(
            self.embedding_dim,
            self.activation,
            self.edge_embedding,
            self.edge_hidden_dim,
            self.edge_rbf,
        )(systems)

        for _ in range(self.n_layer):
            nuc_emb = Update(self.dim, self.activation)(nuc_emb)
        elec_emb = Diffusion(self.dim, self.activation)(
            systems, elec_emb, nuc_emb, elec_nuc_edge, e_normalizer
        )
        return elec_emb
