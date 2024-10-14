from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Float

from neural_pfaffian.nn.module import ParamTypes, ReparamModule
from neural_pfaffian.nn.ops import segment_mean, segment_sum
from neural_pfaffian.nn.utils import Activation, log1p_rescale, residual
from neural_pfaffian.nn.wave_function import EmbeddingP
from neural_pfaffian.systems import Systems

SingleStream = Float[Array, 'n_elec single_dim']
PairStream = tuple[Float[Array, 'n_pairs pair_dim'], Float[Array, 'n_pairs pair_dim']]


def aggregate_features(
    systems: Systems,
    h_one: SingleStream,
    h_two: PairStream,
) -> Tuple[SingleStream, jax.Array]:
    spins = np.array(systems.spins)
    segments = np.repeat(jnp.arange(spins.size), spins.reshape(-1))

    # We use segment_sum since segment_mean requires two segment_sums
    g_inp = segment_sum(h_one, segments, systems.n_mols * 2, True).reshape(
        systems.n_mols, 2, -1
    ) / np.maximum(spins[..., None], 1)
    g_inp = jnp.stack(
        [
            jnp.concatenate([g_inp[:, 0], g_inp[:, 1]], axis=-1),
            jnp.concatenate([g_inp[:, 1], g_inp[:, 0]], axis=-1),
        ],
        axis=1,
    )
    g_inp = g_inp.reshape(2 * systems.n_mols, -1)

    pair = []
    for h, diag in zip(h_two, (True, False)):
        pair.append(segment_mean(h, systems.elec_pair_mask(diag), systems.n_elec, True))
    return jnp.concatenate([h_one, *pair], axis=-1), g_inp


class FermiLayer(nn.Module):
    single_out: int
    pair_out: int
    activation: Activation

    @nn.compact
    def __call__(
        self,
        systems: Systems,
        h_one: SingleStream,
        h_two: PairStream,
    ) -> tuple[SingleStream, PairStream]:
        spins = np.array(systems.spins)

        # Single update
        one_in, global_in = aggregate_features(systems, h_one, h_two)
        h_one_new = nn.Dense(self.single_out)(one_in)
        global_new = nn.Dense(self.single_out, use_bias=False)(global_in)
        global_new = jnp.repeat(global_new, np.reshape(spins, -1), axis=0)
        h_one_new += global_new
        h_one_new = self.activation(h_one_new / jnp.sqrt(2))
        h_one = residual(h_one, h_one_new)

        # Pairwise update
        if self.pair_out > 0:
            # Since we rearranged our pairwise terms such that the diagonals are first,
            # we only need to split the array into the first and second half as these correspond to the
            # diagonal and off diagonal terms.
            h_two_new = tuple(nn.Dense(self.pair_out)(h) for h in h_two)
            if h_two_new[0].shape != h_two[0].shape:
                h_two = jtu.tree_map(jnp.tanh, h_two_new)
            else:
                h_two_new = jtu.tree_map(self.activation, h_two_new)
                h_two = jtu.tree_map(residual, h_two, h_two_new)
        return h_one, h_two


class FermiNetFeatures(ReparamModule):
    out_dim: int

    @nn.compact
    def __call__(self, systems: Systems) -> tuple[SingleStream, PairStream]:
        h_two = log1p_rescale(systems.elec_elec_dists)
        n = systems.n_elec_pair_same
        h_two = (h_two[:n], h_two[n:])

        nuc_idx = systems.elec_nuc_idx[1]
        h_one = log1p_rescale(systems.elec_nuc_dists)
        kernel = self.reparam(
            'kernel',
            jnn.initializers.normal(1 / np.sqrt(h_one.shape[-1])),
            (systems.n_nuc, h_one.shape[-1], self.out_dim),
            param_type=ParamTypes.NUCLEI,
        )[0][nuc_idx]
        bias = self.reparam(
            'bias',
            jnn.initializers.zeros,
            (systems.n_nuc, self.out_dim),
            param_type=ParamTypes.NUCLEI,
        )[0][nuc_idx]
        h_one = jnp.einsum('...d,...dk->...k', h_one, kernel) + bias
        h_one = jnp.tanh(h_one)
        h_one = segment_sum(h_one, systems.elec_nuc_idx[0], systems.n_elec, True)
        return h_one, h_two


class FermiNet(nn.Module, EmbeddingP):
    embedding_dim: int
    hidden_dims: Sequence[tuple[int, int]]
    activation: Activation

    @nn.compact
    def __call__(self, systems: Systems):
        h_one, h_two = FermiNetFeatures(self.embedding_dim)(systems)
        for single_dim, pair_dim in self.hidden_dims:
            h_one, h_two = FermiLayer(single_dim, pair_dim, self.activation)(
                systems, h_one, h_two
            )
        return h_one
