import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
import numpy.typing as npt

from neural_pfaffian.nn.edges import shifted_normal
from neural_pfaffian.nn.module import ParamTypes, ReparamModule
from neural_pfaffian.nn.ops import segment_sum
from neural_pfaffian.nn.utils import Activation, ActivationOrName, log1p_rescale
from neural_pfaffian.nn.wave_function import EmbeddingP
from neural_pfaffian.systems import Systems

ElecEmbedding = Float[Array, 'n_elec embedding_dim']
Edges = Float[Array, 'n_edges 3+1']


class DataScale(nn.Module):
    """Rescales activations by a fixed scalar calibrated to unit std at init."""

    @nn.compact
    def __call__(self, x: Array) -> Array:
        def init(key: jax.Array):
            scale = 1.0 / jnp.std(x)
            return jnp.where(jnp.isfinite(scale), scale, 1.0).astype(jnp.float32)

        return x * self.param('scale', init)


class FiREFilter(ReparamModule):
    """FiRE pairwise filter: directional MLP times radial Gaussian envelopes.

    Infinite-cutoff variant: the smooth cutoff window of the finite-range
    original is dropped; only the learned Gaussian envelopes decay with
    distance. Per-nucleus parameters are used when `center_idx` is given.
    """

    hidden_dim: int
    out_dim: int
    n_envelopes: int
    activation: ActivationOrName
    sigma_init: float = 10.0

    @nn.compact
    def __call__(
        self,
        systems: Systems,
        edges: Edges,
        center_idx: npt.NDArray[np.int64] | None = None,
    ) -> Float[Array, 'n_edges out_dim']:
        kernel = self.edge_reparam(
            'kernel',
            systems,
            jax.nn.initializers.normal(1 / 2, jnp.float32),
            (4, self.hidden_dim),
            None,
            center_idx,
            keep_distr=True,
        )
        bias = self.edge_reparam(
            'bias',
            systems,
            jax.nn.initializers.normal(1, jnp.float32),
            (self.hidden_dim,),
            None,
            center_idx,
        )
        scales = self.edge_reparam(
            'scales',
            systems,
            shifted_normal(self.sigma_init),
            (self.n_envelopes,),
            None,
            center_idx,
        )
        directional = Activation(self.activation)(
            jnp.einsum('...d,...dk->...k', edges, kernel) + bias
        )
        directional = nn.Dense(self.out_dim)(directional)
        envelopes = jnp.exp(-((edges[..., -1:] / nn.softplus(scales)) ** 2))
        envelopes = nn.Dense(self.out_dim, use_bias=False)(envelopes)
        return directional * envelopes


class FiREElecInit(ReparamModule):
    """Nucleus-electron message passing producing initial electron features.

    Every output depends on a single electron (and the fixed nuclei), so the
    jacobian w.r.t. electron positions stays 3-sparse through this stage.

    Besides the initial embedding, `n_messages` per-electron vectors are
    projected for the subsequent electron-electron pair messages.
    """

    embedding_dim: int
    filter_hidden_dim: int
    filter_dim: int
    n_envelopes: int
    activation: ActivationOrName
    n_messages: int = 4

    @nn.compact
    def __call__(
        self, systems: Systems
    ) -> tuple[ElecEmbedding, tuple[ElecEmbedding, ...]]:
        activation = Activation(self.activation)
        edges = systems.elec_nuc_dists
        elec_idx, nuc_idx, _ = systems.elec_nuc_idx

        beta = FiREFilter(
            self.filter_hidden_dim, self.filter_dim, self.n_envelopes, self.activation
        )(systems, edges, nuc_idx)
        gamma = nn.Dense(self.embedding_dim, use_bias=False)(beta)
        nuc_embedding = self.reparam(
            'nuc_embedding',
            jax.nn.initializers.normal(1, jnp.float32),
            (systems.n_nuc, self.embedding_dim),
            param_type=ParamTypes.NUCLEI,
        )[0][nuc_idx]
        edge_embedding = (
            nn.Dense(self.embedding_dim, use_bias=False)(log1p_rescale(edges))
            + nuc_embedding
        )

        h = segment_sum(gamma * edge_embedding, elec_idx, systems.n_elec)
        h = nn.LayerNorm()(h)
        # gated linear unit
        gates, values = jnp.split(
            nn.Dense(2 * self.embedding_dim, use_bias=False)(h), 2, axis=-1
        )
        h = values * jax.nn.silu(gates)
        h = activation(nn.Dense(self.embedding_dim)(h))
        # initial embedding and the pair-message vectors in one projection
        n_out = 1 + self.n_messages
        out = jnp.split(nn.Dense(n_out * self.embedding_dim)(h), n_out, axis=-1)
        return out[0], tuple(out[1:])


class FiREPairMessages(nn.Module):
    """Electron-electron pair messages, one filter per spin block.

    The messages are returned unaggregated: each of them depends on two
    electrons only, so the jacobian stays 6-sparse until a caller reduces them.
    """

    embedding_dim: int
    filter_hidden_dim: int
    filter_dim: int
    n_envelopes: int
    activation: ActivationOrName

    @nn.compact
    def __call__(
        self, systems: Systems, messages: tuple[ElecEmbedding, ...]
    ) -> list[tuple[Float[Array, 'n_pairs embedding_dim'], npt.NDArray[np.int64]]]:
        """Computes the pair messages of both spin blocks.

        Args:
            systems: The systems to compute the messages for.
            messages: The four per-electron message vectors of `FiREElecInit`,
                ordered center/neighbor times same/different spin.

        Returns:
            List of (messages, receiver index) per spin block.
        """
        activation = Activation(self.activation)
        msg_ct_same, msg_ct_diff, msg_nb_same, msg_nb_diff = messages
        r_ij = systems.elec_elec_dists
        i, j, _ = systems.elec_elec_idx
        n_same = systems.n_elec_pair_same
        spin_blocks = (
            (slice(None, n_same), msg_ct_same, msg_nb_same),
            (slice(n_same, None), msg_ct_diff, msg_nb_diff),
        )

        result = []
        for block, msg_ct, msg_nb in spin_blocks:
            edges = r_ij[block]
            beta = FiREFilter(
                self.filter_hidden_dim,
                self.filter_dim,
                self.n_envelopes,
                self.activation,
            )(systems, edges)
            gamma = nn.Dense(self.embedding_dim, use_bias=False)(beta)
            features = nn.Dense(self.embedding_dim)(log1p_rescale(edges))
            result.append(
                (
                    gamma * activation(features + msg_ct[i[block]] + msg_nb[j[block]]),
                    i[block],
                )
            )
        return result


class FiRE(nn.Module, EmbeddingP):
    """FiRE embedding (arXiv:2504.06087) with an infinite cutoff.

    Stage 1 aggregates nucleus-electron messages per electron (3-sparse
    jacobian), stage 2 builds electron-electron pair messages (6-sparse), and
    only the final aggregation over pairs densifies the jacobian. Neither the
    finite cutoff nor low-rank updates of the original are implemented.
    """

    embedding_dim: int
    filter_hidden_dim: int
    filter_dim: int
    n_envelopes: int
    activation: ActivationOrName

    @nn.compact
    def __call__(self, systems: Systems) -> ElecEmbedding:
        activation = Activation(self.activation)
        h, msgs = FiREElecInit(
            self.embedding_dim,
            self.filter_hidden_dim,
            self.filter_dim,
            self.n_envelopes,
            self.activation,
        )(systems)

        pair_messages = FiREPairMessages(
            self.embedding_dim,
            self.filter_hidden_dim,
            self.filter_dim,
            self.n_envelopes,
            self.activation,
        )(systems, msgs)
        for messages, idx in pair_messages:
            # aggregation to electrons; the jacobian is dense from here on
            h = h + segment_sum(messages, idx, systems.n_elec)

        h = DataScale()(h)
        h = nn.Dense(self.embedding_dim)(h)
        h = activation(h)
        h = nn.Dense(self.embedding_dim)(h)
        h = DataScale()(h)
        return h


class FiRELocal(nn.Module, EmbeddingP):
    """FiRE embedding restricted to the nucleus-electron stage.

    Without electron-electron message passing every feature depends on a single
    electron, so the jacobian stays 3-sparse and the per-electron layers cost
    O(n_elec) rather than O(n_elec^2) in the forward laplacian.
    """

    embedding_dim: int
    filter_hidden_dim: int
    filter_dim: int
    n_envelopes: int
    activation: ActivationOrName
    n_layer: int = 1

    @nn.compact
    def __call__(self, systems: Systems) -> ElecEmbedding:
        activation = Activation(self.activation)
        h, _ = FiREElecInit(
            self.embedding_dim,
            self.filter_hidden_dim,
            self.filter_dim,
            self.n_envelopes,
            self.activation,
            n_messages=0,
        )(systems)
        h = DataScale()(h)
        for _ in range(self.n_layer):
            h = h + nn.Dense(self.embedding_dim)(
                activation(nn.Dense(self.embedding_dim)(h))
            )
        return DataScale()(h)
