from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax.struct import PyTreeNode
from jaxtyping import Array, Float

from neural_pfaffian.hf import HFOrbitals
from neural_pfaffian.nn.embedding.fire import (
    DataScale,
    FiREElecInit,
    FiREPairMessages,
)
from neural_pfaffian.nn.envelope import Envelope
from neural_pfaffian.nn.module import ReparamModule
from neural_pfaffian.nn.ops import segment_sum
from neural_pfaffian.nn.utils import Activation, ActivationOrName
from neural_pfaffian.nn.wave_function import AntisymmetrizerP
from neural_pfaffian.systems import Systems, SystemsWithHF

from .slater import Slater, hf_orbital_loss


def pooled_mean(data: Array, segments: npt.NDArray[np.int64], n_segments: int):
    """Segment mean with a static normalization.

    Args:
        data: Values to average, segmented along the leading axis.
        segments: Segment index per entry, known at trace time.
        n_segments: Total number of segments.

    Returns:
        The per-segment mean of shape (n_segments, *data.shape[1:]).
    """
    counts = np.maximum(np.bincount(segments, minlength=n_segments), 1)
    return segment_sum(data, segments, n_segments) / counts.reshape(
        -1, *(1,) * (data.ndim - 1)
    )


class SymmetricFactor(ReparamModule):
    """Permutation-invariant coefficients of the determinants.

    Electron-electron pair messages are pooled per molecule and spin channel,
    which keeps the coefficients invariant under same-spin permutations while
    letting them distinguish the two channels. With `post_layers = 0` nothing of
    size (n_elec, embedding_dim) is materialized, so the only dense-jacobian
    object in the forward laplacian is the pooled vector.
    """

    determinants: int
    embedding_dim: int
    filter_hidden_dim: int
    filter_dim: int
    n_envelopes: int
    hidden_dims: Sequence[int]
    activation: ActivationOrName
    post_layers: int = 0

    @nn.compact
    def __call__(self, systems: Systems) -> Float[Array, 'n_mols determinants']:
        activation = Activation(self.activation)
        filter_args = (
            self.embedding_dim,
            self.filter_hidden_dim,
            self.filter_dim,
            self.n_envelopes,
            self.activation,
        )
        h, msgs = FiREElecInit(*filter_args)(systems)
        pair_messages = FiREPairMessages(*filter_args)(systems, msgs)

        n_seg = 2 * systems.n_mols
        # one segment per molecule and spin channel
        elec_seg = 2 * systems.electron_molecule_mask + systems.spin_mask
        if self.post_layers > 0:
            for messages, idx in pair_messages:
                # aggregation to electrons; the jacobian is dense from here on
                h = h + segment_sum(messages, idx, systems.n_elec)
            h = DataScale()(h)
            for _ in range(self.post_layers):
                h = h + nn.Dense(self.embedding_dim)(
                    activation(nn.Dense(self.embedding_dim)(h))
                )
            pooled = pooled_mean(h, elec_seg, n_seg)
        else:
            pooled = jnp.concatenate(
                [
                    pooled_mean(h, elec_seg, n_seg),
                    sum(
                        pooled_mean(messages, elec_seg[idx], n_seg)
                        for messages, idx in pair_messages
                    ),
                ],
                axis=-1,
            )
        pooled = DataScale()(pooled.reshape(systems.n_mols, -1))

        for dim in self.hidden_dims:
            pooled = activation(nn.Dense(dim)(pooled))
        # zero initialization gives all determinants unit weight at initialization
        out = nn.Dense(self.determinants, kernel_init=jax.nn.initializers.zeros)(pooled)
        return 1 + out


class FermiSetsOrbitals(PyTreeNode):
    orbitals: Float[Array, 'n_mols n_det n_elec n_elec']
    coefficients: Float[Array, 'n_mols n_det']


class FermiSets(ReparamModule, AntisymmetrizerP[FermiSetsOrbitals, None]):
    """Fermi Sets ansatz (arXiv:2601.02508).

    psi = sum_k c_k(R) det[orb^k], where the orbitals are single-particle
    functions of `elec_embeddings` and the coefficients c_k are permutation
    invariant. If the embedding depends on a single electron per output, the
    jacobian of the orbital matrix is row-sparse and folx evaluates the
    laplacian of log|det| in O(n_elec^3).
    """

    determinants: int
    envelope: Envelope
    embedding_dim: int
    filter_hidden_dim: int
    filter_dim: int
    n_envelopes: int
    hidden_dims: Sequence[int]
    activation: ActivationOrName
    post_layers: int = 0

    @nn.compact
    def __call__(self, systems: Systems, elec_embeddings: Float[Array, 'electrons dim']):
        orbitals = Slater(self.determinants, self.envelope)(systems, elec_embeddings)
        coefficients = SymmetricFactor(
            self.determinants,
            self.embedding_dim,
            self.filter_hidden_dim,
            self.filter_dim,
            self.n_envelopes,
            self.hidden_dims,
            self.activation,
            self.post_layers,
        )(systems)
        return FermiSetsOrbitals(orbitals.orbitals, coefficients)

    def to_slog_psi(self, systems: Systems, orbitals: FermiSetsOrbitals):
        sign, logdet = jnp.linalg.slogdet(orbitals.orbitals)
        logpsi, sign = jax.nn.logsumexp(
            logdet, axis=-1, b=sign * orbitals.coefficients, return_sign=True
        )
        return sign, logpsi

    def match_hf_orbitals(
        self,
        systems: Systems,
        hf_orbitals: Sequence[HFOrbitals],
        orbitals: FermiSetsOrbitals,
        state: Sequence[None],
    ):
        return hf_orbital_loss(hf_orbitals, orbitals.orbitals), tuple(state)

    def init_systems(self, key: Array, systems: SystemsWithHF):
        return systems.replace(cache=tuple([None] * systems.n_mols))
