from typing import Sequence

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.struct import PyTreeNode
from jaxtyping import Array, Float

from neural_pfaffian.hf import HFOrbitals
from neural_pfaffian.nn.envelope import Envelope
from neural_pfaffian.nn.module import ReparamModule
from neural_pfaffian.nn.wave_function import AntisymmetrizerP
from neural_pfaffian.systems import Systems, SystemsWithHF

from .utils import hf_to_full


class FixedOrbitals(ReparamModule):
    num_orbitals: int
    determinants: int
    envelope: Envelope

    @nn.compact
    def __call__(self, systems: Systems, elec_embeddings: Float[Array, 'electrons dim']):
        inp_dim = elec_embeddings.shape[-1]
        W = self.param(
            'kernel',
            jax.nn.initializers.normal(1 / jnp.sqrt(inp_dim), dtype=jnp.float32),
            (elec_embeddings.shape[-1], self.num_orbitals * self.determinants),
        )
        # Set envelope output correctly
        env = self.envelope.copy(
            out_dim=self.num_orbitals * self.determinants, out_per_nuc=False
        )(systems)
        assert len(env) == 1
        env = env[0][systems.inverse_unique_indices]  # original order
        env = env.reshape(systems.n_elec, self.num_orbitals * self.determinants)
        orbitals = (elec_embeddings @ W) * env
        return einops.rearrange(orbitals, 'e (o d) -> e o d', o=self.num_orbitals)


class SlaterOrbitals(PyTreeNode):
    orbitals: Float[Array, 'n_mols n_det n_elec n_elec']


class Slater(ReparamModule, AntisymmetrizerP[SlaterOrbitals, None]):
    determinants: int
    envelope: Envelope

    @nn.compact
    def __call__(self, systems: Systems, elec_embeddings: Float[Array, 'electrons dim']):
        assert systems.spins_are_identical, (
            'Slater requires identical spins for all molecules'
        )
        n_up, n_down = systems.spins[0]
        n_orb, n_mols = max(n_up, n_down), systems.n_mols
        # Set envelopes output correctly
        diag = FixedOrbitals(
            n_orb, self.determinants, self.envelope.copy(pi_init=1.0, keep_distr=False)
        )(systems, elec_embeddings)
        off = FixedOrbitals(
            n_orb, self.determinants, self.envelope.copy(pi_init=0.1, keep_distr=True)
        )(systems, elec_embeddings)

        diag = einops.rearrange(
            diag,
            '(mol elec) orb det -> mol det elec orb',
            mol=n_mols,
        )
        off = einops.rearrange(
            off,
            '(mol elec) orb det -> mol det elec orb',
            mol=n_mols,
        )

        uu, dd = diag[..., :n_up, :n_up], diag[..., n_up:, :n_down]
        ud, du = off[..., :n_up, :n_down], off[..., n_up:, :n_up]
        return SlaterOrbitals(
            jnp.concatenate(
                [
                    jnp.concatenate([uu, ud], axis=-1),
                    jnp.concatenate([du, dd], axis=-1),
                ],
                axis=-2,
            )
        )

    def to_slog_psi(self, systems: Systems, orbitals: SlaterOrbitals):
        @jax.vmap  # vmap mols
        def _to_slog_psi(orbitals: SlaterOrbitals):
            sign, logpsi = jnp.linalg.slogdet(orbitals.orbitals)
            logpsi, sign = jax.nn.logsumexp(logpsi, b=sign, return_sign=True)
            return sign, logpsi

        return _to_slog_psi(orbitals)

    def match_hf_orbitals(
        self,
        systems: Systems,
        hf_orbitals: Sequence[HFOrbitals],  # list of molecules
        orbitals: SlaterOrbitals,  # grouped by molecules
        state: Sequence[None],  # list of molecules
    ):
        hf_up, hf_down = jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *hf_orbitals)
        hf_full = hf_to_full(hf_up, hf_down)[..., None, :, :]
        return ((orbitals.orbitals - hf_full) ** 2).mean(), tuple(state)

    def init_systems(self, key: Array, systems: SystemsWithHF):
        return systems.replace(cache=tuple([None] * systems.n_mols))
