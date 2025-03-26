from typing import Sequence

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.struct import PyTreeNode
from jaxtyping import Array, Float

from neural_pfaffian.hf import HFOrbitals
from neural_pfaffian.linalg import skewsymmetric_quadratic, slog_pfaffian_with_updates
from neural_pfaffian.nn.antisymmetrizer.pfaffian import (
    PerNucOrbitals,
    PfaffianPretrainingState,
    max_orbitals,
    orbital_mask,
)
from neural_pfaffian.nn.envelope import Envelope
from neural_pfaffian.nn.module import ParamTypes, ReparamModule
from neural_pfaffian.nn.utils import block
from neural_pfaffian.nn.wave_function import AntisymmetrizerP
from neural_pfaffian.systems import Systems, SystemsWithHF
from neural_pfaffian.utils import EMA
from neural_pfaffian.utils.jax_utils import vmap


class LowRankPfaffianOrbitals(PyTreeNode):
    orbitals: Float[Array, 'mols elec orbitals']
    antisymmetrizer: Float[Array, 'mols orbitals orbitals']
    antisymmetrizer_updates: Float[Array, 'mols orbitals rank det']
    orb_A_orb_product: Float[Array, 'mols elec elec']
    updates: Float[Array, 'mols det elec rank']


class LowRankPfaffian(
    ReparamModule,
    AntisymmetrizerP[list[LowRankPfaffianOrbitals], EMA[PfaffianPretrainingState]],
):
    determinants: int
    orb_per_charge: dict[str, int]
    envelope: Envelope
    rank: int

    hf_match_steps: int
    hf_match_lr: float
    hf_match_orbitals: float
    hf_match_pfaffian: float
    hf_match_ema: float

    @nn.compact
    def __call__(self, systems: Systems, elec_embeddings: Float[Array, 'electrons dim']):
        n_det = self.determinants
        max_orb = max_orbitals(self.orb_per_charge)
        A, A_meta = self.reparam(
            'antisymmetrizer',
            jax.nn.initializers.normal(1, dtype=jnp.float32),
            (systems.n_nn, max_orb, max_orb),
            param_type=ParamTypes.NUCLEI_NUCLEI,
            bias=False,
        )

        A_updates, A_updates_meta = self.reparam(
            'antisymmetrizer_updates',
            jax.nn.initializers.normal(1, dtype=jnp.float32),
            (systems.n_nuc, max_orb, 2, self.rank, self.determinants),
            param_type=ParamTypes.NUCLEI,
        )

        same_orbs = PerNucOrbitals(
            1, self.orb_per_charge, self.envelope.copy(pi_init=1.0, keep_distr=False)
        )(systems, elec_embeddings)
        # init diff orbs with 0
        diff_orbs = PerNucOrbitals(
            1, self.orb_per_charge, self.envelope.copy(pi_init=1e-3, keep_distr=True)
        )(systems, elec_embeddings)
        # If n_elec is odd, we need an extra orbital
        fill_orbs = PerNucOrbitals(
            1,
            jtu.tree_map(lambda _: n_det, self.orb_per_charge),
            self.envelope.copy(pi_init=1.0, keep_distr=False),
        )(systems, elec_embeddings)

        result: list[LowRankPfaffianOrbitals] = []
        for diag, offdiag, fill, A, A_updates, (spins, charges) in zip(
            same_orbs,
            diff_orbs,
            fill_orbs,
            systems.group(A, A_meta.param_type.value.chunk_fn),
            systems.group(A_updates, A_updates_meta.param_type.value.chunk_fn),
            systems.unique_spins_and_charges,
        ):
            n_elec, n_up, n_nuc = sum(spins), spins[0], len(charges)
            orb_mask = orbital_mask(self.orb_per_charge, charges)
            # squeeze out the determinants
            diag, offdiag, fill = diag.squeeze(-1), offdiag.squeeze(-1), fill.squeeze(-1)

            @vmap  # vmap over different molecules
            def _orbitals(
                diag: Array, offdiag: Array, A: Array, A_updates: Array, fill: Array
            ):
                # construct full orbitals
                uu, dd, ud, du = diag[:n_up], diag[n_up:], offdiag[:n_up], offdiag[n_up:]
                orbitals = jnp.concatenate(
                    [
                        jnp.concatenate([uu, ud], axis=1),
                        jnp.concatenate([du, dd], axis=1),
                    ],
                    axis=0,
                )  # (n_elec, 2*n_orbs)

                # A: (2*n_orbs, 2*n_orbs)
                A = einops.rearrange(A, '(n1 n2) o1 o2 -> (n1 o1) (n2 o2)', n1=n_nuc)
                A = A[orb_mask, :][:, orb_mask]  # remove unused orbitals
                A_diag, A_offdiag = (A - A.mT) / 2, (A + A.mT) / 2
                A = jnp.block([[A_diag, A_offdiag], [-A_offdiag, A_diag]])
                A_updates = einops.rearrange(
                    A_updates, 'nuc orb two rank det -> (two nuc orb) rank det'
                )
                updates = jnp.einsum('no,ord->dnr', orbitals, A_updates)

                # Product
                orb_A_orb_product = skewsymmetric_quadratic(orbitals, A)

                # Pad additional orbital if n_elec is odd
                if n_elec % 2 == 1:
                    fill = einops.einsum(fill, 'elec orb -> elec')
                    orb_A_orb_product = block(
                        orb_A_orb_product, fill, -fill, jnp.zeros((), dtype=fill.dtype)
                    )
                return LowRankPfaffianOrbitals(
                    orbitals, A, A_updates, orb_A_orb_product, updates
                )

            result.append(_orbitals(diag, offdiag, A, A_updates, fill))
        return result

    def to_slog_psi(self, systems: Systems, orbitals: list[LowRankPfaffianOrbitals]):
        signs, logpsis = [], []
        for orb in orbitals:
            # Add dimension for the number of determinants
            sign, logpsi = slog_pfaffian_with_updates(
                orb.orb_A_orb_product[:, None], orb.updates
            )
            logpsi, sign = jax.nn.logsumexp(logpsi, axis=1, b=sign, return_sign=True)
            signs.append(sign)
            logpsis.append(logpsi)
        order = systems.inverse_unique_indices
        sign = jnp.concatenate(signs)[order]
        log_psi = jnp.concatenate(logpsis)[order]
        return sign, log_psi

    def match_hf_orbitals(
        self,
        systems: Systems,
        hf_orbitals: Sequence[HFOrbitals],  # list of molecules
        orbitals: list[LowRankPfaffianOrbitals],  # grouped by molecules
        state: Sequence[EMA[PfaffianPretrainingState]],  # list of molecules
    ):
        raise NotImplementedError

    def init_systems(self, key: Array, systems: SystemsWithHF):
        states: list[EMA[PfaffianPretrainingState]] = []
        for sub_sys in systems.sub_configs:
            n_el = sub_sys.n_elec + (sub_sys.n_elec % 2)
            n_orbs = orbital_mask(self.orb_per_charge, tuple(sub_sys.flat_charges)).sum()
            state = EMA[PfaffianPretrainingState].init(
                PfaffianPretrainingState(
                    orbitals=jnp.zeros((2 * n_orbs, 2 * n_orbs), dtype=jnp.float32),
                    pfaffian=jnp.zeros((n_el, n_el), dtype=jnp.float32),
                )
            )
            states.append(state)
        return systems.replace(cache=tuple(states))
