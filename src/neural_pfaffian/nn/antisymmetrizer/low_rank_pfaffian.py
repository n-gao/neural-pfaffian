import functools
from typing import Sequence

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax.struct import PyTreeNode
from jaxtyping import Array, Float

from neural_pfaffian.hf import HFOrbitals
from neural_pfaffian.linalg import (
    antisymmetric_block_diagonal,
    skewsymmetric_quadratic,
    slog_pfaffian_with_updates,
)
from neural_pfaffian.nn.antisymmetrizer.pfaffian import (
    PerNucOrbitals,
    PfaffianOrbitals,
    PfaffianPretrainingState,
    _pfaffian_pretraining_loss,
    max_orbitals,
    orbital_mask,
)
from neural_pfaffian.nn.envelope import Envelope
from neural_pfaffian.nn.module import ParamTypes, ReparamModule
from neural_pfaffian.nn.wave_function import AntisymmetrizerP
from neural_pfaffian.systems import Systems, SystemsWithHF
from neural_pfaffian.utils import EMA, itemgetter
from neural_pfaffian.utils.jax_utils import vmap
from neural_pfaffian.utils.tree_utils import tree_stack


class LowRankPfaffianOrbitals(PyTreeNode):
    orbitals: Float[Array, 'mols elec orbitals']
    antisymmetrizer: Float[Array, 'mols orbitals orbitals']
    antisymmetrizer_updates: Float[Array, 'mols det orbitals rank']
    orb_A_orb_product: Float[Array, 'mols elec elec']
    updates: Float[Array, 'mols det elec rank']

    def to_pfaffian_orbitals(self):
        A = jnp.expand_dims(self.antisymmetrizer, axis=-3)
        updates = self.antisymmetrizer_updates
        rank = updates.shape[-1]
        J = antisymmetric_block_diagonal(rank // 2, updates.dtype)
        As = A + skewsymmetric_quadratic(updates, J)
        orbitals = jnp.expand_dims(self.orbitals, axis=-3)
        orb_A_orb_product = jnp.expand_dims(self.orb_A_orb_product, axis=-3)
        orb_A_orb_product += skewsymmetric_quadratic(self.updates, J)
        return PfaffianOrbitals(
            orbitals=orbitals,
            antisymmetrizer=As,
            orb_A_orb_product=orb_A_orb_product,
        )


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
            chunk_axis=-1,
        )

        same_orbs = PerNucOrbitals(
            1, self.orb_per_charge, self.envelope.copy(pi_init=1.0), 'orb'
        )(systems, elec_embeddings)
        diff_orbs = PerNucOrbitals(
            1, self.orb_per_charge, self.envelope.copy(pi_init=1e-3), 'orb'
        )(systems, elec_embeddings)
        # If n_elec is odd, we need an extra orbital
        fill_vec, fill_vec_meta = self.reparam(
            'fill_coeffs',
            jax.nn.initializers.normal(1, dtype=jnp.float32),
            (systems.n_nuc, 2 * max_orb),
            param_type=ParamTypes.NUCLEI,
        )

        result: list[LowRankPfaffianOrbitals] = []
        for diag, offdiag, fill, A, A_updates, (spins, charges) in zip(
            same_orbs,
            diff_orbs,
            systems.group(fill_vec, fill_vec_meta.param_type.value.chunk_fn),
            systems.group(A, A_meta.param_type.value.chunk_fn),
            systems.group(A_updates, A_updates_meta.param_type.value.chunk_fn),
            systems.unique_spins_and_charges,
        ):
            n_mol, n_elec, n_up, n_nuc = diag.shape[0], sum(spins), spins[0], len(charges)
            orb_mask = orbital_mask(self.orb_per_charge, charges)
            full_mask = np.repeat(orb_mask, 2)
            diag = diag.reshape(n_mol, n_elec, -1)
            offdiag = offdiag.reshape(n_mol, n_elec, -1)

            @vmap  # vmap over different molecules
            def _orbitals(
                diag: Array, offdiag: Array, A: Array, A_updates: Array, fill: Array
            ):
                # construct full orbitals
                uu, dd, ud, du = diag[:n_up], diag[n_up:], offdiag[:n_up], offdiag[n_up:]
                orbitals = jnp.block([[uu, ud], [du, dd]])  # (n_elec, 2*n_orbs)
                # Pad additional orbital if n_elec is odd
                if n_elec % 2 == 1:
                    fill = fill.reshape(1, -1)[:, full_mask]
                    orbitals = jnp.concatenate([orbitals, fill], axis=0)

                # A: (2*n_orbs, 2*n_orbs)
                A = einops.rearrange(A, '(n1 n2) o1 o2 -> (n1 o1) (n2 o2)', n1=n_nuc)
                A = A[orb_mask, :][:, orb_mask]  # remove unused orbitals
                A_diag, A_offdiag = (A - A.mT) / 2, (A + A.mT) / 2
                A = jnp.block([[A_diag, A_offdiag], [-A_offdiag, A_diag]])

                # Product
                orb_A_orb_product = skewsymmetric_quadratic(
                    orbitals.astype(jnp.float64), A.astype(jnp.float64)
                )

                # Updates
                A_updates = einops.rearrange(
                    A_updates, 'nuc orb two rank det -> det (two nuc orb) rank'
                )[:, full_mask]
                updates = jnp.einsum(
                    'no,dor->dnr',
                    orbitals.astype(jnp.float64),
                    A_updates.astype(jnp.float64),
                )
                return LowRankPfaffianOrbitals(
                    orbitals, A, A_updates, orb_A_orb_product, updates
                )

            result.append(_orbitals(diag, offdiag, A, A_updates, fill))
        return result

    def to_slog_psi(self, systems: Systems, orbitals: list[LowRankPfaffianOrbitals]):
        dtype = systems.electrons.dtype
        signs, logpsis = [], []
        for orb in orbitals:
            # Add dimension for the number of determinants
            sign, logpsi = slog_pfaffian_with_updates(
                orb.orb_A_orb_product[:, None], orb.updates.astype(jnp.float64)
            )
            logpsi, sign = jax.nn.logsumexp(logpsi, axis=1, b=sign, return_sign=True)
            signs.append(sign)
            logpsis.append(logpsi)
        order = systems.inverse_unique_indices
        sign = jnp.concatenate(signs)[order]
        log_psi = jnp.concatenate(logpsis)[order]
        return sign.astype(jnp.int32), log_psi.astype(dtype)

    def match_hf_orbitals(
        self,
        systems: Systems,
        hf_orbitals: Sequence[HFOrbitals],  # list of molecules
        orbitals: list[LowRankPfaffianOrbitals],  # grouped by molecules
        state: Sequence[EMA[PfaffianPretrainingState]],  # list of molecules
    ):
        loss_fn = functools.partial(
            _pfaffian_pretraining_loss,
            orb_weight=self.hf_match_orbitals,
            pf_weight=self.hf_match_pfaffian,
            learning_rate=self.hf_match_lr,
            steps=self.hf_match_steps,
            ema=self.hf_match_ema,
        )
        out_state: Sequence[EMA[PfaffianPretrainingState]] = []
        loss = jnp.zeros((), dtype=jnp.float32)
        for idx, pfaff_orbs in zip(systems.unique_indices, orbitals):
            getter = itemgetter(*idx)
            hf_orbs, state_i = getter(hf_orbitals), getter(state)
            # Stack the molecules in the first dimension
            (hf_up, hf_down), state_i = tree_stack(*hf_orbs), tree_stack(*state_i)

            pfaff_orbs = pfaff_orbs.to_pfaffian_orbitals()
            # for orbitals, we expect to the see the molecules in the -4 dim. Thus, we should move it to the front
            pfaff_orbs = jtu.tree_map(lambda x: jnp.moveaxis(x, -4, 0), pfaff_orbs)

            # Matching
            loss_i, state_i = jax.vmap(loss_fn)(hf_up, hf_down, pfaff_orbs, state_i)
            loss += loss_i.sum()

            # out_state is now sorted by the unique indices and not by the original batch!
            for i in range(len(idx)):
                out_state.append(jtu.tree_map(lambda x: x[i], state_i))
        # invert the order of the unique indices
        out_state = itemgetter(*systems.inverse_unique_indices)(out_state)
        return loss / systems.n_mols, out_state

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
