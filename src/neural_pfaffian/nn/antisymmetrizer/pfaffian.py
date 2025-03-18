from typing import Sequence

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from flax.struct import PyTreeNode
from jaxtyping import Array, Float

from neural_pfaffian.hf import HFOrbitals
from neural_pfaffian.linalg import (
    cayley_transform,
    skewsymmetric_quadratic,
    slog_pfaffian,
    slog_pfaffian_skewsymmetric_quadratic,
    to_skewsymmetric_orthogonal,
)
from neural_pfaffian.nn.envelope import Envelope
from neural_pfaffian.nn.module import ParamTypes, ReparamModule
from neural_pfaffian.nn.utils import block
from neural_pfaffian.nn.wave_function import AntisymmetrizerP
from neural_pfaffian.systems import Systems, SystemsWithHF, chunk_electron
from neural_pfaffian.utils import EMA, itemgetter
from neural_pfaffian.utils.jax_utils import vmap
from neural_pfaffian.utils.optim import optimize

from .utils import hf_to_full


def max_orbitals(orb_per_charge: dict[str, int]):
    return max(orb_per_charge.values())


def orbital_mask(orb_per_charge: dict[str, int], charges: Sequence[int]):
    max_orb = max_orbitals(orb_per_charge)
    return np.concatenate(
        [
            [True] * orb + [False] * (max_orb - orb)
            for orb in map(orb_per_charge.__getitem__, map(str, charges))
        ]
    )


class PerNucOrbitals(ReparamModule):
    determinants: int
    orb_per_charge: dict[str, int]
    envelope: Envelope

    @nn.compact
    def __call__(self, systems: Systems, elec_embeddings: Float[Array, 'electrons dim']):
        inp_dim = elec_embeddings.shape[-1]
        max_orb = max_orbitals(self.orb_per_charge)
        W, W_meta = self.reparam(
            'projection',
            jax.nn.initializers.normal(1 / jnp.sqrt(inp_dim), dtype=jnp.float32),
            (
                systems.n_nuc,
                max_orb,
                elec_embeddings.shape[-1],
                self.determinants,
            ),
            param_type=ParamTypes.NUCLEI,
            chunk_axis=-1,
            keep_distr=True,
        )
        # Set envelopes output correctly
        env = self.envelope.copy(
            out_dim=self.determinants * max_orb,
            out_per_nuc=True,
        )(systems)

        result: list[Array] = []
        for emb, env, W, (spins, charges) in zip(
            systems.group(elec_embeddings, chunk_electron),
            env,
            systems.group(W, W_meta.param_type.value.chunk_fn),
            systems.unique_spins_and_charges,
        ):
            n_nuc = len(charges)
            n_orb = max_orb * n_nuc
            norm = (max(spins) / n_orb) ** 0.5
            orb_mask = orbital_mask(self.orb_per_charge, charges)

            @jax.vmap  # vmap over different molecules
            def _orbitals(emb: Array, env: Array, W: Array):
                env = einops.rearrange(env, 'elec (orb det) -> elec orb det', orb=n_orb)
                env = env[:, orb_mask, :]
                W = einops.rearrange(W, 'nuc opa dim det -> (nuc opa) dim det')
                W = W[orb_mask, :, :]
                return norm * einops.einsum(
                    emb,
                    W,
                    env,
                    'elec dim, orb dim det, elec orb det -> elec orb det',
                )

            result.append(_orbitals(emb, env, W))
        return result


class PfaffianPretrainingState(PyTreeNode):
    orbitals: Float[Array, 'n_orb n_orb']
    pfaffian: Float[Array, 'n_el n_el']


class PfaffianOrbitals(PyTreeNode):
    orbitals: Float[Array, 'mols det elec orbitals']
    antisymmetrizer: Float[Array, 'mols det orbitals orbitals']
    orb_A_orb_product: Float[Array, 'mols det elec elec']


class Pfaffian(
    ReparamModule,
    AntisymmetrizerP[list[PfaffianOrbitals], EMA[PfaffianPretrainingState]],
):
    determinants: int
    orb_per_charge: dict[str, int]
    envelope: Envelope

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
            (systems.n_nn, max_orb, max_orb, n_det),
            param_type=ParamTypes.NUCLEI_NUCLEI,
            bias=False,
        )

        same_orbs = PerNucOrbitals(
            n_det, self.orb_per_charge, self.envelope.copy(pi_init=1.0, keep_distr=False)
        )(systems, elec_embeddings)
        # init diff orbs with 0
        diff_orbs = PerNucOrbitals(
            n_det, self.orb_per_charge, self.envelope.copy(pi_init=1e-3, keep_distr=True)
        )(systems, elec_embeddings)
        # If n_elec is odd, we need an extra orbital
        fill_orbs = PerNucOrbitals(
            n_det,
            jtu.tree_map(lambda _: 1, self.orb_per_charge),
            self.envelope.copy(pi_init=1.0, keep_distr=False),
        )(systems, elec_embeddings)

        result: list[PfaffianOrbitals] = []
        for diag, offdiag, fill, A, (spins, charges) in zip(
            same_orbs,
            diff_orbs,
            fill_orbs,
            systems.group(A, A_meta.param_type.value.chunk_fn),
            systems.unique_spins_and_charges,
        ):
            n_elec, n_up, n_nuc = sum(spins), spins[0], len(charges)
            orb_mask = orbital_mask(self.orb_per_charge, charges)

            @vmap  # vmap over different molecules
            @vmap(in_axes=-1, out_axes=0)  # vmap over different determinants
            def _orbitals(diag: Array, offdiag: Array, A: Array, fill: Array):
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

                # Product
                orb_A_orb_product = skewsymmetric_quadratic(orbitals, A)
                # Pad additional orbital if n_elec is odd
                if n_elec % 2 == 1:
                    fill = einops.einsum(fill, 'elec orb -> elec')
                    orb_A_orb_product = block(
                        orb_A_orb_product, fill, -fill, jnp.zeros((), dtype=fill.dtype)
                    )
                return PfaffianOrbitals(orbitals, A, orb_A_orb_product)

            result.append(_orbitals(diag, offdiag, A, fill))
        return result

    def to_slog_psi(self, systems: Systems, orbitals: list[PfaffianOrbitals]):
        signs, logpsis = [], []
        for orb in orbitals:
            if orb.orbitals.shape[-2] == orb.orb_A_orb_product.shape[-2]:
                # We can use the fused version
                sign, logpsi = slog_pfaffian_skewsymmetric_quadratic(
                    orb.orbitals, orb.antisymmetrizer
                )
            else:
                # We need to act on the padded version
                sign, logpsi = slog_pfaffian(orb.orb_A_orb_product)
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
        orbitals: list[PfaffianOrbitals],  # grouped by molecules
        state: Sequence[EMA[PfaffianPretrainingState]],  # list of molecules
    ):
        @jax.vmap
        def hf_match(
            hf_up: jax.Array,
            hf_down: jax.Array,
            pf_orbs: PfaffianOrbitals,
            state: EMA[PfaffianPretrainingState],
        ):
            dtype = hf_up.dtype
            n_up, n_down = hf_up.shape[-2], hf_down.shape[-2]
            n_el = n_up + n_down
            n_orbs = pf_orbs.orbitals.shape[-1] // 2

            # Add dimension for the number of determinants
            if hf_up.ndim == pf_orbs.orbitals.ndim - 1:
                hf_up = hf_up[..., None, :, :]
                hf_down = hf_down[..., None, :, :]

            # Prepare HF orbitals
            hf_orb = hf_to_full(hf_up, hf_down, n_orbs)
            # Prepare HF orbitals for the Pfaffian
            # If the number of electrons is odd, we need to add a dummy electron
            # W.l.o.g., we assume n_up >= n_down
            hf_orb_pf = hf_to_full(hf_up, hf_down)
            if (n_up + n_down) % 2 == 1:
                eye = jnp.broadcast_to(
                    jnp.eye(n_el + 1, dtype=dtype),
                    (*hf_up.shape[:-2], n_el + 1, n_el + 1),
                )
                hf_orb_pf = eye.at[..., :n_el, :n_el].set(hf_orb_pf)

            def loss(state: PfaffianPretrainingState, final: bool = False):
                # Prepare HF targets
                hf_T = cayley_transform(state.orbitals)
                hf_A = to_skewsymmetric_orthogonal(state.pfaffian)
                hf_orb_targets = hf_orb @ hf_T
                hf_pf_targets = skewsymmetric_quadratic(hf_orb_pf, hf_A)
                # Prepare NN outputs
                orb, pf = pf_orbs.orbitals, pf_orbs.orb_A_orb_product
                orb_weight, pf_weight = self.hf_match_orbitals, self.hf_match_pfaffian
                if not final:
                    orb, pf = jax.lax.stop_gradient((orb, pf))
                    orb_weight = pf_weight = 1

                # Calculate loss
                loss_val = jnp.zeros((), dtype=pf.dtype)
                loss_val += ((hf_orb_targets - orb) ** 2).mean() * orb_weight
                loss_val += ((hf_pf_targets - pf) ** 2).mean() * pf_weight
                return loss_val

            optimizer = optax.contrib.prodigy(self.hf_match_lr)
            x_new, _ = optimize(loss, state.value(), optimizer, self.hf_match_steps)
            state = state.update(x_new, self.hf_match_ema)
            loss_val = loss(state.value(), final=True)
            return loss_val, state

        def stack(*x):
            return jnp.stack(x)

        out_state: Sequence[EMA[PfaffianPretrainingState]] = []
        loss = jnp.zeros((), dtype=jnp.float32)
        for idx, pfaff_orbs in zip(systems.unique_indices, orbitals):
            getter = itemgetter(*idx)
            hf_orbs, state_i = getter(hf_orbitals), getter(state)
            # Stack the molecules in the first dimension
            hf_up, hf_down = jtu.tree_map(stack, *hf_orbs)
            state_i = jtu.tree_map(stack, *state_i)
            # for pfaff_orbs, we expect to the see the molecules in the -4 dim. Thus, we should move it to the front
            pfaff_orbs = jtu.tree_map(lambda x: jnp.moveaxis(x, -4, 0), pfaff_orbs)

            # Matching
            loss_i, state_i = hf_match(hf_up, hf_down, pfaff_orbs, state_i)
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
