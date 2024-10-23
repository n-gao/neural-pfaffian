from typing import Sequence

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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
from neural_pfaffian.nn.envelopes import Envelope
from neural_pfaffian.nn.module import ParamTypes, ReparamModule
from neural_pfaffian.nn.utils import block
from neural_pfaffian.nn.wave_function import OrbitalsP
from neural_pfaffian.systems import Systems, SystemsWithHF, chunk_electron
from neural_pfaffian.utils import EMAState, ema_make, ema_update, ema_value, itemgetter
from neural_pfaffian.utils.jax_utils import pmean_if_pmap


def _hf_to_full(hf_up: Array, hf_down: Array):
    n_up, n_down = hf_up.shape[-2], hf_down.shape[-2]
    return jnp.concatenate(
        [
            jnp.concatenate([hf_up, jnp.zeros((*hf_up.shape[:-1], n_down))], axis=-1),
            jnp.concatenate([jnp.zeros((*hf_down.shape[:-1], n_up)), hf_down], axis=-1),
        ],
        axis=-2,
    )


class PerNucOrbitals(ReparamModule):
    determinants: int
    orb_per_nuc: int
    envelope: Envelope

    @nn.compact
    def __call__(self, systems: Systems, elec_embeddings: Float[Array, 'electrons dim']):
        inp_dim = elec_embeddings.shape[-1]
        W, W_meta = self.reparam(
            'projection',
            jax.nn.initializers.normal(1 / jnp.sqrt(inp_dim), dtype=jnp.float32),
            (
                systems.n_nuc,
                self.orb_per_nuc,
                elec_embeddings.shape[-1],
                self.determinants,
            ),
            param_type=ParamTypes.NUCLEI,
            chunk_axis=-1,
        )
        # Set envelopes output correctly
        env = self.envelope(systems)

        result: list[Array] = []
        for emb, env, W, (spins, charges) in zip(
            systems.group(elec_embeddings, chunk_electron),
            env,
            systems.group(W, W_meta.param_type.value.chunk_fn),
            systems.unique_spins_and_charges,
        ):
            n_nuc = len(charges)
            n_orb = self.orb_per_nuc * n_nuc
            norm = (max(spins) / n_orb) ** 0.5

            @jax.vmap  # vmap over different molecules
            def _orbitals(emb, env, W):
                env = einops.rearrange(env, 'e (o d) -> e o d', o=n_orb)
                W = W.reshape(n_orb, inp_dim, self.determinants)
                return jnp.einsum('ei,oid,eod->eod', emb, W, env) * norm

            result.append(_orbitals(emb, env, W))
        return result


class PfaffianPretrainingState(PyTreeNode):
    orbitals: Float[Array, '2 n_orb n_orb']
    pfaffian: Float[Array, 'n_el n_el']


class PfaffianOrbitals(PyTreeNode):
    orbitals: Float[Array, 'mols det elec orbitals']
    antisymmetrizer: Float[Array, 'mols det orbitals orbitals']
    orb_A_orb_product: Float[Array, 'mols det elec elec']


class Pfaffian(
    ReparamModule,
    OrbitalsP[PfaffianOrbitals, EMAState[PfaffianPretrainingState]],
):
    determinants: int
    orb_per_nuc: int
    envelope: Envelope

    hf_match_steps: int
    hf_match_lr: float
    hf_match_orbitals: float
    hf_match_pfaffian: float

    @nn.compact
    def __call__(self, systems: Systems, elec_embeddings: Float[Array, 'electrons dim']):
        A, A_meta = self.reparam(
            'antisymmetrizer',
            jax.nn.initializers.normal(1, dtype=jnp.float32),
            (systems.n_nn, self.orb_per_nuc, self.orb_per_nuc, self.determinants),
            param_type=ParamTypes.NUCLEI_NUCLEI,
            bias=False,
            chunk_axis=-1,
        )
        # Set envelopes output correctly
        env = self.envelope.clone(
            out_dim=self.orb_per_nuc * self.determinants, out_per_nuc=True
        )

        same_orbs = PerNucOrbitals(
            self.determinants, self.orb_per_nuc, env.clone(pi_init=1.0)
        )(systems, elec_embeddings)
        # init diff orbs with 0
        diff_orbs = PerNucOrbitals(
            self.determinants, self.orb_per_nuc, env.clone(pi_init=0.0)
        )(systems, elec_embeddings)
        # If n_elec is odd, we need an extra orbital
        fill_orbs = PerNucOrbitals(
            self.determinants, 1, env.clone(out_dim=self.determinants, pi_init=1.0)
        )(systems, elec_embeddings)

        result = []
        for diag, offdiag, fill, A, (spins, charges) in zip(
            same_orbs,
            diff_orbs,
            fill_orbs,
            systems.group(A, A_meta.param_type.value.chunk_fn),
            systems.unique_spins_and_charges,
        ):
            n_elec, n_up, n_nuc = sum(spins), spins[0], len(charges)

            @jax.vmap  # vmap over different molecules
            def _orbitals(diag: Array, offdiag: Array, A: Array, fill: Array):
                # Orbitals
                uu, dd, ud, du = diag[:n_up], diag[n_up:], offdiag[:n_up], offdiag[n_up:]
                orbitals = jnp.concatenate(
                    [
                        jnp.concatenate([uu, ud], axis=1),
                        jnp.concatenate([du, dd], axis=1),
                    ],
                    axis=0,
                )  # (n_elec, 2*n_orbs, n_det)
                orbitals = jnp.moveaxis(orbitals, -1, 0)  # (n_det, n_elec, 2*n_orbs)

                # A: (2*n_orbs, 2*n_orbs, n_det)
                A = einops.rearrange(A, '(n1 n2) o1 o2 d -> d (n1 o1) (n2 o2)', n1=n_nuc)
                A_diag = (A - A.mT) / 2
                A_offdiag = (A + A.mT) / 2
                A = jnp.concatenate(
                    [
                        jnp.concatenate([A_diag, A_offdiag], axis=2),
                        jnp.concatenate([-A_offdiag, A_diag], axis=2),
                    ],
                    axis=1,
                )

                # fill - orbitals = n_nuc
                fill = einops.rearrange(fill, 'elec orb det -> det elec orb').sum(axis=-1)

                # Product
                orb_A_orb_product = skewsymmetric_quadratic(orbitals, A)
                if n_elec % 2 == 1:
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
        grouped_orbs: Sequence[PfaffianOrbitals],  # grouped by molecules
        state: Sequence[EMAState[PfaffianPretrainingState]],  # list of molecules
    ):
        @jax.vmap
        def hf_match(
            hf_up: jax.Array,
            hf_down: jax.Array,
            pf_orbs: PfaffianOrbitals,
            state: EMAState[PfaffianPretrainingState],
        ):
            leading_dims = hf_up.shape[:-2]
            n_up, n_down = hf_up.shape[-2], hf_down.shape[-2]
            n_orbs = pf_orbs.orbitals.shape[-1] // 2

            # If the number of electrons is odd, we need to add a dummy electron
            # W.l.o.g., we assume n_up >= n_down
            if (n_up + n_down) % 2 == 1:
                eye = jnp.broadcast_to(
                    jnp.eye(n_down + 1), (*leading_dims, n_down + 1, n_down + 1)
                )
                hf_down = eye.at[..., :n_down, :n_down].set(hf_down)
                n_down += 1

            # Add dimension for the number of determinants
            if hf_up.ndim == pf_orbs.orbitals.ndim - 1:
                hf_up = hf_up[..., None, :, :]
                hf_down = hf_down[..., None, :, :]

            # Prepare NN
            # Orbital matching
            nn_up = pf_orbs.orbitals[..., :n_up, :n_orbs]
            nn_down = pf_orbs.orbitals[..., n_up:, n_orbs:]
            # Pfaffian matching
            nn_pf = pf_orbs.orb_A_orb_product

            # Prepare HF
            # These two are for matching orbitals
            hf_up_pad = jnp.concatenate(
                [hf_up, jnp.zeros((*hf_up.shape[:-1], n_orbs - n_up))],
                axis=-1,
            )[..., : nn_up.shape[-2], :]  # Remove padded electrons!
            hf_down_pad = jnp.concatenate(
                [hf_down, jnp.zeros((*hf_down.shape[:-1], n_orbs - n_down))],
                axis=-1,
            )[..., : nn_down.shape[-2], :]  # Remove padded electrons!
            # This one is for matching pfaffians
            hf_full = _hf_to_full(hf_up, hf_down)

            def transform(state: PfaffianPretrainingState):
                orbitals = jax.vmap(cayley_transform)(state.orbitals)
                pfaffian = to_skewsymmetric_orthogonal(state.pfaffian)
                return orbitals, pfaffian

            def loss(state: PfaffianPretrainingState, final: bool = False):
                orb_t, pf_t = transform(state)
                up, down, pf = nn_up, nn_down, nn_pf
                orb_weight, pf_weight = self.hf_match_orbitals, self.hf_match_pfaffian
                if not final:
                    up, down, pf = jax.lax.stop_gradient((up, down, pf))
                    orb_weight = pf_weight = 1

                loss_val = jnp.zeros((), dtype=pf.dtype)
                loss_val += ((hf_up_pad @ orb_t[0] - up) ** 2).mean() * orb_weight
                loss_val += ((hf_down_pad @ orb_t[1] - down) ** 2).mean() * orb_weight
                hf_pf = skewsymmetric_quadratic(hf_full, pf_t)
                loss_val += ((hf_pf - pf) ** 2).mean() * pf_weight
                return loss_val

            def avg_loss_and_grad(x: PfaffianPretrainingState):
                return pmean_if_pmap(jax.value_and_grad(loss)(x))

            def optim(
                optimizer: optax.GradientTransformation,
                x: PfaffianPretrainingState,
                maxiter: int = self.hf_match_steps,
            ):
                def step(state, i):
                    params, opt_state = state
                    value, grads = avg_loss_and_grad(params)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, opt_state), value

                (x, _), _ = jax.lax.scan(
                    step,
                    (x, optimizer.init(x)),  # type: ignore
                    jnp.arange(maxiter),
                )  # type: ignore
                return x

            # Initialize optimizer
            optimizer = optax.contrib.prodigy(self.hf_match_lr)
            # Update the state
            state = ema_update(state, optim(optimizer, ema_value(state)), 0.99)
            loss_val = loss(ema_value(state), final=True)
            return loss_val, state

        def stack(*x):
            return jnp.stack(x)

        out_state: Sequence[EMAState[PfaffianPretrainingState]] = []
        loss = jnp.zeros((), dtype=jnp.float32)
        for idx, pfaff_orbs in zip(systems.unique_indices, grouped_orbs):
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
        return loss, out_state

    def init_systems(self, key: Array, systems: SystemsWithHF):
        states = []
        for sub_sys in systems.sub_configs:
            n_up, n_down = sub_sys.spins[0]
            if sub_sys.n_elec % 2 == 1:
                n_down += 1
            n_orbs = sub_sys.n_nuc * self.orb_per_nuc
            state = ema_make(
                PfaffianPretrainingState(
                    orbitals=jnp.zeros((2, n_orbs, n_orbs), dtype=jnp.float32),
                    pfaffian=jnp.zeros((n_up + n_down, n_up + n_down), dtype=jnp.float32),
                )
            )
            states.append(state)
        return systems.replace(cache=tuple(states))
