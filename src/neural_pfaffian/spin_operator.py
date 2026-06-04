import functools
from typing import Generic, Literal, TypeVar

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Int
from optax import Schedule

from neural_pfaffian.clipping import Clipping, Masking
from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    WaveFunctionParameters,
)
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils import EMA
from neural_pfaffian.utils.jax_utils import jit, pmin_if_pmap, psum_if_pmap
from neural_pfaffian.utils.schedule import ScheduleConfig, get_schedule
from neural_pfaffian.utils.summary_stats import weighted_mean
from neural_pfaffian.utils.tree_utils import (
    tree_add,
    tree_mul,
    tree_squared_norm,
)

O = TypeVar('O')
"""Orbital type"""
COrb = TypeVar('COrb')
"""Core orbital type"""
OS = TypeVar('OS')
"""Orbital state type"""
S = TypeVar('S', bound=Systems)
_SPIN_BETA = 'spin_beta'
_SPIN_EMA = 'spin_ema'


class SpinPenalty(Generic[O, COrb, OS, S], PyTreeNode):
    wave_function: GeneralizedWaveFunction[O, COrb, OS, S] = field(
        pytree_node=False,
    )
    sample_masking: Masking = field(pytree_node=False)
    ratio_clipping: Clipping = field(pytree_node=False)

    penalty_scale: Schedule = field(
        default_factory=lambda: get_schedule(1.0),
        pytree_node=False,
    )
    max_grad_norm: float = 1.0
    penalty_type: Literal['minimize', 'snap'] = 'minimize'
    spin_ema_decay: Schedule = field(
        default_factory=lambda: get_schedule(0.9),
        pytree_node=False,
    )

    @classmethod
    def create(
        cls,
        penalty_scale: ScheduleConfig,
        penalty_type: Literal['minimize', 'snap'],
        spin_ema_decay: ScheduleConfig,
        **kwargs,
    ):
        assert penalty_type in ['minimize', 'snap'], (
            f'Unknown penalty_type: {penalty_type}'
        )
        return cls(
            penalty_scale=get_schedule(penalty_scale),
            penalty_type=penalty_type,
            spin_ema_decay=get_schedule(spin_ema_decay),
            **kwargs,
        )

    def init_systems(self, systems: S) -> S:
        if _SPIN_BETA not in systems.mol_data:
            systems = systems.set_mol_data(
                _SPIN_BETA,
                jnp.zeros((systems.n_mols,), dtype=jnp.int32),
            )
        if _SPIN_EMA not in systems.mol_data:
            spin_ema = EMA.init(data=jnp.zeros((), dtype=jnp.float32))
            spin_ema = jax.tree.map(
                lambda x: jnp.stack([x] * systems.n_mols, axis=0),
                spin_ema,
            )
            systems = systems.set_mol_data(_SPIN_EMA, spin_ema)
        return systems

    def __call__(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        step: Int[Array, ''],
    ):
        @jit
        def batched_slogpsi(params, systems, reparams=None):
            batch_slogpsi = jax.vmap(
                functools.partial(self.wave_function.signed, reparams=reparams),
                in_axes=(None, systems.electron_vmap),
            )(
                params,
                systems,
            )
            return batch_slogpsi

        @jit
        def slog_psi_closure(params):
            batch_slogpsi = batched_slogpsi(params, systems)
            return batch_slogpsi

        ((base_signs, base_logpsis), vjp) = jax.vjp(slog_psi_closure, params)
        # [n_batch, n_mols]
        mask = self.sample_masking(
            base_signs * jnp.exp(-(base_logpsis - pmin_if_pmap(base_logpsis.min()))),
        )
        penalty_scale = self.penalty_scale(step)
        penalty_norm = jnp.asarray(
            2.0 * penalty_scale / systems.n_mols,
            dtype=base_logpsis.dtype,
        )

        sz = 0.5 * jnp.abs(jnp.asarray(systems.total_spins))
        S2_ema = systems.get_mol_data(_SPIN_EMA)
        S2_smooth = S2_ema.value()  # <S^2> = S(S+1)
        if self.penalty_type == 'snap':
            # e.g. N_up - N_down = 2 => S_z = 1 => S_target in [1, 2, 3] => S^2_target in [2, 6, 12]
            S_candidates = jnp.stack(
                [sz + i for i in range(0, 3)],
                axis=-1,
            )  # (n_mols, 3)
            S2_candidates = S_candidates * (S_candidates + 1.0)  # (n_mols, 3)
            S2_idx = jnp.argmin(
                jnp.abs(S2_candidates - S2_smooth[:, None]),
                axis=-1,
            )  # (n_mols,)
            S2_target = jnp.take_along_axis(
                S2_candidates,
                S2_idx[:, None],
                axis=-1,
            ).squeeze(
                -1,
            )  # (n_mols,)
            P_plus_shift = (S2_target - (sz * (sz + 1.0))) / jnp.asarray(
                systems.n_down_by_mol,
                dtype=base_logpsis.dtype,
            )  # (n_mols,)
        else:
            P_plus_shift = jnp.zeros(
                (systems.n_mols,),
                dtype=base_logpsis.dtype,
            )

        permutation_grad = jtu.tree_map(jnp.zeros_like, params)
        logpsi_cotangents = []
        R_betas = []
        P_pluses = []
        for indices in systems.unique_indices:
            subsystem = systems[indices]
            # All systems have the same spins and charges
            betas = subsystem.get_mol_data(_SPIN_BETA)  # (n_mols,)
            sub_base_signs = base_signs[:, np.array(indices)]  # (n_batch, n_mols)
            sub_base_logpsis = base_logpsis[:, np.array(indices)]  # (n_batch, n_mols)
            sub_mask = mask[:, np.array(indices)]  # (n_batch, n_mols)
            sub_P_plus_shift = P_plus_shift[np.array(indices)]  # (n_mols,)
            n_up, n_down = subsystem.spins[0]
            n_elec = n_up + n_down
            electrons = subsystem.electrons  # (n_batch, n_mols * n_electrons, 3)
            n_mols = len(indices)

            offsets = jnp.arange(n_mols)[:, None] * n_elec
            beta_idx = betas + n_up

            def _swap_ratio(params, base_logpsis, alpha, reparams=None):
                swap_idx = (
                    jnp.stack(jnp.broadcast_arrays(alpha, beta_idx), axis=1) + offsets
                )
                swap_electrons = electrons.at[:, swap_idx.flatten()].set(
                    electrons[:, swap_idx[:, ::-1].flatten()],
                )
                new_sign, new_logpsi = batched_slogpsi(
                    params,
                    subsystem.replace(electrons=swap_electrons),
                    reparams=reparams,
                )
                return (
                    -new_sign
                    * sub_base_signs
                    * jnp.exp(
                        new_logpsi - base_logpsis,
                    )
                )

            def _sum_swap_ratios(params, base_logpsis):
                reparams = self.wave_function.reparams(params, subsystem)

                def _loop_fn(sum_ratios, alpha):
                    swap_ratio = _swap_ratio(
                        params,
                        base_logpsis,
                        alpha,
                        reparams=reparams,
                    )
                    return sum_ratios + swap_ratio, None

                sum_ratios, _ = jax.lax.scan(
                    _loop_fn,
                    jnp.zeros_like(base_logpsis),
                    jnp.arange(n_up),
                )
                return sum_ratios

            sum_ratios, vjp_fn = jax.vjp(
                _sum_swap_ratios,
                params,
                sub_base_logpsis,
            )

            # sum over swaps
            R_beta = 1 + sum_ratios
            R_beta = self.ratio_clipping(R_beta, mask=sub_mask)
            R_betas.append(R_beta)
            P_plus = weighted_mean(R_beta, sub_mask)
            P_pluses.append(P_plus)

            active_counts = psum_if_pmap(jnp.sum(sub_mask, axis=0)).astype(P_plus.dtype)
            scale = penalty_norm * (
                (P_plus - sub_P_plus_shift) / jnp.maximum(1.0, active_counts)
            )
            weights = sub_mask * scale

            grad_part1, dR_dlogpsi_weighted = vjp_fn(weights)
            permutation_grad = tree_add(permutation_grad, grad_part1)
            logpsi_cotangent = dR_dlogpsi_weighted + weights * (2.0 * (R_beta - P_plus))
            logpsi_cotangents.append(logpsi_cotangent)

        R_beta = jnp.concatenate(R_betas, axis=-1)[
            ...,
            systems.inverse_unique_indices,
        ]
        P_plus = jnp.concatenate(P_pluses, axis=-1)[
            ...,
            systems.inverse_unique_indices,
        ]
        logpsi_cotangent = jnp.concatenate(logpsi_cotangents, axis=-1)[
            ...,
            systems.inverse_unique_indices,
        ]
        dP_dsign = jnp.zeros_like(base_signs)
        grad_part2 = vjp((dP_dsign, logpsi_cotangent))[0]

        grad_part1 = psum_if_pmap(permutation_grad)
        grad_part2 = psum_if_pmap(grad_part2)
        grad1_norm = jnp.nan_to_num(tree_squared_norm(grad_part1) ** 0.5)
        grad2_norm = jnp.nan_to_num(tree_squared_norm(grad_part2) ** 0.5)

        gradient = tree_add(grad_part1, grad_part2)

        is_nan = jnp.isnan(jfu.ravel_pytree(gradient)[0]).any()
        gradient = jtu.tree_map(
            lambda x: jnp.where(is_nan, jnp.zeros_like(x), x),
            gradient,
        )

        # Clipping
        grad_norm = tree_squared_norm(gradient) ** 0.5
        clipped_norm = jnp.minimum(self.max_grad_norm, grad_norm)
        rescaling = clipped_norm / (grad_norm + 1e-8)
        gradient = tree_mul(gradient, rescaling)
        spin_var = weighted_mean((R_beta - P_plus) ** 2, mask)
        N_down = jnp.asarray(systems.n_down_by_mol, dtype=P_plus.dtype)

        # Update EMA
        S2_hat = sz * (sz + 1.0) + N_down * P_plus  # (n_mols,)
        decay = self.spin_ema_decay(step)
        S2_ema = EMA.update(S2_ema, S2_hat, decay)
        systems = systems.set_mol_data(_SPIN_EMA, S2_ema)

        aux_data = {
            'spin/P_plus': P_plus,
            'spin/P_plus_shift': P_plus_shift,
            'spin/estimator': S2_hat,
            'spin/estimator_smooth': S2_smooth,
            'spin/var': spin_var,
            'spin/std': spin_var**0.5,
            'spin/num_outlier': psum_if_pmap((~mask).sum()),
            'spin/num_nans': psum_if_pmap(jnp.isnan(R_beta).sum()),
            'spin/grad_1_norm': grad1_norm,
            'spin/grad_2_norm': grad2_norm,
            'spin/grad_norm': grad_norm,
            'spin/grad_norm_clipped': clipped_norm,
        }
        systems = self.update_beta(systems)

        return (gradient, aux_data), systems

    def update_beta(self, systems: Systems) -> Systems:
        return systems.set_mol_data(
            _SPIN_BETA,
            jnp.mod(
                (systems.get_mol_data(_SPIN_BETA) + 1),
                jnp.array(systems.n_down_by_mol, dtype=jnp.int32),
            ),
        )
