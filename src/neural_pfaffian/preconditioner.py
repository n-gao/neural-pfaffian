from collections.abc import Sequence
from typing import Protocol

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode, field
from jaxtyping import Array, ArrayLike, DTypeLike, Float

from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    WaveFunctionParameters,
)
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils import Modules
from neural_pfaffian.utils.cg import cg
from neural_pfaffian.utils.jax_utils import (
    jit,
    pall_to_all,
    pgather,
    pidx,
    pmean,
    psum_if_pmap,
    vmap,
)
from neural_pfaffian.utils.segment_utils import unsegment_axis
from neural_pfaffian.utils.summary_stats import weighted_centering
from neural_pfaffian.utils.tree_utils import (
    tree_add,
    tree_mul,
    tree_squared_norm,
    tree_to_dtype,
)


class Preconditioner[PS](Protocol):
    def init(
        self,
        key: Array,
        params: WaveFunctionParameters,
        systems: Systems,
    ) -> PS: ...

    def apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        dL_dlogpsi: Float[Array, 'batch_size n_mols'],
        state: PS,
        auxiliary_grads: WaveFunctionParameters,
    ) -> tuple[WaveFunctionParameters, PS, dict[str, Float[Array, '']]]: ...


class Identity(PyTreeNode, Preconditioner[None]):
    wave_function: GeneralizedWaveFunction = field(pytree_node=False)
    dtype: DTypeLike | None = field(pytree_node=False, default=None)
    """The dtype to compute the gradient in. Careful! Contrary to other
    preconditioners this one will return the gradient in `dtype` and not cast back
    into the input types."""

    def init(self, key: Array, params: WaveFunctionParameters, systems: Systems) -> None:
        return None

    @jit
    def apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        dL_dlogpsi: Float[Array, 'batch_size n_mols'],
        state: None,
        auxiliary_grads: WaveFunctionParameters,
    ) -> tuple[WaveFunctionParameters, None, dict[str, Array]]:
        if self.dtype is not None:
            params, systems, dL_dlogpsi = tree_to_dtype(
                (params, systems, dL_dlogpsi),
                self.dtype,
            )

        N = dL_dlogpsi.size * jax.device_count()  # total number of samples
        out_dtypes = jax.tree.map(lambda x: x.dtype, params)

        def log_p_closure(params):
            return self.wave_function.batched_apply(params, systems) / N

        _, vjp_fn = jax.vjp(log_p_closure, params)

        def center_fn(x):
            x = x.reshape(dL_dlogpsi.shape)
            return weighted_centering(x)

        grad = psum_if_pmap(vjp_fn(center_fn(dL_dlogpsi).astype(dL_dlogpsi.dtype))[0])
        grad = tree_add(grad, auxiliary_grads)

        update = jax.tree.map(jax.lax.convert_element_type, grad, out_dtypes)

        return update, state, {}


class CGState(PyTreeNode):
    last_grad: WaveFunctionParameters
    damping: Float[Array, '']


class CG(PyTreeNode, Preconditioner[CGState]):
    wave_function: GeneralizedWaveFunction = field(pytree_node=False)
    damping: Float[ArrayLike, '']
    decay_factor: Float[ArrayLike, '']
    maxiter: int = field(pytree_node=False)
    precondition_aux_grads: bool = field(pytree_node=False, default=True)

    def init(self, key: Array, params: WaveFunctionParameters, systems: Systems):
        return CGState(
            last_grad=jax.tree.map(lambda x: jnp.zeros_like(x), params),
            damping=jnp.asarray(self.damping, dtype=jnp.float32),
        )

    @jit
    def apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        dL_dlogpsi: Float[Array, 'batch_size n_mols'],
        state: CGState,
        auxiliary_grads: WaveFunctionParameters,
    ):
        n_dev = jax.device_count()
        N = dL_dlogpsi.size * n_dev  # total number of samples
        normalization = 1 / jnp.sqrt(N)

        def log_p_closure(p):
            log_p = 2 * self.wave_function.batched_apply(p, systems) * normalization
            return log_p

        _, vjp_fn = jax.vjp(log_p_closure, params)
        _, jvp_fn = jax.linearize(log_p_closure, params)

        def center_fn(x):
            x = x.reshape(dL_dlogpsi.shape)
            return weighted_centering(x)

        def vjp(x):
            return psum_if_pmap(vjp_fn(center_fn(x).astype(dL_dlogpsi.dtype))[0])

        def jvp(x):
            return center_fn(jvp_fn(x))

        grad = psum_if_pmap(vjp(dL_dlogpsi * normalization))
        grad_types = jax.tree.map(lambda x: x.dtype, params)
        aux_grad = jax.tree.map(
            jax.lax.convert_element_type,
            auxiliary_grads,
            grad_types,
        )

        if self.precondition_aux_grads:
            grad = tree_add(grad, aux_grad)

        last_grad = state.last_grad
        last_grad = jax.tree.map(jax.lax.convert_element_type, last_grad, grad)
        decayed_last_grad = tree_mul(last_grad, self.decay_factor)
        b = tree_add(grad, tree_mul(decayed_last_grad, state.damping))

        @jit
        def Fisher_matmul(v):
            # J^T J v
            result = vjp(jvp(v))
            # add damping
            result = tree_add(result, tree_mul(v, state.damping))
            return result

        # Compute natural gradient
        natgrad = cg(
            A=Fisher_matmul,
            b=b,
            x0=last_grad,
            fixed_iter=n_dev > 1,  # multi gpu
            maxiter=self.maxiter,
        )[0]

        if not self.precondition_aux_grads:
            natgrad = tree_add(natgrad, aux_grad)

        aux_data = {
            'grad_norm': tree_squared_norm(grad) ** 0.5,
            'natgrad_norm': tree_squared_norm(natgrad) ** 0.5,
            'decayed_last_grad_norm': tree_squared_norm(decayed_last_grad) ** 0.5,
        }
        return (
            natgrad,
            state.replace(last_grad=natgrad),
            aux_data,
        )


def batch_parameters(
    arrays: Sequence[jax.Array],
    *extras: Sequence[Array],
    batch_size: int,
):
    current_set = [arrays[0]]
    current_ext_set = [[ext[0]] for ext in extras]
    current_size = arrays[0].shape[-1]

    def _yield(current_set, current_ext_set):
        # Skipping concatenation so no extra buffer is created for huge matrices.
        if len(current_set) == 1:
            return (current_set[0], *[ext[0] for ext in current_ext_set])
        return (
            jnp.concatenate(current_set, axis=-1),
            *[jnp.concatenate(ext, axis=-1) for ext in current_ext_set],
        )

    for arr, *ext in zip(arrays[1:], *[ext[1:] for ext in extras], strict=True):
        n = arr.shape[-1]
        if current_size + n > batch_size:
            yield _yield(current_set, current_ext_set)
            current_set = [arr]
            current_ext_set = [[e] for e in ext]
            current_size = n
        else:
            current_set.append(arr)
            for s, e in zip(current_ext_set, ext, strict=True):
                s.append(e)
            current_size += n
    yield _yield(current_set, current_ext_set)


class SpringState(PyTreeNode):
    last_grad: WaveFunctionParameters
    damping: Float[Array, '']


class Spring(PyTreeNode, Preconditioner[SpringState]):
    wave_function: GeneralizedWaveFunction = field(pytree_node=False)
    damping: Float[ArrayLike, '']
    decay_factor: Float[ArrayLike, '']
    aux_grad_cutoff: Float[ArrayLike, '']
    aux_grad_damping: Float[ArrayLike, '']
    aux_grad_global_damping: Float[ArrayLike, '']
    dtype: DTypeLike | None = field(pytree_node=False)
    clip_eigenvals: Float[ArrayLike, '']
    cutoff_to_zero: bool = field(pytree_node=False, default=True)
    max_acc_size: int = field(
        pytree_node=False,
        default=262_144_000,
    )  # max. 2000 MiB concat buffer

    def init(
        self,
        key: Array,
        params: WaveFunctionParameters,
        systems: Systems,
    ) -> SpringState:
        return SpringState(
            last_grad=jax.tree.map(
                lambda x: jnp.zeros_like(x, dtype=self.dtype),
                params,
            ),
            damping=jnp.asarray(self.damping, dtype=jnp.float32),
        )

    @jit
    def apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        dL_dlogpsi: Float[Array, 'batch_size n_mols'],
        state: SpringState,
        auxiliary_grads: WaveFunctionParameters,
    ):
        n_dev = jax.device_count()
        shape_N = dL_dlogpsi.size * n_dev  # total number of samples
        normalization = 1 / jnp.sqrt(shape_N)

        out_dtypes = jax.tree.map(lambda x: x.dtype, params)
        if self.dtype is not None:
            params, systems, dL_dlogpsi = tree_to_dtype(
                (params, systems, dL_dlogpsi),
                self.dtype,
            )

        @jit
        def log_p(params, systems):
            log_p = self.wave_function.apply(params, systems) * normalization
            return log_p

        @jit
        def log_p_closure(params):
            batch_log_p = jax.vmap(log_p, in_axes=(None, systems.electron_vmap))(
                params,
                systems,
            )  # (batch_size, n_mols)
            return batch_log_p

        def vjp(x):
            return psum_if_pmap(jax.vjp(log_p_closure, params)[1](center_fn(x))[0])

        def jvp(x):
            return center_fn(jax.jvp(log_p_closure, (params,), (x,))[1])

        def center_fn(
            x: Float[Array, 'batch_size n_mols'],
        ) -> Float[Array, 'batch_size n_mols']:
            x = x.reshape(dL_dlogpsi.shape)
            center = pmean(jnp.mean(x, axis=0))
            return x - center

        jacs: list[list[jax.Array]] = []
        segments: list = []
        for sub_systems in systems.iter_stacked_sub_systems():
            # Reduce n_mols dimension to save compute / memory
            electrons = sub_systems.electrons  # (walker, n_mols, n_electrons, 3)
            electrons = unsegment_axis(
                electrons,
                sub_systems.mol_id_groups,
                axis=1,
                num_segments=sub_systems.n_unique_mols,
            )  # (walker, n_unique_mols, n_states, n_electrons, 3)
            electrons = jnp.swapaxes(
                electrons,
                1,
                2,
            )  # (walker, n_states, n_unique_mols, n_electrons, 3)
            nuclei = sub_systems.nuclei  # (n_mols, n_nuclei, 3)
            nuclei = unsegment_axis(
                nuclei,
                sub_systems.mol_id_groups,
                axis=0,
                num_segments=sub_systems.n_unique_mols,
            )  # (n_unique_mols, n_states, n_nuclei, 3)
            # The nuclei are the same for all states, so we can just take the first one
            nuclei = nuclei[:, 0, ...]  # (n_unique_mols, n_nuclei, 3)

            _sub_systems = sub_systems.replace(
                electrons=electrons,
                nuclei=nuclei,
            )

            # Loop over excitations with concrete (static) indices. `excitations` is a
            # static pytree field, so it must not receive a vmap tracer: vmapping over it
            # would put an array into the pytree metadata. We therefore unroll the
            # n_states axis here instead of vmapping over it.
            state_jacs = []
            for excitation in range(_sub_systems.max_num_states):
                state_systems = _sub_systems.replace(
                    electrons=_sub_systems.electrons[:, excitation],
                    excitations=(excitation,),
                    mol_ids=(0,),
                )

                @vmap(in_axes=(None, state_systems.electron_vmap))  # walker
                @vmap(in_axes=(None, state_systems.molecule_vmap))  # n_unique_mol
                @jax.grad
                def jac_fn(params, systems):
                    return log_p(params, systems).sum()

                state_jacs.append(jac_fn(params, state_systems))
            # Re-stack the per-state jacobians into the (walker, n_states, n_unique_mol)
            # layout expected by `concat_jacobians`.
            jacs.append(jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *state_jacs))
            segments.append((sub_systems.mol_id_groups, sub_systems.excitations))

        @jit
        def concat_jacobians(*jacs: jax.Array) -> jax.Array:
            # merge all systems into a single jacobian
            # each jac is (N, n_states, n_unique_mol, params)
            _jacs = []
            for sub_jac, (mol_id_groups, excitations) in zip(
                jacs,
                segments,
                strict=True,
            ):
                # First resegment the (..., n_states, n_unique_mol, ...) dims based on the
                # individual groups segmentation
                sub_jac = sub_jac[:, excitations, mol_id_groups]
                # jac_array is now (N, n_mols, *params) -> flatten params
                _jacs.append(sub_jac.reshape(*sub_jac.shape[:2], -1))
            jac = jnp.concatenate(_jacs, axis=1)[:, systems.inverse_unique_indices]
            return jac

        jac = jax.tree.map(concat_jacobians, *jacs)
        jac_tensors = jax.tree.leaves(jac)

        @jit
        def to_covariance(jac_: tuple[jax.Array, ...]) -> jax.Array:
            jac, *_ = jac_
            n_params = jac.shape[-1]
            # check for parameters that are not split evenly across devices
            num_even = n_params // n_dev * n_dev
            jac, remainder = jac[..., :num_even], jac[..., num_even:]
            jac: Array = pall_to_all(jac, split_axis=2, concat_axis=0, tiled=True)
            jac -= jac.mean(axis=0)
            jac = jac.reshape(shape_N, -1)

            # no need to materalize an NxN constant zero matrix!
            if n_params % n_dev == 0:
                return jac @ jac.T

            # for the remainder we copy it to all devices
            remainder = pgather(remainder, axis=0, tiled=True)
            remainder -= remainder.mean(axis=0)
            remainder = remainder.reshape(shape_N, -1)
            # Since the remainder is summed n_dev times we need to divide by n_dev
            return jac @ jac.T + remainder @ remainder.T / n_dev

        JT_J = psum_if_pmap(
            sum(
                map(
                    to_covariance,
                    batch_parameters(
                        jac_tensors,
                        # concat buffers are [N, batch_size]
                        batch_size=self.max_acc_size // shape_N,
                    ),
                ),
            ),
        )

        # Constructing the Fisher matrix
        T = (JT_J + JT_J.T) / 2
        s, U = jnp.linalg.eigh(T)
        damping = jnp.maximum(s[-1] / 1e10, state.damping)
        s = jnp.maximum(s, 0)  # Ensure positive definiteness
        damped_s = s + damping
        log10_condition = jnp.log10(damped_s[-1]) - jnp.log10(
            damped_s[0],
        )

        # Collect aux grads
        if self.dtype is not None:
            auxiliary_grads = tree_to_dtype(auxiliary_grads, self.dtype)

        # Process aux grads
        epsilon_aux = (
            pgather(jvp(auxiliary_grads), axis=0, tiled=True)
            .astype(self.dtype)
            .reshape(-1)
        )
        aux_coeffs = U.reshape(n_dev, -1, shape_N)[pidx()] @ jnp.where(
            s < self.aux_grad_cutoff,
            0 if self.cutoff_to_zero else (U.T @ epsilon_aux) / damped_s,
            (U.T @ epsilon_aux)
            / (damped_s * (s + self.aux_grad_damping) + self.aux_grad_global_damping),
        )

        # Process momentum
        decayed_last_grad = tree_mul(state.last_grad, self.decay_factor)
        # Process energy gradient
        epsilon_E = dL_dlogpsi * normalization - jvp(
            decayed_last_grad,
        )
        epsilon_E = pgather(epsilon_E, axis=0, tiled=True).astype(self.dtype).reshape(-1)

        x = (U.T @ epsilon_E) / damped_s
        x = jnp.where(s > self.clip_eigenvals, x, 0.0)
        x = U.reshape(n_dev, -1, shape_N)[pidx()] @ x
        preconditioned = vjp(x + aux_coeffs)

        natgrad = tree_add(preconditioned, decayed_last_grad)

        aux_data = {
            'log10_cond': log10_condition,
            'largest_eigenvalue': s[-1],
            'lowest_eigenvalue': s[0],
            'epsilon_E_norm': jnp.linalg.norm(epsilon_E),
            'dL_dlogpsi_norm': jnp.sqrt(psum_if_pmap(jnp.sum(dL_dlogpsi**2))),
            'T_inv_epsilon_norm': jnp.sqrt(psum_if_pmap(jnp.sum(x**2))),
            'epsilon_aux_norm': jnp.linalg.norm(epsilon_aux),
            'aux_coeffs_norm': jnp.sqrt(psum_if_pmap(jnp.sum(aux_coeffs**2))),
            'preconditioned_grad_norm': tree_squared_norm(preconditioned) ** 0.5,
            'natgrad_norm': tree_squared_norm(natgrad) ** 0.5,
        }
        # Convert back to the original dtype
        update = jax.tree.map(jax.lax.convert_element_type, natgrad, out_dtypes)

        return update, state.replace(last_grad=natgrad), aux_data


PRECONDITIONER = Modules[Preconditioner](
    {cls.__name__.lower(): cls for cls in [Identity, CG, Spring]},
)
