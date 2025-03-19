from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.struct import PyTreeNode, field
from jaxtyping import Array, ArrayLike, DTypeLike, Float

from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    WaveFunctionParameters,
)
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils.cg import cg
from neural_pfaffian.utils import Modules
from neural_pfaffian.utils.jax_utils import (
    jit,
    pall_to_all,
    pgather,
    pidx,
    pmean,
    psum_if_pmap,
    vmap,
)
from neural_pfaffian.utils.tree_utils import (
    tree_add,
    tree_mul,
    tree_squared_norm,
    tree_to_dtype,
)

PS = TypeVar('PS')


class Preconditioner[PS](Protocol):
    def init(self, params: WaveFunctionParameters) -> PS: ...

    def apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        dE_dlogpsi: Float[Array, 'batch_size n_mols'],
        state: PS,
    ) -> tuple[WaveFunctionParameters, PS, dict[str, Float[Array, '']]]: ...


class Identity(PyTreeNode, Preconditioner[None]):
    wave_function: GeneralizedWaveFunction = field(pytree_node=False)

    def init(self, params: WaveFunctionParameters) -> None:
        return None

    @jit
    def apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        dE_dlogpsi: Float[Array, 'batch_size n_mols'],
        state: None,
    ):
        N = dE_dlogpsi.size * jax.device_count()

        def log_p_closure(params):
            return self.wave_function.batched_apply(params, systems) / N

        _, vjp = jax.vjp(log_p_closure, params)
        grad = psum_if_pmap(vjp(dE_dlogpsi)[0])
        precond_grad_norm = jnp.sqrt(sum([jnp.sum(g**2) for g in jtu.tree_leaves(grad)]))
        return grad, state, {'opt/precond_grad_norm': precond_grad_norm}


class CGState(PyTreeNode):
    last_grad: WaveFunctionParameters
    damping: Float[Array, '']


class CG(PyTreeNode, Preconditioner[CGState]):
    wave_function: GeneralizedWaveFunction = field(pytree_node=False)
    damping: Float[ArrayLike, '']
    decay_factor: Float[ArrayLike, '']
    maxiter: int = field(pytree_node=False)

    def init(self, params: WaveFunctionParameters):
        return SpringState(
            last_grad=jtu.tree_map(lambda x: jnp.zeros_like(x), params),
            damping=jnp.asarray(self.damping, dtype=jnp.float32),
        )

    @jit
    def apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        dE_dlogpsi: Float[Array, 'batch_size n_mols'],
        state: CGState,
    ):
        n_dev = jax.device_count()
        local_batch_size = dE_dlogpsi.size
        N = local_batch_size * n_dev
        normalization = 1 / jnp.sqrt(N)

        def log_p_closure(p):
            return self.wave_function.batched_apply(p, systems) * normalization

        _, vjp_fn = jax.vjp(log_p_closure, params)
        _, jvp_fn = jax.linearize(log_p_closure, params)

        def center_fn(x):
            # This centers on a per molecule basis rather than center everything
            x = x.reshape(dE_dlogpsi.shape)
            center = pmean(jnp.mean(x, axis=0, keepdims=True))
            return x - center

        def vjp(x):
            return psum_if_pmap(vjp_fn(center_fn(x).astype(dE_dlogpsi.dtype))[0])

        def jvp(x):
            return center_fn(jvp_fn(x))

        grad = psum_if_pmap(vjp(dE_dlogpsi * normalization))
        last_grad = state.last_grad
        last_grad = jtu.tree_map(jax.lax.convert_element_type, last_grad, grad)
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

        aux_data = dict(
            grad_norm=tree_squared_norm(grad) ** 0.5,
            natgrad_norm=tree_squared_norm(natgrad) ** 0.5,
            decayed_last_grad_norm=tree_squared_norm(decayed_last_grad) ** 0.5,
        )
        return (
            natgrad,
            state.replace(last_grad=natgrad),
            aux_data,
        )


class SpringState(PyTreeNode):
    last_grad: WaveFunctionParameters
    damping: Float[Array, '']


class Spring(PyTreeNode, Preconditioner[SpringState]):
    wave_function: GeneralizedWaveFunction = field(pytree_node=False)
    damping: Float[ArrayLike, '']
    decay_factor: Float[ArrayLike, '']
    dtype: DTypeLike | None = field(pytree_node=False)

    def init(self, params: WaveFunctionParameters) -> SpringState:
        return SpringState(
            last_grad=jtu.tree_map(lambda x: jnp.zeros_like(x, dtype=self.dtype), params),
            damping=jnp.asarray(self.damping, dtype=jnp.float32),
        )

    @jit
    def apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        dE_dlogpsi: Float[Array, 'batch_size n_mols'],
        state: SpringState,
    ):
        n_dev = jax.device_count()
        local_batch_size = dE_dlogpsi.size
        N = local_batch_size * n_dev
        normalization = 1 / jnp.sqrt(N)

        out_dtypes = jtu.tree_map(lambda x: x.dtype, params)
        if self.dtype is not None:
            params, systems, dE_dlogpsi = tree_to_dtype(
                (params, systems, dE_dlogpsi), self.dtype
            )

        @jit
        def log_p(params, systems):
            return self.wave_function.apply(params, systems) * normalization  # type: ignore

        @jit
        def log_p_closure(params):
            return jax.vmap(log_p, in_axes=(None, systems.electron_vmap))(params, systems)

        def vjp(x):
            return psum_if_pmap(jax.vjp(log_p_closure, params)[1](center_fn(x))[0])

        def jvp(x):
            return center_fn(jax.jvp(log_p_closure, (params,), (x,))[1])

        def center_fn(
            x: Float[Array, 'batch_size n_mols'],
        ) -> Float[Array, 'batch_size n_mols']:
            x = x.reshape(dE_dlogpsi.shape)
            center = pmean(jnp.mean(x, axis=0))
            return x - center

        jacs: list[list[jax.Array]] = []
        for elec, nuc, (spins, charges) in systems.iter_grouped_molecules():
            sub_systems = Systems((spins,), (charges,), elec, nuc, {})

            @vmap(in_axes=(None, sub_systems.electron_vmap))
            @vmap(in_axes=(None, sub_systems.molecule_vmap))
            @jax.grad
            def jac_fn(params, systems):
                return log_p(params, systems).sum()

            jacs.append(jac_fn(params, sub_systems))

        @jit
        def to_covariance(*jacs: jax.Array) -> jax.Array:
            # merge all systems into a single jacobian
            # jac is (N, mols, params)
            jacs = tuple(x.reshape(*x.shape[:2], -1) for x in jacs)
            jac = jnp.concatenate(jacs, axis=1)[:, systems.inverse_unique_indices]
            n_params = jac.shape[-1]
            # check for parameters that are not split evenly across devices
            num_even = n_params // n_dev * n_dev
            jac, remainder = jac[..., :num_even], jac[..., num_even:]
            jac = pall_to_all(jac, split_axis=2, concat_axis=0, tiled=True)
            jac -= jac.mean(axis=0)
            jac = jac.reshape(N, -1)
            # for the remainder we copy it to all devices
            remainder = pgather(remainder, axis=0, tiled=True)
            remainder -= remainder.mean(axis=0)
            remainder = remainder.reshape(N, -1)
            # Since the remainder is summed n_dev times we need to divide by n_dev
            return jac @ jac.T + remainder @ remainder.T / n_dev

        # Compute covariance
        JT_J = jtu.tree_reduce(jnp.add, jtu.tree_map(to_covariance, *jacs))
        JT_J = psum_if_pmap(JT_J)
        # These don't change anything if the numerics are correct and are just for numerical stability
        JT_J = (JT_J + JT_J.T) / 2

        # construct epsilon tilde
        # we hack this into the natgrad state
        decayed_last_grad = tree_mul(state.last_grad, self.decay_factor)
        epsilon_tilde = dE_dlogpsi * normalization - jvp(decayed_last_grad)
        epsilon_tilde = pgather(epsilon_tilde, axis=0, tiled=True)
        epsilon_tilde = epsilon_tilde.astype(self.dtype).reshape(-1)

        sigma, U = jnp.linalg.eigh(JT_J, symmetrize_input=True)
        # Ensure that condition number is at worst 1e10
        damping = jnp.maximum(sigma[-1] / 1e10, state.damping)
        sigma = jnp.maximum(sigma, 0) + damping

        x = U.reshape(n_dev, -1, U.shape[1])[pidx()] @ ((U.T @ epsilon_tilde) / sigma)
        preconditioned = vjp(x)
        natgrad = tree_add(preconditioned, decayed_last_grad)
        # Convert back to the original dtype
        update = jtu.tree_map(jax.lax.convert_element_type, natgrad, out_dtypes)
        aux_data = {
            'cond_nr10': jnp.log10(sigma[-1]) - jnp.log10(sigma[0]),
            'grad_norm': tree_squared_norm(dE_dlogpsi) ** 0.5,
        }
        return update, state.replace(last_grad=natgrad), aux_data


PRECONDITIONER = Modules[Preconditioner](
    {cls.__name__.lower(): cls for cls in [Identity, CG, Spring]}
)
