import functools
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
from neural_pfaffian.utils import Modules
from neural_pfaffian.utils.jax_utils import (
    pall_to_all,
    pgather,
    pidx,
    pmean,
    psum_if_pmap,
)
from neural_pfaffian.utils.tree_utils import tree_add, tree_mul, tree_to_dtype

PS = TypeVar('PS')


class Preconditioner(Protocol[PS]):
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

        def log_p(params, systems):
            return self.wave_function.apply(params, systems) * normalization  # type: ignore

        def center_fn(
            x: Float[Array, 'batch_size n_mols'],
        ) -> Float[Array, 'batch_size n_mols']:
            x = x.reshape(dE_dlogpsi.shape)
            center = pmean(jnp.mean(x, axis=0))
            return x - center

        jacs: list[list[jax.Array]] = []
        for elec, nuc, (spins, charges) in systems.iter_grouped_molecules():
            sub_systems = Systems((spins,), (charges,), elec, nuc, {})

            @functools.partial(jax.vmap, in_axes=(None, sub_systems.electron_vmap))
            @functools.partial(jax.vmap, in_axes=(None, sub_systems.molecule_vmap))
            @jax.grad
            def jac_fn(params, systems):
                return log_p(params, systems).sum()

            jac_tree = jac_fn(params, sub_systems)
            jac_tree = jtu.tree_map(lambda x: x.reshape(*elec.shape[:2], -1), jac_tree)
            jacs.append(jtu.tree_leaves(jac_tree))

        def to_covariance(jacs: list[jax.Array]) -> jax.Array:
            # merge all systems into a single jacobian
            # jac is (N, mols, params)
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
        JT_J = jnp.zeros((N, N), dtype=self.dtype)
        for i in range(len(jacs[0])):
            JT_J += to_covariance([j[i] for j in jacs])
        JT_J = psum_if_pmap(JT_J)

        def log_p_closure(params):
            return jax.vmap(log_p, in_axes=(None, systems.electron_vmap))(params, systems)

        _, vjp_fn = jax.vjp(log_p_closure, params)

        def vjp(x):
            return psum_if_pmap(vjp_fn(center_fn(x))[0])

        def jvp(x):
            return center_fn(jax.jvp(log_p_closure, (params,), (x,))[1])

        # construct epsilon tilde
        # we hack this into the natgrad state
        decayed_last_grad = tree_mul(state.last_grad, self.decay_factor)
        epsilon_tilde = dE_dlogpsi * normalization - jvp(decayed_last_grad)
        epsilon_tilde = pgather(epsilon_tilde, axis=0, tiled=True)
        epsilon_tilde = epsilon_tilde.astype(self.dtype).reshape(-1)

        T = JT_J + state.damping * jnp.eye(N, dtype=JT_J.dtype) + 1 / N
        # mathematically does nothing but may have better numerics
        T = (T + T.T) / 2

        x = jax.scipy.linalg.solve(T, epsilon_tilde, assume_a='pos', check_finite=False)
        x = x.reshape(n_dev, -1)[pidx()]
        preconditioned = vjp(x)
        natgrad = tree_add(preconditioned, decayed_last_grad)
        # Convert back to the original dtype
        update = jtu.tree_map(jax.lax.convert_element_type, natgrad, out_dtypes)
        return update, state.replace(last_grad=natgrad), {}


PRECONDITIONER = Modules[Preconditioner](
    {cls.__name__.lower(): cls for cls in [Identity, Spring]}
)
