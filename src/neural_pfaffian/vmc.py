import functools
from typing import Generic, TypeVar

import folx
import jax
import jax.numpy as jnp
import optax
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Float, Integer

from neural_pfaffian.clipping import Clipping
from neural_pfaffian.hamiltonian import KineticEnergyOp, make_local_energy
from neural_pfaffian.mcmc import MetroplisHastings
from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    WaveFunctionParameters,
)
from neural_pfaffian.preconditioner import Preconditioner
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils.jax_utils import (
    REPLICATE_SHARD,
    distribute_keys,
    jit,
    pmean,
    shmap,
)
from neural_pfaffian.utils.tree_utils import tree_squared_norm

LocalEnergy = Float[Array, 'batch_size n_mols']
S = TypeVar('S', bound=Systems)


@functools.partial(jax.vmap, in_axes=-1, out_axes=-1)
def local_energy_diff(e_loc: LocalEnergy) -> LocalEnergy:
    return e_loc - pmean(jnp.mean(e_loc))


O = TypeVar('O')
OS = TypeVar('OS')
PS = TypeVar('PS')


class VMCState(Generic[PS], PyTreeNode):
    params: WaveFunctionParameters
    optimizer: optax.OptState
    preconditioner: PS
    step: Integer[Array, '']

    @property
    def sharding(self):
        return REPLICATE_SHARD


class VMC(Generic[PS, O, OS], PyTreeNode):
    wave_function: GeneralizedWaveFunction[O, OS] = field(pytree_node=False)
    preconditioner: Preconditioner[PS] = field(pytree_node=False)
    optimizer: optax.GradientTransformation = field(pytree_node=False)
    sampler: MetroplisHastings = field(pytree_node=False)
    clipping: Clipping = field(pytree_node=False)

    def init(self, key: Array, systems: Systems):
        params = self.wave_function.init(key, systems.example_input)
        return VMCState(
            params=params,
            optimizer=self.optimizer.init(params),  # type: ignore
            preconditioner=self.preconditioner.init(params),
            step=jnp.zeros((), dtype=jnp.int32),
        )

    def init_systems(self, key: Array, systems: S) -> S:
        @shmap(
            in_specs=(REPLICATE_SHARD, systems.sharding),
            out_specs=systems.sharding,
        )
        def init(key: Array, systems: S):
            key = distribute_keys(key)
            key, subkey = jax.random.split(key)
            systems = self.sampler.init_systems(subkey, systems)
            return systems

        return init(key, systems)

    @jit
    def local_energy(self, state: VMCState, systems: Systems):
        local_energy_fn = make_local_energy(self.wave_function, KineticEnergyOp.FORWARD)
        local_energy_fn = folx.batched_vmap(
            local_energy_fn,
            max_batch_size=32,
            in_axes=(None, systems.electron_vmap, None),
        )
        e_l = local_energy_fn(
            state.params,
            systems,
            self.wave_function.reparams(state.params, systems),
        )
        return e_l

    @jit
    def mcmc_step(self, key: Array, state: VMCState[PS], systems: Systems):
        @shmap(
            in_specs=(REPLICATE_SHARD, state.sharding, systems.sharding),
            out_specs=systems.sharding,
            check_rep=False,
        )
        def _mcmc_step(key: Array, state: VMCState[PS], systems: Systems):
            key = distribute_keys(key)
            # Sampling
            key, subkey = jax.random.split(key)
            systems = self.sampler(subkey, state.params, systems)
            return systems

        return _mcmc_step(key, state, systems)

    @jit
    def step(self, key: Array, state: VMCState[PS], systems: Systems):
        @shmap(
            in_specs=(REPLICATE_SHARD, state.sharding, systems.sharding),
            out_specs=(state.sharding, systems.sharding, REPLICATE_SHARD),
            check_rep=False,
        )
        def _step(key: Array, state: VMCState[PS], systems: Systems):
            key = distribute_keys(key)
            # Sampling
            key, subkey = jax.random.split(key)
            systems = self.sampler(subkey, state.params, systems)

            # Local energy
            e_l = self.local_energy(state, systems)
            clipped_e_l = self.clipping(e_l)
            dE_dlogpsi = local_energy_diff(clipped_e_l)

            # Preconditioning
            gradient, preconditioner_state, aux_data = self.preconditioner.apply(
                state.params, systems, dE_dlogpsi, state.preconditioner
            )

            # Update step
            updates, opt_state = self.optimizer.update(gradient, state.optimizer)  # type: ignore
            params = optax.apply_updates(state.params, updates)  # type: ignore

            # Logging
            E_per_mol = pmean(e_l.mean(0))
            E = E_per_mol.mean()
            E_std = (pmean(((e_l - E_per_mol) ** 2).mean(0)) ** 0.5).mean()
            grad_norm = tree_squared_norm(gradient) ** 0.5
            aux_data = aux_data | dict(E=E, E_std=E_std, grad=grad_norm)

            return (
                state.replace(
                    params=params,
                    optimizer=opt_state,
                    preconditioner=preconditioner_state,
                    step=state.step + 1,
                ),
                systems,
                aux_data,
            )

        return _step(key, state, systems)
