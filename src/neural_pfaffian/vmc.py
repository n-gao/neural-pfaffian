from enum import Enum
from typing import Generic, TypeVar

import folx
import jax
import jax.numpy as jnp
import optax
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Float, Integer

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
    pgather,
    pmean,
    shmap,
)
from neural_pfaffian.utils.tree_utils import tree_squared_norm

LocalEnergy = Float[Array, 'batch_size n_mols']


class ClipStatistic(Enum):
    MEDIAN = 'median'
    MEAN = 'mean'


def clip_local_energies(
    e_loc: LocalEnergy, clip_local_energy: float, stat: ClipStatistic
) -> LocalEnergy:
    stat = ClipStatistic(stat)
    match stat:
        case ClipStatistic.MEAN:
            stat_fn = jnp.mean
        case ClipStatistic.MEDIAN:
            stat_fn = jnp.median
        case _:
            raise ValueError(f'Unknown statistic {stat}')
    if clip_local_energy > 0.0:
        full_e = pgather(e_loc, axis=0, tiled=True)
        clip_center = stat_fn(full_e, keepdims=True)
        mad = jnp.mean(jnp.abs(full_e - clip_center), keepdims=True)
        max_dev = clip_local_energy * mad
        e_loc = jnp.clip(e_loc, clip_center - max_dev, clip_center + max_dev)
    return e_loc


def local_energy_diff(
    e_loc: LocalEnergy, clip_local_energy: float, stat: ClipStatistic
) -> LocalEnergy:
    e_loc = clip_local_energies(e_loc, clip_local_energy, stat)
    center = jnp.mean(e_loc, keepdims=True)
    e_loc -= center
    return e_loc


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
    clip_local_energy: float = field(pytree_node=False)
    clip_statistic: ClipStatistic | str = field(pytree_node=False)

    def init(self, key: Array):
        demo_system = Systems(
            ((2, 1),),
            ((3,),),
            jax.random.uniform(key, (3, 3), dtype=jnp.float32),
            jnp.zeros((1, 3), dtype=jnp.float32),
            {},
        )
        params = self.wave_function.init(key, demo_system)
        return VMCState(
            params=params,
            optimizer=self.optimizer.init(params),  # type: ignore
            preconditioner=self.preconditioner.init(params),
            step=jnp.zeros((), dtype=jnp.int32),
        )

    def init_systems_data(self, key: Array, systems: Systems):
        @shmap(
            in_specs=(REPLICATE_SHARD, systems.sharding),
            out_specs=systems.sharding,
        )
        def init(key: Array, systems: Systems):
            key = distribute_keys(key)
            key, subkey = jax.random.split(key)
            systems = self.sampler.init(subkey, systems)
            return systems

        return init(key, systems)

    def mcmc_step(self, key: Array, state: VMCState[PS], systems: Systems):
        return self.sampler(key, state.params, systems)

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
            systems = self.mcmc_step(subkey, state, systems)

            # Local energy
            e_l = self.local_energy(state, systems)
            dE_dlogpsi = local_energy_diff(
                e_l, self.clip_local_energy, ClipStatistic(self.clip_statistic)
            )

            # Preconditioning
            gradient, preconditioner_state, aux_data = self.preconditioner.apply(
                state.params, systems, dE_dlogpsi, state.preconditioner
            )

            # Update step
            updates, opt_state = self.optimizer.update(gradient, state.optimizer)  # type: ignore
            params = optax.apply_updates(state.params, updates)  # type: ignore

            # Logging
            E = pmean(e_l.mean())
            E_std = (pmean(e_l.var(0)) ** 0.5).mean()
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
