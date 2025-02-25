from enum import Enum
from typing import Generic, TypeVar

import jax
import numpy as np
import optax
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Float, PyTree
from scipy.special import factorial2

from neural_pfaffian.nn.module import ParamMeta
from neural_pfaffian.nn.wave_function import HFWaveFunction, WaveFunctionParameters
from neural_pfaffian.systems import SystemsWithHF
from neural_pfaffian.utils.jax_utils import (
    REPLICATE_SHARD,
    distribute_keys,
    jit,
    pmean_if_pmap,
    shmap,
)
from neural_pfaffian.utils.tree_utils import tree_sum
from neural_pfaffian.vmc import VMC, VMCState

PS = TypeVar('PS')
O = TypeVar('O')
OS = TypeVar('OS')


class PretrainingDistribution(Enum):
    HF = 'hf'
    WAVE_FUNCTION = 'wave_function'


def reparam_loss(
    meta: PyTree[ParamMeta],
    reparams: PyTree[Float[Array, '...']],
    loss_scale: float,
    max_moment: int,
):
    if loss_scale <= 0:
        return 0

    p = np.arange(1, max_moment + 1)
    # all odd moments are 0
    # https://en.wikipedia.org/wiki/Normal_distribution#Moments:~:text=standard%20normal%20distribution.-,Moments,-See%20also%3A
    target_moments = 1**p * factorial2(p - 1) * (1 - p % 2)

    def loss(reparam: Float[Array, '...'], meta: ParamMeta):
        if not meta.keep_distr:
            return 0

        p_norm = (reparam - meta.mean) / (1e-6 + meta.std)  # type: ignore
        x = p_norm[..., None] ** p
        # average over all but last dim
        observed_moments = x.mean(axis=tuple(range(x.ndim - 1)))
        return ((target_moments - observed_moments) ** 2).sum()

    return loss_scale * tree_sum(jax.tree.map(loss, reparams, meta))


class PretrainingState(Generic[PS], PyTreeNode):
    vmc_state: VMCState[PS]
    pre_opt_state: optax.OptState

    @property
    def sharding(self):
        return self.replace(
            vmc_state=self.vmc_state.sharding, pre_opt_state=REPLICATE_SHARD
        )


class Pretraining(Generic[PS, O, OS], PyTreeNode):
    vmc: VMC[PS, O, OS] = field(pytree_node=False)
    optimizer: optax.GradientTransformation = field(pytree_node=False)
    reparam_loss_scale: float = field(pytree_node=False)
    sample_from: PretrainingDistribution = field(pytree_node=False)

    @property
    def mcmc(self):
        match PretrainingDistribution(self.sample_from):
            case PretrainingDistribution.HF:
                return self.vmc.sampler.replace(
                    wave_function=HFWaveFunction(), steps=1, blocks=1
                )
            case PretrainingDistribution.WAVE_FUNCTION:
                return self.vmc.sampler
            case _:
                raise ValueError(f'Unknown sampler: {self.sample_from}')

    def init(self, vmc_state: VMCState[PS]) -> PretrainingState[PS]:
        return PretrainingState(
            vmc_state=vmc_state,
            pre_opt_state=self.optimizer.init(vmc_state.params),  # type: ignore
        )

    def init_systems(self, key: jax.Array, systems: SystemsWithHF) -> SystemsWithHF:
        key, subkey = jax.random.split(key)
        systems = self.vmc.init_systems(subkey, systems)
        key, subkey = jax.random.split(key)
        systems = self.vmc.wave_function.wave_function.orbital_module.init_systems(
            subkey, systems
        )
        return systems

    @jit
    def step(self, key: jax.Array, state: PretrainingState[PS], systems: SystemsWithHF):
        @shmap(
            in_specs=(REPLICATE_SHARD, state.sharding, systems.sharding),
            out_specs=(state.sharding, systems.sharding, REPLICATE_SHARD),
            check_rep=False,
        )
        def _step(
            key: jax.Array, state: PretrainingState[PS], systems: SystemsWithHF
        ) -> tuple[PretrainingState[PS], SystemsWithHF, dict[str, jax.Array]]:
            key = distribute_keys(key)
            key, subkey = jax.random.split(key)
            systems = self.mcmc(subkey, state.vmc_state.params, systems)
            batched_orbitals = jax.vmap(
                self.vmc.wave_function.orbitals, in_axes=(None, systems.electron_vmap)
            )

            def loss(params: WaveFunctionParameters):
                orbitals = batched_orbitals(params, systems)

                orbital_loss_val, state = (
                    self.vmc.wave_function.wave_function.orbital_module.match_hf_orbitals(
                        systems, systems.hf_orbitals, orbitals, systems.cache
                    )
                )
                reparam_loss_val = reparam_loss(
                    self.vmc.wave_function.reparam_meta,
                    self.vmc.wave_function.reparams(params, systems),
                    self.reparam_loss_scale,
                    4,
                )

                loss_val = orbital_loss_val + reparam_loss_val
                return loss_val, (
                    state,
                    dict(
                        loss=loss_val,
                        orbital_loss=orbital_loss_val,
                        reparam_loss=reparam_loss_val,
                    ),
                )

            (_, (cache, log_data)), grad = pmean_if_pmap(
                jax.value_and_grad(loss, has_aux=True)(state.vmc_state.params)
            )

            updates, pre_opt_state = self.optimizer.update(
                grad,
                state.pre_opt_state,
                state.vmc_state.params,  # type: ignore
            )
            params = optax.apply_updates(state.vmc_state.params, updates)  # type: ignore

            return (
                PretrainingState(
                    vmc_state=state.vmc_state.replace(params=params),
                    pre_opt_state=pre_opt_state,
                ),
                systems.replace(cache=cache),
                log_data,
            )

        return _step(key, state, systems)
