import time

import jax
import optax
import tqdm.auto as tqdm

from neural_pfaffian.logger import Logger
from neural_pfaffian.pretraining import Pretraining, PretrainingDistribution
from neural_pfaffian.systems import Systems, SystemsWithHF
from neural_pfaffian.utils import batch
from neural_pfaffian.vmc import VMC, VMCState


def thermalize(
    key: jax.Array,
    vmc: VMC,
    state: VMCState,
    systems: Systems,
    n_epochs: int,
    batch_size: int,
    logger: Logger,
):
    batches = list(map(Systems.merge, batch(systems, batch_size)))
    # Initialize batches
    key, subkey = jax.random.split(key)
    batch_keys = jax.random.split(subkey, len(batches))
    batches = list(map(vmc.init_systems, batch_keys, batches))

    for _ in tqdm.trange(n_epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            batches[i], aux_data = vmc.mcmc_step(subkey, state, batches[i])
            logger.log(aux_data, prefix='mcmc')
    return Systems.merge(batches)


def pretrain(
    key: jax.Array,
    vmc: VMC,
    state: VMCState,
    systems: Systems,
    optimizer: optax.GradientTransformation,
    reparam_loss_scale: float,
    sample_from: PretrainingDistribution | str,
    epochs: int,
    batch_size: int,
    basis: str,
    logger: Logger,
):
    pretrainer = Pretraining(
        vmc, optimizer, reparam_loss_scale, PretrainingDistribution(sample_from)
    )
    pre_state = pretrainer.init(state)

    # Initialize batches
    batches: list[SystemsWithHF] = []
    for b in map(Systems.merge, batch(systems, batch_size)):
        key, subkey = jax.random.split(key)
        batches.append(pretrainer.init_systems(subkey, b.with_hf(basis)))

    last_time = time.perf_counter()
    for epoch in tqdm.trange(epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            # Update step
            pre_state, batches[i], log_data = pretrainer.step(
                subkey, pre_state, batches[i]
            )
            # Logging
            log_data = jax.tree.map(lambda x: x.item(), log_data)
            log_data['time_step'] = time.perf_counter() - last_time
            logger.log(log_data, prefix='pretrain')
            last_time = time.perf_counter()
    return pre_state.vmc_state, Systems.merge(batches).to_systems


def train(
    key: jax.Array,
    vmc: VMC,
    state: VMCState,
    systems: Systems,
    epochs: int,
    batch_size: int,
    logger: Logger,
):
    # Init systems
    key, subkey = jax.random.split(key)
    systems = vmc.init_systems(subkey, systems)
    # Initialize batches
    batches = list(map(Systems.merge, batch(systems, batch_size)))
    key, subkey = jax.random.split(key)
    batch_keys = jax.random.split(subkey, len(batches))
    batches = list(map(vmc.init_systems, batch_keys, batches))

    last_time = time.perf_counter()
    for epoch in tqdm.trange(epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            # Update step
            state, batches[i], log_data = vmc.step(subkey, state, batches[i])
            # Logging
            log_data = jax.tree.map(lambda x: x.item(), log_data)
            log_data['time_step'] = time.perf_counter() - last_time
            logger.log(log_data, prefix='train')
            last_time = time.perf_counter()
        if epoch % 100 == 0:
            logger.checkpoint(state, Systems.merge(batches))
    return state, Systems.merge(batches)
