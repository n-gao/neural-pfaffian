import logging
import time

import jax
import jax.numpy as jnp
import optax
import tqdm.auto as tqdm

from neural_pfaffian.logger import Logger
from neural_pfaffian.pretraining import Pretraining
from neural_pfaffian.systems import Systems, SystemsWithPretrainTarget
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
    key, subkey = jax.random.split(key)
    systems = vmc.init_systems(subkey, systems)
    # Batch after initialization
    batches = list(map(Systems.merge, batch(systems, batch_size)))

    for _ in tqdm.trange(n_epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            batches[i], mcmc_aux = vmc.mcmc_step(
                subkey,
                state.sharded,
                batches[i].sharded,
            )
            log_data = jax.tree.map(lambda x: x.item(), mcmc_aux)
            logger.log(log_data, prefix='thermalize/mcmc')
    return Systems.merge(batches)


def pretrain(
    key: jax.Array,
    vmc: VMC,
    state: VMCState,
    systems: Systems,
    optimizer: optax.GradientTransformation,
    reparam_loss_scale: float,
    epochs: int,
    batch_size: int,
    basis: str,
    hf_config: dict,
    logger: Logger,
):
    pretrainer = Pretraining(vmc, optimizer, reparam_loss_scale)
    pre_state = pretrainer.init(state)

    # Initialize batches
    batches: list[SystemsWithPretrainTarget] = []
    for b in map(Systems.merge, batch(systems, batch_size)):
        key, subkey = jax.random.split(key)
        batches.append(pretrainer.init_systems(subkey, b.with_hf(basis, **hf_config)))

    last_time = time.perf_counter()
    step = 0
    for _epoch in tqdm.trange(epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            # Update step
            pre_state, batches[i], log_data = pretrainer.step(
                subkey,
                pre_state.sharded,
                batches[i].sharded,
            )
            # Logging
            log_data = jax.tree.map(lambda x: x.item(), log_data)
            log_data['time_step'] = time.perf_counter() - last_time
            log_data['step'] = step
            step += 1
            logger.log(log_data, prefix='pretrain')
            last_time = time.perf_counter()
    return pre_state.vmc_state, SystemsWithPretrainTarget.merge(batches).to_systems


def train(
    key: jax.Array,
    vmc: VMC,
    state: VMCState,
    systems: Systems,
    epochs: int,
    batch_size: int,
    logger: Logger,
    *,
    continue_training: bool,
):
    # Init systems
    key, subkey = jax.random.split(key)
    systems = vmc.init_systems(subkey, systems)
    num_walker_per_mol = systems.electrons.shape[0]
    if continue_training:
        logging.info('Loading checkpoint and resuming training')
        state, systems = logger.load_checkpoint(state, systems)

    #  In case of restarting from a pretrained state after an OOM
    # we may need to update the systems batch size
    key, subkey = jax.random.split(key)
    systems = systems.update_batch_size(subkey, num_walker_per_mol)

    epoch = int(state.epoch)
    if epoch >= epochs:
        logging.info('Training already done, skipping')
        return state, systems

    batches = [Systems.merge(batch) for batch in Systems.safe_batch(systems, batch_size)]

    last_time = time.perf_counter()
    epoch_bar = tqdm.tqdm(total=epochs, initial=epoch)
    while epoch < epochs:
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            state, batches[i], raw_log_data = vmc.step(
                subkey,
                state.sharded,
                batches[i].sharded,
            )
            log_data = jax.tree.map(lambda x: x.item(), raw_log_data)
            current_step_value = int(state.step.item())

            log_data['time_step'] = time.perf_counter() - last_time
            log_data['epoch'] = epoch
            log_data['step'] = current_step_value + 1
            logger.log(log_data, prefix='train')

            state = state.replace(
                step=jnp.array(state.step + 1, dtype=state.step.dtype),
            )

            last_time = time.perf_counter()

        if epoch % 100 == 0:
            logger.checkpoint(state, Systems.merge(batches))

        epoch += 1
        epoch_bar.update(1)
        state = state.replace(epoch=jnp.array(epoch, dtype=state.epoch.dtype))

    return state, Systems.merge(batches)
