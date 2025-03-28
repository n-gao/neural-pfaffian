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
    key, subkey = jax.random.split(key)
    systems = vmc.init_systems(subkey, systems)
    # Batch
    batches = list(map(Systems.merge, batch(systems, batch_size)))

    for _ in tqdm.trange(n_epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            batches[i], log_data = vmc.mcmc_step(
                subkey, state.sharded, batches[i].sharded
            )
            logger.log(log_data, prefix='mcmc')
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

    for epoch in tqdm.trange(epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            # Update step
            pre_state, batches[i], log_data = pretrainer.step(
                subkey, pre_state.sharded, batches[i].sharded
            )
            logger.log(log_data, prefix='pretrain')
    return pre_state.vmc_state, SystemsWithHF.merge(batches).without_hf


def train(
    key: jax.Array,
    vmc: VMC,
    state: VMCState,
    systems: Systems,
    epochs: int,
    batch_size: int,
    logger: Logger,
):
    key, subkey = jax.random.split(key)
    systems = vmc.init_systems(subkey, systems)
    # Batch
    batches = list(map(Systems.merge, batch(systems, batch_size)))

    for epoch in tqdm.trange(epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            # Update step
            state, batches[i], log_data = vmc.step(
                subkey, state.sharded, batches[i].sharded
            )
            logger.log(log_data, prefix='train')
        if epoch % 100 == 0:
            logger.checkpoint(state, Systems.merge(batches))
    return state, Systems.merge(batches)
