import jax
import optax
import tqdm.auto as tqdm

import wandb
from neural_pfaffian.pretraining import Pretraining
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
):
    batches = list(map(Systems.merge, batch(systems, batch_size)))
    # Initialize batches
    key, subkey = jax.random.split(key)
    batch_keys = jax.random.split(subkey, len(batches))
    batches = list(map(vmc.init_systems, batch_keys, batches))

    for _ in tqdm.trange(n_epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            batches[i] = vmc.mcmc_step(subkey, state, batches[i])
    return Systems.merge(batches)


def pretrain(
    key: jax.Array,
    vmc: VMC,
    state: VMCState,
    systems: Systems,
    n_epochs: int,
    batch_size: int,
    basis: str,
):
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_trust_ratio(),
        optax.scale_by_schedule(lambda t: -1e-3 / (1 + 1e-4 * t)),
    )
    pretrainer = Pretraining(vmc, optimizer, 1e-6)
    pre_state = pretrainer.init(state)

    # Initialize batches
    batches: list[SystemsWithHF] = []
    for b in map(Systems.merge, batch(systems, batch_size)):
        key, subkey = jax.random.split(key)
        batches.append(pretrainer.init_systems(subkey, b.with_hf(basis)))

    step = 0
    for epoch in tqdm.trange(n_epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            # Update step
            pre_state, batches[i], log_data = pretrainer.step(
                subkey, pre_state, batches[i]
            )
            # Logging
            log_data = jax.tree.map(lambda x: x.item(), log_data)
            wandb.log({**log_data, 'pretrain_step': step})
            step += 1
    return pre_state.vmc_state, Systems.merge(batches).to_systems


def train(
    key: jax.Array,
    vmc: VMC,
    state: VMCState,
    systems: Systems,
    n_epochs: int,
    batch_size: int,
):
    batches = list(map(Systems.merge, batch(systems, batch_size)))
    # Initialize batches
    key, subkey = jax.random.split(key)
    batch_keys = jax.random.split(subkey, len(batches))
    batches = list(map(vmc.init_systems, batch_keys, batches))

    step = 0
    for epoch in tqdm.trange(n_epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            # Update step
            state, batches[i], log_data = vmc.step(subkey, state, batches[i])
            # Logging
            log_data = jax.tree.map(lambda x: x.item(), log_data)
            wandb.log({**log_data, 'train_step': step})
            step += 1
    return state, Systems.merge(batches)
