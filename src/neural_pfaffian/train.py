from typing import Sequence

import jax
import tqdm.auto as tqdm

import wandb
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils import batch
from neural_pfaffian.vmc import VMC, VMCState


def thermalize(
    key: jax.Array,
    vmc: VMC,
    state: VMCState,
    systems: Sequence[Systems],
    n_epochs: int,
    batch_size: int,
):
    batches = list(map(Systems.merge, batch(systems, batch_size)))
    # Initialize batches
    key, subkey = jax.random.split(key)
    batch_keys = jax.random.split(subkey, len(batches))
    batches = list(map(vmc.init_systems_data, batch_keys, batches))

    for _ in tqdm.trange(n_epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            batches[i] = vmc.mcmc_step(subkey, state, batches[i])

    systems = [sys for batch in batches for sys in batch.sub_configs]
    return systems


def train(
    key: jax.Array,
    vmc: VMC,
    state: VMCState,
    systems: Sequence[Systems],
    n_epochs: int,
    batch_size: int,
):
    batches = list(map(Systems.merge, batch(systems, batch_size)))
    # Initialize batches
    key, subkey = jax.random.split(key)
    batch_keys = jax.random.split(subkey, len(batches))
    batches = list(map(vmc.init_systems_data, batch_keys, batches))

    for epoch in tqdm.trange(n_epochs):
        for i in range(len(batches)):
            key, subkey = jax.random.split(key)
            # Update step
            state, batches[i], log_data = vmc.step(subkey, state, batches[i])
            log_data = jax.tree.map(lambda x: x.item(), log_data)
            # Logging
            wandb.log(log_data)

    systems = [sys for batch in batches for sys in batch.sub_configs]
    return state, systems
