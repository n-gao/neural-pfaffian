import logging
import os
from pathlib import Path

os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'

import jax
import seml

import wandb
from neural_pfaffian.nn import (
    ANTISYMMETRIZERS,
    EMBEDDINGS,
    ENVELOPES,
    JASTROWS,
    GeneralizedWaveFunction,
    MetaGNN,
    WaveFunction,
)
from neural_pfaffian.clipping import CLIPPINGS
from neural_pfaffian.dataset import create_systems
from neural_pfaffian.mcmc import MetroplisHastings
from neural_pfaffian.preconditioner import PRECONDITIONER
from neural_pfaffian.train import pretrain, thermalize, train
from neural_pfaffian.utils.optim import make_optimizer
from neural_pfaffian.vmc import VMC

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'float32')

ex = seml.Experiment()

root_dir = Path(__file__).parent
ex.add_config(str(root_dir / 'config' / 'default.yaml'))


def main(
    seed,
    vmc_config,
    wave_function_config,
    pretraining_config,
    systems_config,
    logging_config,
):
    # Proper main file
    config = locals()
    logging.info(f'Running with config:\n{config}')
    key = jax.random.key(seed)

    logging.info('Creating systems')
    key, subkey = jax.random.split(key)
    systems = create_systems(subkey, **systems_config)

    # Initialize the wave function
    logging.info('Initializing wave function')
    wave_function = GeneralizedWaveFunction.create(
        WaveFunction(
            EMBEDDINGS.init(**wave_function_config['embedding']),
            ANTISYMMETRIZERS.init(
                **wave_function_config['orbitals'],
                envelope=ENVELOPES.init(**wave_function_config['envelope']),
            ),
            JASTROWS.init_many(wave_function_config['jastrows']),
        ),
        MetaGNN(**wave_function_config['meta_gnn']),
        systems,
    )

    # Initialize VMC object
    logging.info('Initializing VMC')
    preconditioner = PRECONDITIONER.init(
        **vmc_config['preconditioner'], wave_function=wave_function
    )
    optimizer = make_optimizer(**vmc_config['optimizer'])
    mcmc = MetroplisHastings(wave_function, **vmc_config['mcmc'])
    clipping = CLIPPINGS.init(**vmc_config['clipping'])
    vmc = VMC(wave_function, preconditioner, optimizer, mcmc, clipping)

    # init state
    logging.info('Initializing VMC state')
    key, subkey = jax.random.split(key)
    state = vmc.init(subkey, systems)

    # Init wandb
    wandb.init(**logging_config['wandb'])
    wandb.config.update(config)
    # Pretraining
    logging.info('Pretraining')
    key, subkey = jax.random.split(key)
    state, systems = pretrain(
        subkey,
        vmc,
        state,
        systems,
        make_optimizer(**pretraining_config['optimizer']),
        reparam_loss_scale=pretraining_config['reparam_loss_scale'],
        epochs=pretraining_config['epochs'],
        batch_size=pretraining_config['batch_size'],
        basis=pretraining_config['basis'],
    )

    # Thermalizing
    logging.info('Thermalizing')
    key, subkey = jax.random.split(key)
    systems = thermalize(
        subkey,
        vmc,
        state,
        systems,
        n_epochs=vmc_config['thermalizing_epochs'],
        batch_size=vmc_config['batch_size'],
    )

    # VMC Training
    logging.info('VMC')
    key, subkey = jax.random.split(key)
    state, systems = train(
        subkey,
        vmc,
        state,
        systems,
        epochs=vmc_config['epochs'],
        batch_size=vmc_config['batch_size'],
    )

    logging.info('Done')
    return


@ex.automain
def _main(seed, vmc, wave_function, pretraining, systems, logging):
    # A wrapper to have simpler yaml keys
    return main(seed, vmc, wave_function, pretraining, systems, logging)


def cli_main():
    ex.run_commandline()
