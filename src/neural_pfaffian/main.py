import os

os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'
import logging
from copy import deepcopy

import jax
import numpy as np
import rich.syntax
import seml
import yaml
from seml.utils.yaml import YamlDumper

import wandb
from neural_pfaffian.clipping import CLIPPINGS, MASKINGS
from neural_pfaffian.config import DEFAULT_CONFIG
from neural_pfaffian.dataset import create_systems
from neural_pfaffian.logger import Logger
from neural_pfaffian.mcmc import MetropolisHastings
from neural_pfaffian.nn import (
    ANTISYMMETRIZERS,
    EMBEDDINGS,
    ENVELOPES,
    JASTROWS,
    META_NETWORKS,
    GeneralizedWaveFunction,
    WaveFunction,
)
from neural_pfaffian.nn.wave_function import MixtureLogAmplitude
from neural_pfaffian.overlap import OverlapPenalty
from neural_pfaffian.overlap_scaler import OVERLAP_SCALER
from neural_pfaffian.preconditioner import PRECONDITIONER
from neural_pfaffian.spin_operator import SpinPenalty
from neural_pfaffian.train import pretrain, thermalize, train
from neural_pfaffian.utils.optim import make_optimizer
from neural_pfaffian.vmc import VMC

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'float32')

ex = seml.Experiment()
ex.add_config(DEFAULT_CONFIG)


def main(
    seed,
    vmc_config,
    wave_function_config,
    pretraining_config,
    systems_config,
    logging_config,
):
    # Proper main file
    mutable_config = get_config()  # type: ignore
    logging.info('Running with config:')
    cfg_str = yaml.dump(
        mutable_config,
        indent=2,
        default_flow_style=None,
        Dumper=YamlDumper,
    )
    rich.print(rich.syntax.Syntax(cfg_str.strip(), 'yaml', background_color='default'))
    key = jax.random.key(seed)
    np.random.seed(seed)

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
        META_NETWORKS.init_or_none(**wave_function_config['meta_network']),
        systems,
    )

    # Initialize VMC object
    logging.info('Initializing VMC')
    preconditioner = PRECONDITIONER.init(
        **vmc_config['preconditioner'],
        wave_function=wave_function,
    )
    optimizer = make_optimizer(vmc_config['optimizer'])
    mcmc = MetropolisHastings(wave_function, **vmc_config['mcmc'])
    clipping = CLIPPINGS.init(**vmc_config['clipping'])
    overlap_penalty = None
    if systems.max_num_states > 1:
        logging.info('Found excitations, initializing overlap penalty')
        overlap_config = vmc_config['state_overlap']
        overlap_clipping = CLIPPINGS.init(**overlap_config['clipping'])
        overlap_masking = MASKINGS.init(**overlap_config['masking'])
        overlap_penalty = OverlapPenalty(
            wave_function,
            overlap_clipping,
            OVERLAP_SCALER.init(**overlap_config['scaler']),
            penalty_scale=overlap_config['penalty_scale'],
            dtype=overlap_config['dtype'],
            masking=overlap_masking,
        )
    spin_penalty_config = vmc_config.get('spin_penalty')
    spin_penalty = None
    if (spin_penalty_config or {}).get('enabled', False):
        logging.info('Found spin penalty, initializing spin penalty')
        spin_penalty = SpinPenalty.create(
            wave_function=wave_function,
            sample_masking=MASKINGS.init(**spin_penalty_config['masking']),
            ratio_clipping=CLIPPINGS.init(**spin_penalty_config['clipping']),
            penalty_scale=spin_penalty_config['penalty_scale'],
            max_grad_norm=spin_penalty_config['max_grad_norm'],
            penalty_type=spin_penalty_config['penalty_type'],
            spin_ema_decay=spin_penalty_config['decay'],
        )
    vmc = VMC(
        wave_function,
        preconditioner,
        optimizer,
        mcmc,
        clipping,
        overlap_penalty,
        spin_penalty,
        reweight_overlap_mean=vmc_config['reweight_overlap_mean'],
    )

    # init state
    logging.info('Initializing VMC state')
    key, subkey = jax.random.split(key)
    state = vmc.init(subkey, systems)

    # Init wandb
    logging.info('Initializing logger')
    logger = Logger(str(systems), logging_config)
    logger.config(mutable_config)
    continue_training = logger.has_checkpoint()
    # Pretraining
    if continue_training:
        logging.info('Found checkpoint, skipping pretraining')
    elif pretraining_config.get('epochs', 0) == 0:
        logging.info('Pretraining epochs set to 0, skipping pretraining')
    else:
        logging.info('Pretraining')
        mcmc_config = pretraining_config['mcmc'].copy()
        mix_log_amp = MixtureLogAmplitude(
            wave_function,
            mcmc_config.pop('hf_fraction'),
        )
        pre_mcmc = MetropolisHastings(mix_log_amp, **mcmc_config)

        key, subkey = jax.random.split(key)
        state, systems = pretrain(
            subkey,
            vmc.replace(sampler=pre_mcmc),
            state,
            systems,
            make_optimizer(pretraining_config['optimizer']),
            reparam_loss_scale=pretraining_config['reparam_loss_scale'],
            epochs=pretraining_config['epochs'],
            batch_size=pretraining_config['batch_size'],
            basis=pretraining_config['basis'],
            hf_config=pretraining_config['hf_config'],
            logger=logger,
        )

    # Thermalizing
    if not continue_training:
        logging.info('Thermalizing')
        key, subkey = jax.random.split(key)
        systems = thermalize(
            subkey,
            vmc,
            state,
            systems,
            n_epochs=vmc_config['thermalizing_epochs'],
            batch_size=vmc_config['batch_size'],
            logger=logger,
        )
        logger.checkpoint(state, systems)

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
        logger=logger,
        continue_training=continue_training,
    )
    logger.checkpoint(state, systems)

    wandb.finish()

    logging.info('Done')
    return


@ex.capture
def get_config(seed, vmc, wave_function, pretraining, systems, logging):
    return deepcopy(locals())


@ex.automain
def _main(
    seed,
    vmc,
    wave_function,
    pretraining,
    systems,
    logging,
):
    # A wrapper to have simpler yaml keys
    return main(seed, vmc, wave_function, pretraining, systems, logging)


def cli_main():
    ex.run_commandline()
