import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest

from neural_pfaffian.clipping import MedianClipping
from neural_pfaffian.mcmc import MetropolisHastings
from neural_pfaffian.nn.antisymmetrizer import Pfaffian, Slater
from neural_pfaffian.nn.embedding import FermiNet, Moon, PsiFormer
from neural_pfaffian.nn.embedding.psiformer import AttentionImplementation
from neural_pfaffian.nn.envelope import EfficientEnvelope, FullEnvelope
from neural_pfaffian.nn.jastrow import CuspJastrow, MLPJastrow
from neural_pfaffian.nn.meta_network import MetaGNN
from neural_pfaffian.nn.module import ParamMeta, ParamTypes
from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    WaveFunction,
)
from neural_pfaffian.preconditioner import CG, Identity, Preconditioner, Spring
from neural_pfaffian.pretraining import Pretraining
from neural_pfaffian.sample_reweighting import LOG_NORMALIZER_CONSTANTS_KEY
from neural_pfaffian.systems import PseudopotentialProperties, Systems
from neural_pfaffian.vmc import VMC

# ---------------------------------------------------------------------------
# Plain helper functions (importable by test files)
# ---------------------------------------------------------------------------


def _systems_with_default_pp(
    *,
    spins,
    charges,
    electrons,
    nuclei,
    mol_data,
    mol_ids,
    excitations,
):
    return Systems(
        spins=spins,
        charges=charges,
        electrons=electrons,
        nuclei=nuclei,
        mol_data=mol_data,
        mol_ids=mol_ids,
        excitations=excitations,
        effective_charges=charges,
        pp_data=tuple(
            PseudopotentialProperties.create_empty(len(c), electrons.dtype)
            for c in charges
        ),
    )


_ORBS_PER_CHARGE = {
    '1': 2,
    '2': 2,
    '3': 5,
    '4': 5,
    '5': 5,
    '6': 5,
    '7': 5,
    '8': 5,
    '9': 5,
    '10': 5,
}


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------


@pytest.fixture(scope='session')
def one_system():
    charges = ((3,),)
    electrons = jax.random.normal(jax.random.key(0), (3, 3), dtype=jnp.float32)
    return _systems_with_default_pp(
        spins=((2, 1),),
        charges=charges,
        electrons=electrons,
        nuclei=jax.random.normal(jax.random.key(1), (1, 3), dtype=jnp.float32),
        mol_data={},
        mol_ids=(0,),
        excitations=(0,),
    )


@pytest.fixture(scope='session')
def two_systems():
    charges = ((4,), (4, 2))
    electrons = jax.random.normal(jax.random.key(0), (10, 3), dtype=jnp.float32)
    return _systems_with_default_pp(
        spins=((2, 2), (3, 3)),
        charges=charges,
        electrons=electrons,
        nuclei=jax.random.normal(jax.random.key(1), (3, 3), dtype=jnp.float32),
        mol_data={},
        mol_ids=(0, 1),
        excitations=(0, 0),
    )


@pytest.fixture(scope='session')
def excited_systems():
    charges = ((2, 2), (2, 2))
    electrons = jnp.concatenate(
        [jax.random.normal(jax.random.key(0), (4, 3), dtype=jnp.float32)] * 2,
        axis=0,
    )
    systems = _systems_with_default_pp(
        spins=((2, 2), (2, 2)),
        charges=charges,
        electrons=electrons,
        nuclei=jnp.concatenate(
            [jax.random.normal(jax.random.key(1), (2, 3), dtype=jnp.float32)] * 2,
            axis=0,
        ),
        mol_data={},
        mol_ids=(0, 0),
        excitations=(0, 1),
    )
    return systems.set_mol_data(
        LOG_NORMALIZER_CONSTANTS_KEY,
        jnp.zeros((systems.n_mols,), dtype=jnp.float64),
    )


@pytest.fixture(scope='session')
def batched_excited_systems(excited_systems):
    electrons = excited_systems.electrons
    batched_electrons = jnp.stack(
        [electrons, electrons + jnp.array(0.05, dtype=jnp.float32)],
    )
    return excited_systems.replace(electrons=batched_electrons)


@pytest.fixture(scope='session')
def two_excited_systems():
    charges = ((2, 2), (2, 2), (2, 2, 1), (2, 2, 1))
    electrons = jax.random.normal(jax.random.key(0), (18, 3), dtype=jnp.float32)
    systems = _systems_with_default_pp(
        spins=((2, 2), (2, 2), (2, 3), (2, 3)),
        charges=charges,
        electrons=electrons,
        nuclei=jax.random.normal(jax.random.key(1), (10, 3), dtype=jnp.float32),
        mol_data={},
        mol_ids=(0, 0, 1, 1),
        excitations=(0, 1, 0, 1),
    )
    return systems.set_mol_data(
        LOG_NORMALIZER_CONSTANTS_KEY,
        jnp.zeros((systems.n_mols,), dtype=jnp.float64),
    )


@pytest.fixture(scope='session')
def three_state_system():
    charges = ((2, 2),) * 3
    electrons = jnp.concatenate(
        [jax.random.normal(jax.random.key(2), (4, 3), dtype=jnp.float32)] * 3,
        axis=0,
    )
    return _systems_with_default_pp(
        spins=((2, 2),) * 3,
        charges=charges,
        electrons=electrons,
        nuclei=jnp.concatenate(
            [jax.random.normal(jax.random.key(3), (2, 3), dtype=jnp.float32)] * 3,
            axis=0,
        ),
        mol_data={},
        mol_ids=(0, 0, 0),
        excitations=(0, 1, 2),
    )


@pytest.fixture(scope='session')
def li_all_electron_system():
    return Systems.create(
        (2, 1),
        (3,),
        jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
    ).replace(
        electrons=jnp.array(
            [[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.4]],
            dtype=jnp.float32,
        ),
    )


@pytest.fixture(scope='session')
def li_pseudopotential_system():
    from neural_pfaffian.pseudopotential import attach_pseudopotentials

    systems = Systems.create(
        (2, 1),
        (3,),
        jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
    )
    systems = attach_pseudopotentials(systems, enable=True, ecp='ccecp', symbols=['Li'])
    return systems.replace(electrons=jnp.array([[0.2, 0.0, 0.0]], dtype=jnp.float32))


@pytest.fixture(
    scope='session',
    params=['one_system', 'two_systems', 'excited_systems', 'two_excited_systems'],
)
def systems(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='session')
def batched_systems(systems):
    return systems.replace(
        electrons=jax.random.normal(
            jax.random.key(0),
            (2 * jax.device_count(), *systems.electrons.shape),
            dtype=systems.electrons.dtype,
        ),
    )


@pytest.fixture(scope='session')
def systems_float64(systems):
    return systems.replace(
        electrons=systems.electrons.astype(jnp.float64),
        nuclei=systems.nuclei.astype(jnp.float64),
    )


# Embedding
@pytest.fixture(scope='module')
def ferminet():
    return FermiNet(
        embedding_dim=4,
        hidden_dims=[(4, 2), (4, 2)],
        activation=jnp.tanh,
    )


@pytest.fixture(scope='module')
def psiformer_iterative():
    return PsiFormer(
        embedding_dim=4,
        dim=4,
        n_head=2,
        n_layer=1,
        activation=jnp.tanh,
        attention_implementation=AttentionImplementation.ITERATIVE,
    )


@pytest.fixture(scope='module')
def psiformer_parallel():
    return PsiFormer(
        embedding_dim=4,
        dim=4,
        n_head=2,
        n_layer=1,
        activation=jnp.tanh,
        attention_implementation=AttentionImplementation.PARALLEL,
    )


@pytest.fixture(scope='session')
def moon():
    return Moon(
        embedding_dim=4,
        dim=4,
        n_layer=1,
        edge_embedding=4,
        edge_hidden_dim=2,
        edge_rbf=2,
        activation=jnp.tanh,
    )


@pytest.fixture(
    scope='module',
    params=['ferminet', 'psiformer_iterative', 'psiformer_parallel', 'moon'],
)
def embedding_model(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def embedding_params(embedding_model: nn.Module, systems: Systems):
    params = embedding_model.init(jax.random.key(42), systems)
    return params


@pytest.fixture(scope='module')
def embedding_fwdpass(embedding_model: nn.Module):
    return jax.jit(embedding_model.apply)


# Envelopes
@pytest.fixture(scope='module')
def full_envelope():
    return FullEnvelope()


@pytest.fixture(scope='session')
def efficient_envelope():
    return EfficientEnvelope(2)


@pytest.fixture(scope='module', params=['full_envelope', 'efficient_envelope'])
def envelope(request):
    return request.getfixturevalue(request.param)


# Antisymmetrizer
@pytest.fixture(scope='module')
def pfaffian(envelope):
    return Pfaffian(2, _ORBS_PER_CHARGE, envelope, 1.0, 1.0, 0.0)


@pytest.fixture(scope='module')
def slater(envelope):
    return Slater(2, envelope)


@pytest.fixture(scope='module', params=['pfaffian', 'slater'])
def antisymmetrizer(request, envelope):
    # envelope must be here since orbitals depend on it
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='session')
def singular_pfaffian(efficient_envelope):
    return Pfaffian(
        2,
        _ORBS_PER_CHARGE,
        efficient_envelope,
        1.0,
        1.0,
        0.0,
    )


@pytest.fixture(scope='module')
def singular_slater(efficient_envelope):
    return Slater(2, efficient_envelope)


@pytest.fixture(
    scope='module',
    params=[
        'singular_pfaffian',
        'singular_slater',
    ],
)
def singular_antisymmetrizer(request, efficient_envelope):
    # envelope must be here since orbitals depend on it
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def no_jastrow():
    return []


@pytest.fixture(scope='session')
def mlp_jastrow():
    return [MLPJastrow([8, 4], jnp.tanh)]


@pytest.fixture(scope='session')
def cusp_jastrow():
    return [CuspJastrow()]


@pytest.fixture(scope='session')
def double_jastrow(mlp_jastrow, cusp_jastrow):
    return mlp_jastrow + cusp_jastrow


# Jastrows
@pytest.fixture(
    scope='module',
    params=['no_jastrow', 'mlp_jastrow', 'cusp_jastrow', 'double_jastrow'],
)
def jastrow_models(request):
    return request.getfixturevalue(request.param)


# Wave Function
@pytest.fixture(scope='module')
def wave_function(embedding_model, antisymmetrizer, jastrow_models, systems):
    if isinstance(antisymmetrizer, Slater) and systems.max_num_states > 1:
        pytest.skip('Slater determinants do not support excitation')
    wf = WaveFunction(embedding_model, antisymmetrizer, jastrow_models)
    wf.antisymmetrizer.max_num_states = systems.max_num_states
    return wf


@pytest.fixture(scope='module')
def wf_params(wave_function: WaveFunction, systems: Systems):
    if isinstance(wave_function.antisymmetrizer, Slater) and len(set(systems.spins)) > 1:
        pytest.skip('Slater requires identical spins for all molecules')
    return wave_function.init(jax.random.key(42), systems)


@pytest.fixture(scope='module')
def wf_signed(wave_function: WaveFunction):
    return jax.jit(wave_function.signed)


@pytest.fixture(scope='module')
def wf_apply(wave_function: WaveFunction):
    return jax.jit(wave_function.apply)


# MetaGNN
@pytest.fixture(scope='session')
def meta_gnn():
    return MetaGNN(
        out_structure=None,
        message_dim=4,
        embedding_dim=8,
        num_layers=1,
        activation=jnp.tanh,
        n_rbf=4,
        charges=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    )


@pytest.fixture(scope='module')
def global_meta():
    return ParamMeta(
        param_type=ParamTypes.GLOBAL,
        shape_and_dtype=jax.ShapeDtypeStruct((4,), jnp.float32),
        mean=0,
        std=1,
        bias=True,
        param_sharing_axis=None,
        keep_distr=False,
    )


@pytest.fixture(scope='module')
def nuclei_meta():
    return ParamMeta(
        param_type=ParamTypes.NUCLEI,
        shape_and_dtype=jax.ShapeDtypeStruct((3,), jnp.float32),
        mean=0,
        std=1,
        bias=True,
        param_sharing_axis=None,
        keep_distr=False,
    )


@pytest.fixture(scope='module')
def nuclei_nuclei_meta():
    return ParamMeta(
        param_type=ParamTypes.NUCLEI_NUCLEI,
        shape_and_dtype=jax.ShapeDtypeStruct((5,), jnp.float32),
        mean=0,
        std=1,
        bias=False,
        param_sharing_axis=None,
        keep_distr=False,
    )


@pytest.fixture(scope='module')
def param_sharing_meta():
    return ParamMeta(
        param_type=ParamTypes.NUCLEI,
        shape_and_dtype=jax.ShapeDtypeStruct((3, 6), jnp.float32),
        mean=0,
        std=1,
        bias=True,
        param_sharing_axis=0,
        keep_distr=False,
    )


@pytest.fixture(
    scope='module',
    params=['global_meta', 'nuclei_meta', 'nuclei_nuclei_meta', 'param_sharing_meta'],
)
def out_meta(request):
    return request.getfixturevalue(request.param)


# Generalized Wave function
@pytest.fixture(scope='module')
def generalized_wf(moon, pfaffian, double_jastrow, meta_gnn, excited_systems):
    return GeneralizedWaveFunction.create(
        WaveFunction(moon, pfaffian, double_jastrow),
        meta_gnn,
        excited_systems,
    )


@pytest.fixture(scope='module')
def generalized_wf_params(
    generalized_wf: GeneralizedWaveFunction,
    excited_systems: Systems,
):
    return generalized_wf.init(jax.random.key(42), excited_systems)


# Example WF
@pytest.fixture(scope='session')
def neural_pfaffian(singular_pfaffian, moon, double_jastrow, meta_gnn, excited_systems):
    return GeneralizedWaveFunction.create(
        WaveFunction(moon, singular_pfaffian, double_jastrow),
        meta_gnn,
        excited_systems,
    )


@pytest.fixture(scope='module')
def neural_pfaffian_params(
    neural_pfaffian: GeneralizedWaveFunction,
    excited_systems: Systems,
):
    return neural_pfaffian.init(jax.random.key(42), excited_systems)


@pytest.fixture(scope='module')
def identity_preconditioner(neural_pfaffian: GeneralizedWaveFunction):
    return Identity(neural_pfaffian)


@pytest.fixture(scope='module')
def spring_preconditioner(neural_pfaffian: GeneralizedWaveFunction):
    return Spring(neural_pfaffian, 1e-3, 0.99, 0.0, 0.0, 1e-3, jnp.float64, 1e-6)


@pytest.fixture(scope='module')
def cg_preconditioner(neural_pfaffian: GeneralizedWaveFunction):
    return CG(neural_pfaffian, 1e-3, 0.99, 10)


@pytest.fixture(
    scope='module',
    params=['identity_preconditioner', 'spring_preconditioner', 'cg_preconditioner'],
)
def preconditioner(request) -> Preconditioner:
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def mcmc(neural_pfaffian: GeneralizedWaveFunction):
    return MetropolisHastings(neural_pfaffian, 5, jnp.array(0.01), 2, 0.5, 0.025, 1, 0, 0)


@pytest.fixture(scope='module')
def block_mcmc(neural_pfaffian: GeneralizedWaveFunction):
    return MetropolisHastings(neural_pfaffian, 5, jnp.array(0.01), 2, 0.5, 0.025, 3, 0, 0)


@pytest.fixture(scope='module')
def nonlocal_mcmc(neural_pfaffian: GeneralizedWaveFunction):
    return MetropolisHastings(
        neural_pfaffian,
        1,
        jnp.array(0.01),
        2,
        0.5,
        0.025,
        1,
        10,
        2.0,
    )


@pytest.fixture(scope='module', params=['mcmc', 'block_mcmc', 'nonlocal_mcmc'])
def mcmcs(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def optimizer():
    return optax.adam(1e-4)


@pytest.fixture(scope='module')
def vmc(neural_pfaffian, identity_preconditioner, mcmc, optimizer):
    return VMC(
        wave_function=neural_pfaffian,
        preconditioner=identity_preconditioner,
        optimizer=optimizer,
        sampler=mcmc,
        clipping=MedianClipping(5),
    )


@pytest.fixture(scope='module')
def vmc_state(vmc: VMC, one_system):
    return vmc.init(jax.random.key(0), one_system)


@pytest.fixture(scope='module')
def vmc_systems(vmc: VMC, batched_systems: Systems):
    return vmc.init_systems(jax.random.key(7), batched_systems)


@pytest.fixture(scope='module')
def pretrain_wf(singular_antisymmetrizer, moon, double_jastrow, meta_gnn, systems):
    if isinstance(singular_antisymmetrizer, Slater):
        if len(set(systems.spins)) > 1:
            pytest.skip('Slater requires identical spins for all molecules')
        if systems.max_num_states > 1:
            pytest.skip('Slater determinants do not support excitation')
    return GeneralizedWaveFunction.create(
        WaveFunction(moon, singular_antisymmetrizer, double_jastrow),
        meta_gnn,
        systems,
    )


@pytest.fixture(scope='module')
def pretrain_vmc(pretrain_wf, spring_preconditioner, mcmc, optimizer):
    return VMC(
        wave_function=pretrain_wf,
        preconditioner=spring_preconditioner,
        optimizer=optimizer,
        sampler=mcmc.replace(wave_function=pretrain_wf),
        clipping=MedianClipping(5),
    )


@pytest.fixture(scope='module')
def pretrain_vmc_state(pretrain_vmc: VMC, systems):
    return pretrain_vmc.init(jax.random.key(0), systems)


@pytest.fixture(scope='module')
def pretrainer(pretrain_vmc, optimizer):
    pretrainer = Pretraining(pretrain_vmc, optimizer, 1e-6)
    return pretrainer


@pytest.fixture(scope='module')
def systems_with_hf(batched_systems):
    return batched_systems.with_hf('aug-cc-pVDZ')


@pytest.fixture(scope='module')
def pretrainer_state(pretrainer, pretrain_vmc_state):
    return pretrainer.init(pretrain_vmc_state)


@pytest.fixture(scope='module')
def pretraining_systems(pretrainer, systems_with_hf):
    return pretrainer.init_systems(jax.random.key(8), systems_with_hf)


# -- Full neural pfaffian setup (used by test_regression_excited_states) --


@pytest.fixture(scope='module')
def regression_wf_params(regression_wf, regression_systems):
    return regression_wf.init(jax.random.key(42), regression_systems.example_input)
