import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest

from neural_pfaffian.clipping import MedianClipping
from neural_pfaffian.mcmc import MetropolisHastings
from neural_pfaffian.nn.antisymmetrizer.slater import RestrictedSlater
from neural_pfaffian.nn.embedding import FermiNet, PsiFormer, Moon
from neural_pfaffian.nn.antisymmetrizer import Pfaffian, Slater
from neural_pfaffian.nn.embedding.psiformer import AttentionImplementation
from neural_pfaffian.nn.envelope import EfficientEnvelope, FullEnvelope
from neural_pfaffian.nn.jastrow import CuspJastrow, MLPJastrow
from neural_pfaffian.nn.meta_network import MetaGNN
from neural_pfaffian.nn.module import ParamMeta, ParamTypes
from neural_pfaffian.nn.wave_function import GeneralizedWaveFunction, WaveFunction
from neural_pfaffian.preconditioner import CG, Identity, Preconditioner, Spring
from neural_pfaffian.pretraining import Pretraining, PretrainingDistribution
from neural_pfaffian.systems import Systems
from neural_pfaffian.vmc import VMC


# Systems
@pytest.fixture(scope='session')
def one_system():
    return Systems(
        spins=((2, 1),),
        charges=((3,),),
        electrons=jax.random.normal(jax.random.key(0), (3, 3), dtype=jnp.float32),
        nuclei=jax.random.normal(jax.random.key(1), (1, 3), dtype=jnp.float32),
        mol_data={},
    )


@pytest.fixture(scope='session')
def two_systems():
    return Systems(
        spins=((2, 2), (3, 3)),
        charges=((4,), (4, 2)),
        electrons=jax.random.normal(jax.random.key(0), (10, 3), dtype=jnp.float32),
        nuclei=jax.random.normal(jax.random.key(1), (3, 3), dtype=jnp.float32),
        mol_data={},
    )


@pytest.fixture(scope='session', params=['one_system', 'two_systems'])
def systems(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='session')
def batched_systems(systems):
    return systems.replace(
        electrons=jax.random.normal(
            jax.random.key(0),
            (2 * jax.device_count(), *systems.electrons.shape),
            dtype=systems.electrons.dtype,
        )
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
        embedding_dim=16,
        hidden_dims=[(16, 4), (16, 4)],
        activation=jnp.tanh,
    )


@pytest.fixture(scope='module')
def psiformer_iterative():
    return PsiFormer(
        embedding_dim=32,
        dim=32,
        n_head=4,
        n_layer=2,
        activation=jnp.tanh,
        attention_implementation=AttentionImplementation.ITERATIVE,
    )


@pytest.fixture(scope='module')
def psiformer_parallel():
    return PsiFormer(
        embedding_dim=32,
        dim=32,
        n_head=4,
        n_layer=2,
        activation=jnp.tanh,
        attention_implementation=AttentionImplementation.ITERATIVE,
    )


@pytest.fixture(scope='session')
def moon():
    return Moon(
        embedding_dim=32,
        dim=32,
        n_layer=2,
        edge_embedding=8,
        edge_hidden_dim=4,
        edge_rbf=4,
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
    params = embedding_model.lazy_init(jax.random.key(42), systems)
    return params


@pytest.fixture(scope='module')
def embedding_fwdpass(embedding_model: nn.Module):
    return jax.jit(embedding_model.apply)


# Enveloeps
@pytest.fixture(scope='module')
def full_envelope():
    return FullEnvelope()


@pytest.fixture(scope='session')
def efficient_envelope():
    return EfficientEnvelope(2)


@pytest.fixture(scope='module', params=['full_envelope', 'efficient_envelope'])
def envelope(request):
    return request.getfixturevalue(request.param)


# Orbitals
@pytest.fixture(scope='module')
def pfaffian(envelope):
    return Pfaffian(
        2,
        {'1': 2, '2': 2, '3': 5, '4': 5, '5': 5, '6': 5, '7': 5, '8': 5, '9': 5, '10': 5},
        envelope,
        10,
        0.1,
        1.0,
        1.0,
        0.99,
    )


@pytest.fixture(scope='module')
def slater(envelope):
    return Slater(2, envelope)


@pytest.fixture(scope='module')
def restricted_slater(envelope):
    return RestrictedSlater(2, envelope)


@pytest.fixture(scope='module', params=['pfaffian', 'slater', 'restricted_slater'])
def orbital_model(request, envelope):
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
    scope='module', params=['no_jastrow', 'mlp_jastrow', 'cusp_jastrow', 'double_jastrow']
)
def jastrow_models(request):
    return request.getfixturevalue(request.param)


# Wave Function
@pytest.fixture(scope='module')
def wave_function(moon, orbital_model, jastrow_models):
    wf = WaveFunction(moon, orbital_model, jastrow_models)
    return wf


@pytest.fixture(scope='module')
def wf_params(wave_function: WaveFunction, systems: Systems):
    if (
        isinstance(wave_function.orbital_module, (Slater, RestrictedSlater))
        and systems.n_mols > 1
    ):
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
        num_layers=2,
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
        chunk_axis=None,
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
        chunk_axis=None,
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
        chunk_axis=None,
        keep_distr=False,
    )


@pytest.fixture(scope='module')
def chunked_meta():
    return ParamMeta(
        param_type=ParamTypes.NUCLEI,
        shape_and_dtype=jax.ShapeDtypeStruct((3, 6), jnp.float32),
        mean=0,
        std=1,
        bias=True,
        chunk_axis=0,
        keep_distr=False,
    )


@pytest.fixture(
    scope='module',
    params=['global_meta', 'nuclei_meta', 'nuclei_nuclei_meta', 'chunked_meta'],
)
def out_meta(request):
    return request.getfixturevalue(request.param)


# Generalized Wave fucntion
@pytest.fixture(scope='module')
def generalized_wf(moon, pfaffian, double_jastrow, meta_gnn, one_system):
    return GeneralizedWaveFunction.create(
        WaveFunction(moon, pfaffian, double_jastrow), meta_gnn, one_system
    )


@pytest.fixture(scope='module')
def generalized_wf_params(generalized_wf: GeneralizedWaveFunction, one_system: Systems):
    return generalized_wf.init(jax.random.key(42), one_system)


# Example WF
@pytest.fixture(scope='session')
def neural_pfaffian(moon, double_jastrow, efficient_envelope, meta_gnn, one_system):
    pfaffian = Pfaffian(
        3,
        {'1': 2, '2': 2, '3': 5, '4': 5, '5': 5, '6': 5, '7': 5, '8': 5, '9': 5, '10': 5},
        efficient_envelope,
        10,
        0.1,
        1.0,
        1.0,
        0.99,
    )
    return GeneralizedWaveFunction.create(
        WaveFunction(moon, pfaffian, double_jastrow), meta_gnn, one_system
    )


@pytest.fixture(scope='session')
def neural_pfaffian_params(neural_pfaffian: GeneralizedWaveFunction, one_system: Systems):
    return neural_pfaffian.init(jax.random.key(42), one_system)


@pytest.fixture(scope='module')
def identity_preconditioner(neural_pfaffian: GeneralizedWaveFunction):
    return Identity(neural_pfaffian)


@pytest.fixture(scope='module')
def spring_preconditioner(neural_pfaffian: GeneralizedWaveFunction):
    return Spring(neural_pfaffian, 1e-3, 0.99, jnp.float64)


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
    return MetropolisHastings(neural_pfaffian, 5, jnp.array(1.0), 2, 0.5, 0.025, 1)


@pytest.fixture(scope='module')
def block_mcmc(neural_pfaffian: GeneralizedWaveFunction):
    return MetropolisHastings(neural_pfaffian, 5, jnp.array(1.0), 2, 0.5, 0.025, 3)


@pytest.fixture(scope='module', params=['mcmc', 'block_mcmc'])
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
def fixed_vmc(neural_pfaffian, spring_preconditioner, mcmc, optimizer):
    return VMC(
        wave_function=neural_pfaffian,
        preconditioner=spring_preconditioner,
        optimizer=optimizer,
        sampler=mcmc,
        clipping=MedianClipping(5),
    )


@pytest.fixture(scope='module')
def fixed_vmc_state(fixed_vmc: VMC, one_system):
    return fixed_vmc.init(jax.random.key(0), one_system)


@pytest.fixture(scope='module')
def wf_pretrainer(fixed_vmc, optimizer):
    return Pretraining(
        fixed_vmc, optimizer, 1e-6, sample_from=PretrainingDistribution.WAVE_FUNCTION
    )


@pytest.fixture(scope='module')
def hf_pretrainer(fixed_vmc, optimizer):
    return Pretraining(fixed_vmc, optimizer, 1e-6, sample_from=PretrainingDistribution.HF)


@pytest.fixture(scope='module', params=['wf_pretrainer', 'hf_pretrainer'])
def pretrainer(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def systems_with_hf(batched_systems):
    return batched_systems.with_hf('sto-6g')


@pytest.fixture(scope='module')
def pretrainer_state(pretrainer, fixed_vmc_state):
    return pretrainer.init(fixed_vmc_state)


@pytest.fixture(scope='module')
def pretraining_systems(pretrainer, systems_with_hf):
    return pretrainer.init_systems(jax.random.key(8), systems_with_hf)
