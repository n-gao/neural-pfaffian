import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest

from neural_pfaffian.mcmc import MetroplisHastings
from neural_pfaffian.nn.envelopes import EfficientEnvelope, FullEnvelope
from neural_pfaffian.nn.ferminet import FermiNet
from neural_pfaffian.nn.jastrow import CuspJastrow, MLPJastrow
from neural_pfaffian.nn.meta_network import MetaGNN
from neural_pfaffian.nn.module import ParamMeta, ParamTypes
from neural_pfaffian.nn.moon import Moon
from neural_pfaffian.nn.orbitals import Pfaffian
from neural_pfaffian.nn.psiformer import PsiFormer
from neural_pfaffian.nn.wave_function import GeneralizedWaveFunction, WaveFunction
from neural_pfaffian.preconditioner import Identity, Preconditioner, Spring
from neural_pfaffian.pretraining import Pretraining
from neural_pfaffian.systems import Systems
from neural_pfaffian.vmc import VMC, ClipStatistic


# Systems
@pytest.fixture
def one_system():
    return Systems(
        spins=((2, 1),),
        charges=((3,),),
        electrons=jax.random.normal(jax.random.key(0), (3, 3), dtype=jnp.float32),
        nuclei=jax.random.normal(jax.random.key(1), (1, 3), dtype=jnp.float32),
        mol_data={},
    )


@pytest.fixture
def two_systems():
    return Systems(
        spins=((2, 2), (3, 3)),
        charges=((4,), (4, 2)),
        electrons=jax.random.normal(jax.random.key(0), (10, 3), dtype=jnp.float32),
        nuclei=jax.random.normal(jax.random.key(1), (3, 3), dtype=jnp.float32),
        mol_data={},
    )


@pytest.fixture(params=['one_system', 'two_systems'])
def systems(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def batched_systems(systems):
    return systems.replace(
        electrons=jax.random.normal(
            jax.random.key(0),
            (2 * jax.device_count(), *systems.electrons.shape),
            dtype=systems.electrons.dtype,
        )
    )


@pytest.fixture
def systems_float64(systems):
    return systems.replace(
        electrons=systems.electrons.astype(jnp.float64),
        nuclei=systems.nuclei.astype(jnp.float64),
    )


# Embedding
@pytest.fixture
def ferminet():
    return FermiNet(
        embedding_dim=16,
        hidden_dims=[(16, 4), (16, 4)],
        activation=jnp.tanh,
    )


@pytest.fixture
def psiformer():
    return PsiFormer(
        embedding_dim=32,
        dim=32,
        n_head=4,
        n_layer=2,
        activation=jnp.tanh,
    )


@pytest.fixture
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


@pytest.fixture(params=['ferminet', 'psiformer', 'moon'])
def embedding_model(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def embedding_params(embedding_model: nn.Module, systems: Systems):
    params = embedding_model.lazy_init(jax.random.key(42), systems)
    return params


@pytest.fixture
def embedding_fwdpass(embedding_model: nn.Module):
    return jax.jit(embedding_model.apply)


# Enveloeps
@pytest.fixture
def full_envelope():
    return FullEnvelope(1, 1, True, False)


@pytest.fixture(scope='function')
def efficient_envelope():
    return EfficientEnvelope(1, 1, True, False, 8)


@pytest.fixture(params=['full_envelope', 'efficient_envelope'])
def envelope(request):
    return request.getfixturevalue(request.param)


# Orbitals
@pytest.fixture
def pfaffian(envelope):
    return Pfaffian(2, 4, envelope, 10, 0.1, 1.0, 1.0)


@pytest.fixture(params=['pfaffian'])
def orbital_model(request, envelope):
    # envelope must be here since orbitals depend on it
    return request.getfixturevalue(request.param)


@pytest.fixture
def no_jastrow():
    return []


@pytest.fixture
def mlp_jastrow():
    return [MLPJastrow([8, 4], jnp.tanh)]


@pytest.fixture
def cusp_jastrow():
    return [CuspJastrow()]


@pytest.fixture
def double_jastrow(mlp_jastrow, cusp_jastrow):
    return mlp_jastrow + cusp_jastrow


# Jastrows
@pytest.fixture(params=['no_jastrow', 'mlp_jastrow', 'cusp_jastrow', 'double_jastrow'])
def jastrow_models(request):
    return request.getfixturevalue(request.param)


# Wave Function
@pytest.fixture
def wave_function(embedding_model, orbital_model, jastrow_models):
    wf = WaveFunction(embedding_model, orbital_model, jastrow_models)
    return wf


@pytest.fixture
def wf_params(wave_function: WaveFunction, systems: Systems):
    return wave_function.init(jax.random.key(42), systems)


@pytest.fixture
def wf_signed(wave_function: WaveFunction):
    return jax.jit(wave_function.signed)


@pytest.fixture
def wf_apply(wave_function: WaveFunction):
    return jax.jit(wave_function.apply)


# MetaGNN
@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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
    params=['global_meta', 'nuclei_meta', 'nuclei_nuclei_meta', 'chunked_meta']
)
def out_meta(request):
    return request.getfixturevalue(request.param)


# Generalized Wave fucntion
@pytest.fixture
def generalized_wf(wave_function, meta_gnn):
    return GeneralizedWaveFunction.create(wave_function, meta_gnn)


@pytest.fixture
def generalized_wf_params(generalized_wf: GeneralizedWaveFunction, one_system: Systems):
    return generalized_wf.init(jax.random.key(42), one_system)


# Example WF
@pytest.fixture
def neural_pfaffian(moon, double_jastrow, efficient_envelope, meta_gnn):
    pfaffian = Pfaffian(3, 4, efficient_envelope, 10, 0.1, 1.0, 1.0)
    return GeneralizedWaveFunction.create(
        WaveFunction(moon, pfaffian, double_jastrow), meta_gnn
    )


@pytest.fixture
def neural_pfaffian_params(neural_pfaffian: GeneralizedWaveFunction, one_system: Systems):
    return neural_pfaffian.init(jax.random.key(42), one_system)


@pytest.fixture
def identity_preconditioner(neural_pfaffian: GeneralizedWaveFunction):
    return Identity(neural_pfaffian)


@pytest.fixture
def spring_preconditioner(neural_pfaffian: GeneralizedWaveFunction):
    return Spring(neural_pfaffian, 1e-3, 0.99, jnp.float64)


@pytest.fixture(params=['identity_preconditioner', 'spring_preconditioner'])
def preconditioner(request) -> Preconditioner:
    return request.getfixturevalue(request.param)


@pytest.fixture
def mcmc(neural_pfaffian: GeneralizedWaveFunction):
    return MetroplisHastings(neural_pfaffian, 5, jnp.array(1.0), 2, 0.5, 0.025)


@pytest.fixture
def optimizer():
    return optax.adam(1e-4)


@pytest.fixture
def vmc(neural_pfaffian, preconditioner, mcmc, optimizer):
    return VMC(
        wave_function=neural_pfaffian,
        preconditioner=preconditioner,
        optimizer=optimizer,
        sampler=mcmc,
        clip_local_energy=5.0,
        clip_statistic=ClipStatistic.MEDIAN,
    )


@pytest.fixture
def vmc_state(vmc: VMC):
    return vmc.init(jax.random.key(0))


@pytest.fixture
def vmc_systems(vmc: VMC, batched_systems: Systems):
    return vmc.init_systems(jax.random.key(7), batched_systems)


@pytest.fixture
def fixed_vmc(neural_pfaffian, spring_preconditioner, mcmc, optimizer):
    return VMC(
        wave_function=neural_pfaffian,
        preconditioner=spring_preconditioner,
        optimizer=optimizer,
        sampler=mcmc,
        clip_local_energy=5.0,
        clip_statistic=ClipStatistic.MEDIAN,
    )


@pytest.fixture
def fixed_vmc_state(fixed_vmc: VMC):
    return fixed_vmc.init(jax.random.key(0))


@pytest.fixture
def pretrainer(fixed_vmc, optimizer):
    pretrainer = Pretraining(fixed_vmc, optimizer, 1e-6)
    return pretrainer


@pytest.fixture
def systems_with_hf(batched_systems):
    return batched_systems.with_hf('sto-6g')


@pytest.fixture
def pretrainer_state(pretrainer, fixed_vmc_state):
    return pretrainer.init(fixed_vmc_state)


@pytest.fixture
def pretraining_systems(pretrainer, systems_with_hf):
    return pretrainer.init_systems(jax.random.key(8), systems_with_hf)
