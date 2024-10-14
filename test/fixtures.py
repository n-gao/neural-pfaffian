import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from neural_pfaffian.nn.envelopes import EfficientEnvelope, FullEnvelope
from neural_pfaffian.nn.ferminet import FermiNet
from neural_pfaffian.nn.moon import Moon
from neural_pfaffian.nn.orbitals import Pfaffian
from neural_pfaffian.nn.psiformer import PsiFormer
from neural_pfaffian.nn.wave_function import WaveFunction
from neural_pfaffian.systems import Systems


@pytest.fixture
def one_system():
    return Systems(
        spins=((2, 2),),
        charges=((4,),),
        electrons=jax.random.normal(jax.random.key(0), (4, 3)),
        nuclei=jax.random.normal(jax.random.key(1), (1, 3)),
    )


@pytest.fixture
def two_systems():
    return Systems(
        spins=((2, 2), (3, 3)),
        charges=((4,), (2, 1)),
        electrons=jax.random.normal(jax.random.key(0), (10, 3)),
        nuclei=jax.random.normal(jax.random.key(1), (3, 3)),
    )


@pytest.fixture(params=['one_system', 'two_systems'])
def systems(request):
    return request.getfixturevalue(request.param)


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
def full_envelope():
    return FullEnvelope(1, 1, True)


@pytest.fixture(scope='function')
def efficient_envelope():
    return EfficientEnvelope(1, 1, True, 8)


@pytest.fixture(params=['full_envelope', 'efficient_envelope'])
def envelope(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pfaffian(envelope):
    return Pfaffian(2, 4, envelope, 10, 0.1, 1.0, 1.0)


@pytest.fixture(params=['pfaffian'])
def orbital_model(request, envelope):
    # envelope must be here since orbitals depend on it
    return request.getfixturevalue(request.param)


@pytest.fixture
def jastrow_models():
    return []


@pytest.fixture
def systems_embedding_params(systems: Systems, embedding_model: nn.Module):
    params = embedding_model.init(jax.random.key(42), systems)
    fwd_pass = jax.jit(embedding_model.apply)
    return systems, fwd_pass, params


@pytest.fixture
def wave_function(embedding_model, orbital_model, jastrow_models):
    wf = WaveFunction(embedding_model, orbital_model, jastrow_models)
    return wf


@pytest.fixture
def systems_wf_params(wave_function: WaveFunction, systems: Systems):
    params = wave_function.init(jax.random.key(42), systems)
    fwd_pass = jax.jit(wave_function.apply)
    return systems, fwd_pass, params
