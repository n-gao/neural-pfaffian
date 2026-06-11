from typing import override

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neural_pfaffian.nn.wave_function import (
    AntisymmetrizerP,
    EmbeddingP,
    GeneralizedWaveFunction,
    WaveFunction,
    WaveFunctionParameters,
)
from neural_pfaffian.systems import Systems


class MockEmbed(nn.Module, EmbeddingP):
    @nn.compact
    def __call__(self, systems):
        return systems.electrons


class MockOrbital(nn.Module, AntisymmetrizerP):
    max_num_states: int | None = 2

    @nn.compact
    def __call__(self, systems, elec_embeddings):
        dummy = self.param('dummy', lambda key: jnp.array(0.0, dtype=jnp.float32))
        centroids = 3.0 * jnp.arange(self.max_num_states, dtype=jnp.float32) + 0.0 * dummy
        excitations = jnp.array(systems.excitations)
        electrons = jnp.split(
            systems.electrons,
            np.cumsum(np.array(systems.n_elec_by_mol))[:-1],
            axis=-2,
        )

        def _apply(excitation, electron):
            return -jnp.mean(
                jnp.linalg.norm(electron - centroids[excitation], axis=-1),
                axis=-1,
            )

        return jnp.array(
            [_apply(e, el) for e, el in zip(excitations, electrons, strict=False)],
        )

    def core_orbitals(self, systems, elec_embeddings):
        return None

    def apply_excitation(self, systems, core_orbitals):
        return self.__call__(systems, None)

    def match_hf_orbitals(self, systems, orbitals):
        raise NotImplementedError

    def init_systems(self, key: jax.Array, systems):
        return systems

    @override
    def to_slog_psi(self, systems, orbitals):
        return jnp.ones_like(orbitals), orbitals


class MockOrthogonalWf(GeneralizedWaveFunction):
    def init(self, key: jax.Array, systems: Systems):
        return WaveFunctionParameters(
            self.wave_function.init(key, systems.example_input),
            {},
        )

    def group_reparams(self, systems, reparams, *, include_excitation: bool = False):
        for _ in systems.iter_stacked_sub_systems():
            yield reparams

    @override
    def signed(self, params, systems, orbitals=None, embeddings=None, reparams=None):
        return jnp.ones_like(psi := self.apply(params, systems, reparams)), psi


# ---------------------------------------------------------------------------
# Fixtures (depend on the `excited_systems` fixture from fixtures.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def mock_orthogonal_wf(excited_systems):
    return MockOrthogonalWf.create(
        WaveFunction(MockEmbed(), MockOrbital(), []),
        None,
        excited_systems,
    )


@pytest.fixture(scope='module')
def mock_wf_params(mock_orthogonal_wf, excited_systems):
    return mock_orthogonal_wf.init(jax.random.PRNGKey(0), excited_systems)


@pytest.fixture(scope='module')
def perfect_sample_systems(excited_systems):
    electrons = jnp.zeros_like(excited_systems.electrons, dtype=jnp.float32)
    n_first = excited_systems.n_elec_by_mol[0]
    electrons = electrons.at[n_first:].set(3.0)
    return excited_systems.replace(electrons=electrons)


@pytest.fixture(scope='module')
def batched_perfect_samples(perfect_sample_systems):
    return perfect_sample_systems.replace(
        electrons=jnp.stack([perfect_sample_systems.electrons] * 2),
    )


@pytest.fixture(scope='module')
def batched_perfect_random_samples(perfect_sample_systems):
    electrons = (
        jax.random.normal(
            jax.random.PRNGKey(0),
            (512, *perfect_sample_systems.electrons.shape),
            dtype=jnp.float32,
        )
        * 0.1
    )
    electrons = electrons.at[:, 4:].add(3.0)
    return perfect_sample_systems.replace(electrons=electrons)
