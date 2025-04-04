from typing import Generic, Protocol, Self, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax.core import unfreeze
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Float, PyTree
from typing_extensions import NotRequired, TypedDict

from neural_pfaffian.hf import HFOrbitals
from neural_pfaffian.nn.module import (
    REPARAM_KEY,
    REPARAM_META_KEY,
    ParamMeta,
    ReparamModule,
)
from neural_pfaffian.systems import Systems, SystemsWithHF


class Parameters(TypedDict):
    params: PyTree[Array]
    reparam: NotRequired[PyTree[Array]]
    reparam_meta: NotRequired[PyTree[ParamMeta]]


Orb = TypeVar('Orb')

ElecEmbedding = Float[Array, 'electrons embedding']
ElecNucDistances = Float[Array, 'electrons_nuclei 4']
ElecElecDistances = Float[Array, 'electrons_electrons 4']
Sign = Float[Array, 'n_mols']
LogAmplitude = Float[Array, 'n_mols']
SignedLogAmplitude = tuple[Sign, LogAmplitude]
Loss = Float[Array, '']
S = TypeVar('S')


class MetaNetworkP(Protocol):
    charges: Sequence[int] | None

    def init(self, key: Array, systems: Systems) -> Parameters: ...
    def apply(self, params: Parameters, systems: Systems) -> PyTree[Array]: ...
    def copy(self, out_structure: PyTree[ParamMeta], charges: Sequence[int]) -> Self: ...


class EmbeddingP(Protocol):
    def __call__(self, systems: Systems) -> ElecEmbedding: ...


class AntisymmetrizerP[Orb, S](Protocol):
    def __call__(self, systems: Systems, elec_embeddings: ElecEmbedding) -> Orb: ...

    def to_slog_psi(self, systems: Systems, orbitals: Orb) -> SignedLogAmplitude: ...

    def match_hf_orbitals(
        self,
        systems: Systems,
        hf_orbitals: Sequence[HFOrbitals],
        orbitals: Orb,
        state: Sequence[S],
    ) -> tuple[Loss, list[S]]: ...

    def init_systems(self, key: Array, systems: SystemsWithHF) -> SystemsWithHF: ...


class JastrowP(Protocol):
    def init(
        self, key: Array, systems: Systems, elec_embeddings: ElecEmbedding
    ) -> Parameters: ...
    def apply(
        self, params: Parameters, systems: Systems, elec_embeddings: ElecEmbedding
    ) -> SignedLogAmplitude: ...
    def __call__(
        self, systems: Systems, elec_embeddings: ElecEmbedding
    ) -> SignedLogAmplitude: ...


class WaveFunctionParameters(PyTreeNode):
    wave_function: PyTree[Array]
    meta_network: PyTree[Array]


class WaveFunction(Generic[Orb, S], ReparamModule):
    embedding_module: EmbeddingP
    antisymmetrizer: AntisymmetrizerP[Orb, S]
    jastrow_modules: Sequence[JastrowP]

    def init(self, key: Array, systems: Systems) -> Parameters:
        return super().init(key, systems)  # type: ignore

    def apply(
        self, params: Parameters, systems: Systems, method=None, **kwargs
    ) -> LogAmplitude:
        return super().apply(params, systems, method=method, **kwargs)  # type: ignore

    def _embedding(self, systems: Systems) -> ElecEmbedding:
        return self.embedding_module(systems)

    def _orbitals(self, systems: Systems) -> Orb:
        return self.antisymmetrizer(systems, self.embedding_module(systems))

    def _signed(self, systems: Systems) -> SignedLogAmplitude:
        embedding = self.embedding_module(systems)
        sign, logpsi = self.antisymmetrizer.to_slog_psi(
            systems, self.antisymmetrizer(systems, embedding)
        )
        for jastrow in self.jastrow_modules:
            J_sign, J_logpsi = jastrow(systems, embedding)
            sign, logpsi = sign * J_sign, logpsi + J_logpsi
        return sign, logpsi

    def __call__(self, systems: Systems) -> LogAmplitude:
        return self._signed(systems)[1]

    def embedding(self, params: Parameters, systems: Systems) -> ElecEmbedding:
        return self.apply(params, systems, method=self._embedding)

    def orbitals(self, params: Parameters, systems: Systems) -> Orb:
        return self.apply(params, systems, method=self._orbitals)  # type: ignore

    def signed(self, params: Parameters, systems: Systems) -> SignedLogAmplitude:
        return self.apply(params, systems, method=self._signed)  # type: ignore


class GeneralizedWaveFunction(Generic[Orb, S], PyTreeNode):
    wave_function: WaveFunction[Orb, S] = field(pytree_node=False)
    meta_network: MetaNetworkP | None = field(pytree_node=False)
    reparam_meta: PyTree[ParamMeta] = field(pytree_node=False)
    _reparam: PyTree[Array] | None = None

    @classmethod
    def create(
        cls,
        wave_function: WaveFunction,
        meta_network: MetaNetworkP | None,
        systems: Systems,
    ):
        # We need to initialize a dummy system to obtain the meta information about
        # the reparametrized parameters.
        key = jax.random.key(0)
        params = wave_function.init(key, systems.example_input)
        reparam_meta = params.get(REPARAM_META_KEY, {})

        if meta_network is not None:
            charges = meta_network.charges
            if not charges:
                charges = tuple(np.unique(systems.flat_charges).astype(int))
            meta_network = meta_network.copy(out_structure=reparam_meta, charges=charges)
        else:
            meta_network = None
            reparam_meta = jax.tree_map(
                lambda x: x.replace(keep_distr=False),
                reparam_meta,
                is_leaf=lambda x: isinstance(x, ParamMeta),
            )

        return cls(
            wave_function=wave_function,
            meta_network=meta_network,
            reparam_meta=reparam_meta,
        )

    def init(self, key: Array, systems: Systems):
        params = self.wave_function.init(key, systems)
        if self.meta_network is not None:
            meta_params = self.meta_network.init(key, systems)
        else:
            meta_params = params.get(REPARAM_KEY, {})
        # Remove the reparametrization parameters from the actual parameters
        del params[REPARAM_META_KEY]
        del params[REPARAM_KEY]
        return WaveFunctionParameters(params, meta_params)

    def reparams(self, params: WaveFunctionParameters, systems: Systems) -> PyTree[Array]:
        if self._reparam is not None:
            return self._reparam
        if self.meta_network is not None:
            return unfreeze(self.meta_network.apply(params.meta_network, systems))
        else:
            return params.meta_network

    def wf_params(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        reparams: PyTree[Array] | None = None,
    ):
        if reparams is None:
            reparams = self.reparams(params, systems)
        wf_params = params.wave_function | {
            REPARAM_KEY: reparams,
            REPARAM_META_KEY: self.reparam_meta,
        }
        return wf_params

    def embedding(self, params: WaveFunctionParameters, systems: Systems):
        return self.wave_function.embedding(self.wf_params(params, systems), systems)

    def orbitals(self, params: WaveFunctionParameters, systems: Systems):
        return self.wave_function.orbitals(self.wf_params(params, systems), systems)

    def signed(self, params: WaveFunctionParameters, systems: Systems):
        return self.wave_function.signed(self.wf_params(params, systems), systems)

    def apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        reparams: PyTree[Array] | None = None,
    ):
        return self.wave_function.apply(
            self.wf_params(params, systems, reparams), systems
        )

    def batched_apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        reparams: PyTree[Array] | None = None,
    ):
        vmapped_apply = jax.vmap(self.apply, in_axes=(None, systems.electron_vmap, None))
        return vmapped_apply(params, systems, reparams)

    def fix_structure(self, params: WaveFunctionParameters, systems: Systems):
        return self.__class__(
            wave_function=self.wave_function,
            meta_network=self.meta_network,
            reparam_meta=self.reparam_meta,
            _reparam=self.reparams(params, systems),
        )

    def group_reparams(self, systems: Systems, reparams: PyTree[Array]):
        params, tree_def = jtu.tree_flatten(reparams)
        metas: list[ParamMeta] = tree_def.flatten_up_to(self.reparam_meta)
        for tensors in zip(
            *[
                systems.group(p, meta.param_type.value.chunk_fn)
                for p, meta in zip(params, metas)
            ]
        ):
            yield jtu.tree_unflatten(tree_def, tensors)


class HFWaveFunction(GeneralizedWaveFunction):
    # A utility wave function that only computes the HF log amplitude
    def __init__(self, **kwargs):
        pass

    def fix_structure(self, params, systems):
        return self

    def apply(self, params, systems: SystemsWithHF):
        result = []
        for up, down in systems.hf_orbitals:
            _, logdet_up = jnp.linalg.slogdet(up)
            _, logdet_down = jnp.linalg.slogdet(down)
            result.append(logdet_up + logdet_down)
        result = jnp.stack(result)
        return result[systems.inverse_unique_indices]
