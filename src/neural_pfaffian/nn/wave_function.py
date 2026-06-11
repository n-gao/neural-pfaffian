from collections.abc import Sequence
from typing import Generic, NotRequired, Protocol, Self, TypeVar

import jax
import numpy as np
from flax.core import unfreeze
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Float, PyTree
from typing_extensions import TypedDict

from neural_pfaffian.nn.module import (
    REPARAM_KEY,
    REPARAM_META_KEY,
    ParamMeta,
    ReparamModule,
)
from neural_pfaffian.systems import Systems, SystemsWithPretrainTarget


class Parameters(TypedDict):
    params: PyTree[Array]
    reparam: NotRequired[PyTree[Array]]
    reparam_meta: NotRequired[PyTree[ParamMeta]]


Orb = TypeVar('Orb')
COrb = TypeVar('COrb')
"""Core orbitals independent of excitation"""


ElecEmbedding = Float[Array, 'electrons embedding']
ElecNucDistances = Float[Array, 'electrons_nuclei 4']
ElecElecDistances = Float[Array, 'electrons_electrons 4']
Sign = Float[Array, 'n_mols']
LogAmplitude = Float[Array, 'n_mols']
SignedLogAmplitude = tuple[Sign, LogAmplitude]
Loss = Float[Array, '']
AuxData = dict[str, jax.Array]
S = TypeVar('S')
SP = TypeVar('SP', bound=Systems, contravariant=True)


class MetaNetworkP(Protocol):
    charges: Sequence[int] | None

    def init(self, key: Array, systems: Systems) -> Parameters: ...
    def apply(self, params: Parameters, systems: Systems) -> PyTree[Array]: ...
    def copy(
        self,
        out_structure: PyTree[ParamMeta],
        charges: Sequence[int],
    ) -> Self: ...


class EmbeddingP(Protocol):
    def __call__(self, systems: Systems) -> ElecEmbedding: ...


class AntisymmetrizerP(Protocol[Orb, COrb, S]):
    max_num_states: int | None = None
    """Maximum number of states the wave function can represent."""

    def __call__(
        self,
        systems: Systems,
        elec_embeddings: ElecEmbedding,
    ) -> Orb: ...

    def core_orbitals(
        self,
        systems: Systems,
        elec_embeddings: ElecEmbedding,
    ) -> COrb: ...

    def apply_excitation(self, systems: Systems, core_orbitals: COrb) -> Orb: ...

    def to_slog_psi(self, systems: Systems, orbitals: Orb) -> SignedLogAmplitude: ...

    def match_hf_orbitals(
        self,
        systems: SystemsWithPretrainTarget,
        orbitals: Orb,
    ) -> tuple[Loss, list[S], AuxData]: ...

    def init_systems(
        self,
        key: Array,
        systems: SystemsWithPretrainTarget,
    ) -> SystemsWithPretrainTarget: ...


class JastrowP(Protocol):
    def __call__(
        self,
        systems: Systems,
        elec_embeddings: ElecEmbedding,
    ) -> SignedLogAmplitude: ...


class WaveFunctionParameters(PyTreeNode):
    wave_function: PyTree[Array]
    meta_network: PyTree[Array]


class WaveFunction(Generic[Orb, COrb, S], ReparamModule):
    embedding_module: EmbeddingP
    antisymmetrizer: AntisymmetrizerP[Orb, COrb, S]
    jastrow_modules: Sequence[JastrowP]

    def init(self, key: Array, systems: Systems) -> Parameters:  # type: ignore
        return super().init(key, systems)  # type: ignore

    def apply(
        self,
        params: Parameters,
        systems: Systems,
        method=None,
        **kwargs,
    ) -> LogAmplitude:
        return super().apply(params, systems, method=method, **kwargs)  # type: ignore

    def _embedding(self, systems: Systems) -> ElecEmbedding:
        return self.embedding_module(systems)

    def _orbitals(self, systems: Systems) -> Orb:
        return self.antisymmetrizer(systems, self.embedding_module(systems))

    def _core_orbitals(
        self,
        systems: Systems,
        embeddings: ElecEmbedding | None = None,
    ) -> COrb:
        embeddings = (
            embeddings if embeddings is not None else self.embedding_module(systems)
        )
        return self.antisymmetrizer.core_orbitals(systems, embeddings)

    def _apply_excitation(self, systems: Systems, core_orbitals: COrb) -> Orb:
        return self.antisymmetrizer.apply_excitation(systems, core_orbitals)

    def _signed(
        self,
        systems: Systems,
        orbitals: Orb | None = None,
        embeddings: ElecEmbedding | None = None,
    ) -> SignedLogAmplitude:
        # Do not recompute orbitals and embeddings if they are provided
        embedding = (
            embeddings if embeddings is not None else self.embedding_module(systems)
        )
        orbitals = (
            orbitals if orbitals is not None else self.antisymmetrizer(systems, embedding)
        )
        sign, logpsi = self.antisymmetrizer.to_slog_psi(systems, orbitals)
        for jastrow in self.jastrow_modules:
            jas_sign, jas_logpsi = jastrow(systems, embedding)
            sign, logpsi = sign * jas_sign, logpsi + jas_logpsi
        return sign, logpsi

    def __call__(self, systems: Systems) -> LogAmplitude:
        return self._signed(systems)[1]

    def embedding(self, params: Parameters, systems: Systems) -> ElecEmbedding:
        return self.apply(params, systems, method=self._embedding)

    def orbitals(self, params: Parameters, systems: Systems) -> Orb:
        return self.apply(params, systems, method=self._orbitals)  # type: ignore

    def core_orbitals(
        self,
        params: Parameters,
        systems: Systems,
        embeddings: ElecEmbedding | None = None,
    ) -> COrb:
        """Core orbitals that are independent of excitations. Excitation dependence can be applied
        with `apply_excitation`."""
        return self.apply(
            params,
            systems,
            method=self._core_orbitals,
            embeddings=embeddings,
        )  # type: ignore[return-value]

    def apply_excitation(
        self,
        params: Parameters,
        systems: Systems,
        core_orbitals: COrb,
    ) -> Orb:
        """Applies excitation dependence to `core_orbitals`"""
        return self.apply(
            params,
            systems,
            core_orbitals=core_orbitals,
            method=self._apply_excitation,
        )  # type: ignore[return-value]

    def signed(
        self,
        params: Parameters,
        systems: Systems,
        orbitals: Orb | None = None,
        embeddings: ElecEmbedding | None = None,
    ) -> SignedLogAmplitude:
        return self.apply(
            params,
            systems,
            method=self._signed,
            orbitals=orbitals,
            embeddings=embeddings,
        )  # type: ignore[return-value]


class GeneralizedLogAmplitude(Protocol[SP]):
    def fix_structure(self, params: WaveFunctionParameters, systems: SP) -> Self: ...

    def apply(self, params: WaveFunctionParameters, systems: SP) -> LogAmplitude: ...


class GeneralizedWaveFunction(
    Generic[Orb, COrb, S, SP],
    GeneralizedLogAmplitude[SP],
    PyTreeNode,
):
    wave_function: WaveFunction[Orb, COrb, S] = field(pytree_node=False)
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
        # Update dynamic configuration of network components
        wave_function.antisymmetrizer.max_num_states = systems.max_num_states

        # We need to initialize a dummy system to obtain the meta information about
        # the reparametrized parameters.
        key = jax.random.key(0)
        params = wave_function.init(key, systems.example_input)
        reparam_meta = params.get(REPARAM_META_KEY, {})

        if meta_network is not None:
            charges = meta_network.charges
            if not charges:
                charges = tuple(np.unique(systems.flat_charges).astype(int))
            meta_network = meta_network.copy(
                out_structure=reparam_meta,
                charges=charges,
            )
        else:
            meta_network = None
            reparam_meta = jax.tree.map(
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

    def reparams(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
    ) -> PyTree[Array]:
        if self._reparam is not None:
            return self._reparam
        if self.meta_network is not None:
            return jax.tree.map(
                lambda x, meta: systems.replicate_unique_array_per_mol(
                    x,
                    meta.param_type.value.chunk_fn,
                ),
                unfreeze(
                    self.meta_network.apply(
                        params.meta_network,
                        systems.unique_systems,
                    ),
                ),
                self.reparam_meta,
            )
        return params.meta_network

    def wf_params(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        reparams: PyTree[Array] | None = None,
    ):
        if reparams is None:
            reparams = self.reparams(params, systems)
        return params.wave_function | {
            REPARAM_KEY: reparams,
            REPARAM_META_KEY: self.reparam_meta,
        }

    def embedding(self, params: WaveFunctionParameters, systems: Systems):
        return self.wave_function.embedding(self.wf_params(params, systems), systems)

    def orbitals(self, params: WaveFunctionParameters, systems: Systems) -> Orb:
        return self.wave_function.orbitals(self.wf_params(params, systems), systems)

    def core_orbitals(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        embeddings: ElecEmbedding | None = None,
    ) -> COrb:
        """Core orbitals that are independent of excitations. Excitation dependence can be applied
        with `apply_excitation`."""
        return self.wave_function.core_orbitals(
            self.wf_params(params, systems),
            systems,
            embeddings,
        )

    def apply_excitation(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        core_orbitals: COrb,
    ) -> Orb:
        """Applies excitation dependence to `core_orbitals`"""
        return self.wave_function.apply_excitation(
            self.wf_params(params, systems),
            systems,
            core_orbitals,
        )

    def signed(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        orbitals: Orb | None = None,
        embeddings: ElecEmbedding | None = None,
        reparams: PyTree[Array] | None = None,
    ):
        return self.wave_function.signed(
            self.wf_params(params, systems, reparams),
            systems,
            orbitals,
            embeddings,
        )

    def apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        reparams: PyTree[Array] | None = None,
    ):
        return self.wave_function.apply(
            self.wf_params(params, systems, reparams),
            systems,
        )

    def batched_apply(
        self,
        params: WaveFunctionParameters,
        systems: Systems,
        reparams: PyTree[Array] | None = None,
    ):
        vmapped_apply = jax.vmap(
            self.apply,
            in_axes=(None, systems.electron_vmap, None),
        )
        result = vmapped_apply(params, systems, reparams)
        return result

    def fix_structure(self, params: WaveFunctionParameters, systems: Systems):
        # TODO: Replace with self.replace?
        return self.__class__(
            wave_function=self.wave_function,
            meta_network=self.meta_network,
            reparam_meta=self.reparam_meta,
            _reparam=self.reparams(params, systems),
        )

    def group_reparams(
        self,
        systems: Systems,
        reparams: PyTree[Array],
        *,
        include_excitation: bool = False,
    ):
        params, tree_def = jax.tree.flatten(reparams)
        metas: list[ParamMeta] = tree_def.flatten_up_to(self.reparam_meta)
        for tensors in zip(
            *[
                systems.group(
                    p,
                    meta.param_type.value.chunk_fn,
                    include_excitation=include_excitation,
                )
                for p, meta in zip(params, metas, strict=False)
            ],
            strict=False,
        ):
            yield jax.tree.unflatten(tree_def, tensors)


class MixtureLogAmplitude(
    GeneralizedLogAmplitude[SystemsWithPretrainTarget],
    PyTreeNode,
):
    wave_function: GeneralizedWaveFunction
    hf_fraction: float = field(pytree_node=False, default=0.0)

    def fix_structure(self, params: WaveFunctionParameters, systems: Systems) -> Self:
        return self.replace(
            wave_function=self.wave_function.fix_structure(params, systems),
        )

    def apply(
        self,
        params: WaveFunctionParameters,
        systems: SystemsWithPretrainTarget,
    ) -> LogAmplitude:
        if self.hf_fraction == 0:
            return self.wave_function.apply(params, systems)
        if self.hf_fraction == 1:
            return systems.slogamplitudes[1]

        hf_logamp = systems.slogamplitudes[1]
        wf_logamp = self.wave_function.apply(params, systems)

        return (1 - self.hf_fraction) * wf_logamp + self.hf_fraction * hf_logamp


class HFLogAmplitude(GeneralizedLogAmplitude[SystemsWithPretrainTarget], PyTreeNode):
    def fix_structure(self, params: WaveFunctionParameters, systems: Systems) -> Self:
        return self

    def apply(
        self,
        params: WaveFunctionParameters,
        systems: SystemsWithPretrainTarget,
    ) -> LogAmplitude:
        return systems.slogamplitudes[1]
