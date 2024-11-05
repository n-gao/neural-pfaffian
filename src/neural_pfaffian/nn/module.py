from enum import Enum
from typing import Callable, Final, TypeVar, TypeVarTuple

import flax.linen as nn
import jax
import numpy as np
from flax.struct import PyTreeNode, field
from jaxtyping import Array, ArrayLike, Float, Integer

from neural_pfaffian.systems import (
    ChunkSizeFunction,
    Systems,
    chunk_molecule,
    chunk_nuclei,
    chunk_nuclei_nuclei,
)

A = TypeVar('A', bound=jax.Array)
Ts = TypeVarTuple('Ts')

REPARAM_KEY: Final[str] = 'reparam'
REPARAM_META_KEY: Final[str] = 'reparam_meta'


class ParamType(PyTreeNode):
    name: str
    chunk_fn: ChunkSizeFunction = field(pytree_node=False)


class ParamTypes(Enum):
    GLOBAL = ParamType('global', chunk_molecule)
    NUCLEI = ParamType('nuclei', chunk_nuclei)
    NUCLEI_NUCLEI = ParamType('nuclei_nuclei', chunk_nuclei_nuclei)


class ParamMeta(PyTreeNode):
    param_type: ParamTypes = field(pytree_node=False)
    shape_and_dtype: jax.ShapeDtypeStruct
    mean: ArrayLike
    std: ArrayLike
    bias: bool
    chunk_axis: int | None
    keep_distr: bool


class ReparamModule(nn.Module):
    def reparam(
        self,
        name: str,
        init_fn: Callable[[jax.Array, *Ts], A],
        *init_args: *Ts,
        param_type: ParamTypes,
        bias: bool = True,
        chunk_axis: int | None = None,
        keep_distr: bool = False,
    ):
        # Like in self.param, we add the random key
        def init_with_random(*args: *Ts):
            return init_fn(self.make_rng(), *args)

        parameter = self.variable(REPARAM_KEY, name, init_with_random, *init_args).value
        exp_shape = jax.eval_shape(lambda: init_fn(jax.random.key(0), *init_args))
        if parameter.shape != exp_shape.shape:
            raise ValueError(
                f'Expected shape {exp_shape.shape} for parameter {name}, got {parameter.shape}'
            )

        def array_to_meta(array: A):
            # Remove the first dimension which will be taken by the system configuration
            return ParamMeta(
                param_type=param_type,
                shape_and_dtype=jax.ShapeDtypeStruct(array.shape[1:], array.dtype),
                mean=array.mean(),
                std=array.std(),
                bias=bias,
                chunk_axis=chunk_axis,
                keep_distr=keep_distr,
            )

        meta = self.variable(REPARAM_META_KEY, name, array_to_meta, parameter)
        return parameter, meta.value

    def edge_reparam(
        self,
        name: str,
        systems: Systems,
        init_fn: Callable[[jax.Array, tuple[int, ...]], A],
        shape: tuple[int, ...],
        max_charge: int | None,
        center_idx: Integer[ArrayLike, ' n_center'] | None,
        keep_distr: bool = False,
    ) -> Float[Array, ' n_nuc *shape']:
        # A utility function to reuse the same modules within the wave function and the meta network.
        n_out = np.prod(shape).item()
        # If the indices are missing, we just use the same params for all.
        if center_idx is None:
            return self.param(name, init_fn, shape)
        if max_charge is None:
            # Adaption per nucleus
            return self.reparam(
                name,
                init_fn,
                (systems.n_nuc, *shape),
                param_type=ParamTypes.NUCLEI,
                keep_distr=keep_distr,
            )[0][center_idx]
        elif max_charge >= 0:
            # Adaption per species
            return nn.Embed(
                num_embeddings=max_charge,
                features=n_out,
                embedding_init=lambda key, shape, dtype: init_fn(key, tuple(shape)),
                name=name,
            )(systems.flat_charges).reshape(systems.n_nuc, *shape)[center_idx]
        else:
            raise ValueError(f'Invalid max_charge {max_charge}')
