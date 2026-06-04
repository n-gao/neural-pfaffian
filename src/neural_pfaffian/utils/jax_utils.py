"""
Utilities for working with JAX.
Some of these functions are taken from
https://github.com/deepmind/ferminet/tree/jax/ferminet
"""

import functools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Self, TypeVar, cast, overload

import jax
import jax.numpy as jnp
from flax.serialization import from_bytes, to_bytes
from flax.struct import PyTreeNode
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec

_BATCH_AXIS = 'qmc_batch'


MESH = jax.make_mesh((jax.device_count(),), (_BATCH_AXIS,))
BATCH_SPEC = PartitionSpec(_BATCH_AXIS)
BATCH_SHARDING = NamedSharding(MESH, BATCH_SPEC)
REPLICATE_SPEC = PartitionSpec()
REPLICATE_SHARDING = NamedSharding(MESH, REPLICATE_SPEC)


def distribute_keys(key: jax.Array) -> jax.Array:
    return jax.random.split(key, jax.device_count())[pidx()]


pmean = functools.partial(jax.lax.pmean, axis_name=_BATCH_AXIS)
psum = functools.partial(jax.lax.psum, axis_name=_BATCH_AXIS)
pmax = functools.partial(jax.lax.pmax, axis_name=_BATCH_AXIS)
pmin = functools.partial(jax.lax.pmin, axis_name=_BATCH_AXIS)
pgather = functools.partial(jax.lax.all_gather, axis_name=_BATCH_AXIS)
pall_to_all = functools.partial(jax.lax.all_to_all, axis_name=_BATCH_AXIS)
pidx = functools.partial(jax.lax.axis_index, axis_name=_BATCH_AXIS)


def pvary(x):
    if hasattr(jax.lax, 'pvary'):
        return jax.lax.pvary(x, axis_name=_BATCH_AXIS)
    return x


def wrap_if_pmap[C: Callable](p_func: C) -> C:
    @functools.wraps(p_func)
    def p_func_if_pmap[T](obj: T, *args, **kwargs) -> T:
        try:
            jax.lax.axis_index(_BATCH_AXIS)
            return p_func(obj, *args, **kwargs)
        except NameError:
            return obj

    return p_func_if_pmap  # type: ignore


pmean_if_pmap = wrap_if_pmap(pmean)
psum_if_pmap = wrap_if_pmap(psum)
pmax_if_pmap = wrap_if_pmap(pmax)
pmin_if_pmap = wrap_if_pmap(pmin)
pgather_if_pmap = wrap_if_pmap(pgather)


C = TypeVar('C', bound=Callable)


@overload
def jit[C: Callable](fun: None = None, *jit_args, **jit_kwargs) -> Callable[[C], C]: ...


@overload
def jit[C: Callable](fun: C, *jit_args, **jit_kwargs) -> C: ...


@functools.wraps(jax.jit)
def jit[C: Callable](
    fun: C | None = None,
    *jit_args,
    **jit_kwargs,
) -> C | Callable[[C], C]:
    def inner_jit(fun: C) -> C:
        jitted = jax.jit(fun, *jit_args, **jit_kwargs)

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            return jitted(*args, **kwargs)

        return wrapper  # type: ignore

    if fun is None:
        return inner_jit

    return inner_jit(fun)


@overload
def vectorize(fun: None = None, *vec_args, **vec_kwargs) -> Callable[[C], C]: ...


@overload
def vectorize[C: Callable](fun: C, *vec_args, **vec_kwargs) -> C: ...


def vectorize[C: Callable](
    fun: C | None = None,
    *vec_args,
    **vec_kwargs,
) -> C | Callable[[C], C]:
    def inner_jit(fun: C) -> C:
        vectorized = jnp.vectorize(fun, *vec_args, **vec_kwargs)

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            return vectorized(*args, **kwargs)

        return wrapper  # type: ignore

    if fun is None:
        return inner_jit

    return inner_jit(fun)


@functools.wraps(shard_map)
def shmap[C: Callable](
    fun: C | None = None,
    *shmap_args,
    **shmap_kwargs,
) -> C | Callable[[C], C]:
    def inner_shmap(fun: C) -> C:
        return shard_map(fun, *shmap_args, mesh=MESH, **shmap_kwargs)  # type: ignore

    if fun is None:
        return inner_shmap

    return inner_shmap(fun)


@overload
def vmap(fun: None = None, *vmap_args, **vmap_kwargs) -> Callable[[C], C]: ...


@overload
def vmap[C: Callable](fun: C, *vmap_args, **vmap_kwargs) -> C: ...


@functools.wraps(jax.vmap)
def vmap[C: Callable](
    fun: C | None = None,
    *vmap_args,
    **vmap_kwargs,
) -> C | Callable[[C], C]:
    def inner_vmap(fun: C) -> C:
        vmapped = jax.vmap(fun, *vmap_args, **vmap_kwargs)

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            return vmapped(*args, **kwargs)

        return wrapper  # type: ignore

    if fun is None:
        return inner_vmap

    return inner_vmap(fun)


Axis = int | tuple[int, ...]


def pad_along_axis(
    array: jax.Array,
    pad_width: int | tuple[int, int],
    axis: Axis = -1,
    mode: str | Callable[..., Any] = 'constant',
    **kwargs,
) -> jax.Array:
    """
    Pads an array along a specified axis.
    This is a convenience wrapper around `jax.numpy.pad` that simplifies padding.
    Instead of specifying padding along all axes, you can specify the axes
    along which to pad.

    Args:
        array: The input array to pad.
        pad_width: The number of elements to pad on both sides of the specified axis.
            If `pad_width` is an integer, it will pad all axes on both sides.
            If `pad_width` is a tuple, it should contain two integers specifying the
            number of elements to pad on both sides of the specified axes.
        axis: The axes along which to pad the array.
            Defaults to padding the last axis -1.

    Returns:
        A new array with the specified padding applied.
    """
    if isinstance(pad_width, int):
        pad_width = (pad_width, pad_width)
    full_pad_width = [(0, 0)] * array.ndim

    if isinstance(axis, int):
        full_pad_width[axis] = pad_width
    elif isinstance(axis, Iterable):
        for ax in axis:
            full_pad_width[ax] = pad_width
    return jnp.pad(array, full_pad_width, mode=mode, **kwargs)


class SerializeablePyTree(PyTreeNode):
    serialize = to_bytes
    deserialize = from_bytes

    def to_file(self, path: str | Path):
        with Path(path).open('wb') as f:
            f.write(self.serialize())

    def from_file(self, path: str | Path) -> Self:
        return cast('Self', self.deserialize(Path(path).read_bytes()))

    @property
    def partition_spec(self) -> PartitionSpec | Self:
        return REPLICATE_SPEC

    @property
    def sharding(self):
        def to_sharding(x: PartitionSpec):
            return NamedSharding(MESH, x)

        return jax.tree.map(
            to_sharding,
            self.partition_spec,
            is_leaf=lambda x: isinstance(x, PartitionSpec),
        )

    @property
    def sharded(self):
        return jax.device_put(self, self.sharding)
