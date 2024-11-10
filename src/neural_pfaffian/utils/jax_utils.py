"""
Utilities for working with JAX.
Some of these functions are taken from
https://github.com/deepmind/ferminet/tree/jax/ferminet
"""

import functools
from typing import Callable, ParamSpec, TypeVar, overload

import jax
import jax.numpy as jnp
from chex import ArrayTree
from jax import core
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

_BATCH_SHARD = 'qmc_batch'


MESH = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), (_BATCH_SHARD,))
BATCH_SHARD = P(_BATCH_SHARD)
REPLICATE_SHARD = P()


T = TypeVar('T')


def replicate(pytree: T) -> T:
    return jax.device_put(pytree, REPLICATE_SHARD)


def broadcast(pytree: T) -> T:
    return jax.device_put(pytree, BATCH_SHARD)


def distribute_keys(key: jax.Array) -> jax.Array:
    return jax.random.split(key, jax.device_count())[pidx()]


Tree = TypeVar('Tree', bound=ArrayTree)

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
P = ParamSpec('P')
R = TypeVar('R')

pmean = functools.partial(jax.lax.pmean, axis_name=_BATCH_SHARD)
psum = functools.partial(jax.lax.psum, axis_name=_BATCH_SHARD)
pmax = functools.partial(jax.lax.pmax, axis_name=_BATCH_SHARD)
pmin = functools.partial(jax.lax.pmin, axis_name=_BATCH_SHARD)
pgather = functools.partial(jax.lax.all_gather, axis_name=_BATCH_SHARD)
pall_to_all = functools.partial(jax.lax.all_to_all, axis_name=_BATCH_SHARD)
pidx = functools.partial(jax.lax.axis_index, axis_name=_BATCH_SHARD)


C = TypeVar('C', bound=Callable)


def wrap_if_pmap(p_func: C) -> C:
    @functools.wraps(p_func)
    def p_func_if_pmap(obj: T, *args, **kwargs) -> T:
        try:
            core.axis_frame(_BATCH_SHARD)
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
def jit(fun: None = None, *jit_args, **jit_kwargs) -> Callable[[C], C]: ...


@overload
def jit(fun: C, *jit_args, **jit_kwargs) -> C: ...


def jit(fun: C | None = None, *jit_args, **jit_kwargs) -> C | Callable[[C], C]:
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
def vectorize(fun: C, *vec_args, **vec_kwargs) -> C: ...


def vectorize(fun: C | None = None, *vec_args, **vec_kwargs) -> C | Callable[[C], C]:
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
def shmap(fun: C | None = None, *shmap_args, **shmap_kwargs) -> C | Callable[[C], C]:
    def inner_shmap(fun: C) -> C:
        return shard_map(fun, MESH, *shmap_args, **shmap_kwargs)  # type: ignore

    if fun is None:
        return inner_shmap

    return inner_shmap(fun)
