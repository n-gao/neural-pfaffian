"""
Utilities for working with JAX.
Some of these functions are taken from
https://github.com/deepmind/ferminet/tree/jax/ferminet
"""

import functools
from typing import Callable, ParamSpec, TypeVar, overload

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from chex import ArrayTree, PRNGKey
from jax import core

T = TypeVar('T')


broadcast = jax.pmap(lambda x: x)
instance = functools.partial(jtu.tree_map, lambda x: x[0])

_p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def p_split(key: PRNGKey) -> tuple[PRNGKey, ...]:
    return _p_split(key)


Tree = TypeVar('Tree', bound=ArrayTree)


def replicate(pytree: T) -> T:
    n = jax.local_device_count()
    stacked_pytree = jtu.tree_map(lambda x: jax.lax.broadcast(x, (n,)), pytree)
    return broadcast(stacked_pytree)


# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
P = ParamSpec('P')
R = TypeVar('R')

pmean = functools.partial(jax.lax.pmean, axis_name=PMAP_AXIS_NAME)
psum = functools.partial(jax.lax.psum, axis_name=PMAP_AXIS_NAME)
pmax = functools.partial(jax.lax.pmax, axis_name=PMAP_AXIS_NAME)
pmin = functools.partial(jax.lax.pmin, axis_name=PMAP_AXIS_NAME)
pgather = functools.partial(jax.lax.all_gather, axis_name=PMAP_AXIS_NAME)
pall_to_all = functools.partial(jax.lax.all_to_all, axis_name=PMAP_AXIS_NAME)
pidx = functools.partial(jax.lax.axis_index, axis_name=PMAP_AXIS_NAME)


def wrap_if_pmap(p_func: Callable[[T], T]) -> Callable[[T], T]:
    @functools.wraps(p_func)
    def p_func_if_pmap(obj: T) -> T:
        try:
            core.axis_frame(PMAP_AXIS_NAME)
            return p_func(obj)
        except NameError:
            return obj

    return p_func_if_pmap


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


@overload
def pmap(fun: None = None, *jit_args, **jit_kwargs) -> Callable[[C], C]: ...


@overload
def pmap(fun: C, *jit_args, **jit_kwargs) -> C: ...


# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
def pmap(fun: C | None = None, *pmap_args, **pmap_kwargs) -> C | Callable[[C], C]:
    def inner_pmap(fun: C) -> C:
        pmapped = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)(
            fun, *pmap_args, **pmap_kwargs
        )

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            return pmapped(*args, **kwargs)

        return wrapper  # type: ignore

    if fun is None:
        return inner_pmap

    return inner_pmap(fun)
