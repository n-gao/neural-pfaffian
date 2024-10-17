"""
Utilities for working with JAX.
Some of these functions are taken from
https://github.com/deepmind/ferminet/tree/jax/ferminet
"""

import functools
import inspect
from typing import Callable, ParamSpec, Self, TypeVar, overload

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from chex import ArrayTree, PRNGKey
from flax.struct import PyTreeNode
from jax import core

T = TypeVar('T')


class SplittablePyTreeNode(PyTreeNode):
    @classmethod
    def static_fields(cls):
        static_fields = [
            k
            for k, v in cls.__dataclass_fields__.items()
            if not v.metadata.get('pytree_node', True)
        ]
        return static_fields

    @classmethod
    def data_fields(cls):
        data_fields = [
            k
            for k, v in cls.__dataclass_fields__.items()
            if v.metadata.get('pytree_node', True)
        ]
        return data_fields

    @property
    def static_args(self) -> Self:
        return self.__class__(
            **{k: getattr(self, k) for k in self.static_fields()},
            **{k: None for k in self.data_fields()},
        )

    @property
    def data_args(self) -> Self:
        return self.__class__(
            **{k: None for k in self.static_fields()},
            **{k: getattr(self, k) for k in self.data_fields()},
        )

    @classmethod
    def merge_static_and_data(cls, static: Self, data: Self) -> Self:
        return cls(
            **{k: getattr(static, k) for k in cls.static_fields()},
            **{k: getattr(data, k) for k in cls.data_fields()},
        )


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
    assert 'axis_name' not in pmap_kwargs, 'axis_name must not be specified'

    def inner_pmap(fun: C) -> C:
        static_argnums = pmap_kwargs.pop('static_broadcasted_argnums', ())
        in_axes = pmap_kwargs.pop('in_axes', 0)

        # check via signature if any of the arguments are SplittablePyTreeNodes
        arg_types = [arg.annotation for arg in inspect.signature(fun).parameters.values()]
        arg_is_splittable = [
            isinstance(t, type) and issubclass(t, SplittablePyTreeNode) for t in arg_types
        ]

        # Check via in_axes if any of the arguments are SplittablePyTreeNodes
        if not isinstance(in_axes, int):
            # if not an int, it must be a sequence
            for i, arg in enumerate(in_axes):
                if isinstance(arg, SplittablePyTreeNode):
                    arg_is_splittable[i] = True

        # indices of the splittable arguments
        splittable_idx = np.where(arg_is_splittable)[0]

        # Extend static_argnums to include the static args
        if isinstance(static_argnums, int):
            static_argnums = (static_argnums,)
        static_argnums = static_argnums + (len(arg_types),)

        # Extend in_axes to include the static args
        if not isinstance(in_axes, int):
            in_axes = [*in_axes, None]
            for i in splittable_idx:
                if isinstance(in_axes[i], SplittablePyTreeNode):
                    in_axes[i] = in_axes[i].data_args
            in_axes = tuple(in_axes)
        else:
            in_axes = tuple([in_axes] * (len(arg_types)) + [None])

        # Wrapper function that merges data and static
        def fun_with_static(*args, **kwargs):
            args, static = list(args[:-1]), args[-1]
            for i, j in enumerate(splittable_idx):
                if args[j] is not None:
                    args[j] = args[j].merge_static_and_data(static[i], args[j])
            return fun(*args, **kwargs)

        pmapped_fn = jax.pmap(
            fun_with_static,
            *pmap_args,
            **pmap_kwargs,
            axis_name=PMAP_AXIS_NAME,
            in_axes=in_axes,  # type: ignore
            static_broadcasted_argnums=static_argnums,
        )

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            assert len(kwargs) == 0, 'kwargs not supported'
            # Here we split the static and data arguments
            args, statics = list(args), []
            for i in splittable_idx:
                if args[i] is not None:
                    static, data = args[i].static_args, args[i].data_args
                    args[i] = data
                    statics.append(static)
                else:
                    statics.append(None)
            return pmapped_fn(*args, tuple(statics), **kwargs)

        return wrapper  # type: ignore

    if fun is None:
        return inner_pmap

    return inner_pmap(fun)
