from typing import Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, DTypeLike, PyTree


def tree_scale[T](tree: T, x: ArrayLike) -> T:
    return jtu.tree_map(lambda a: a * x, tree)


def tree_mul[T](tree: T, x: T | ArrayLike) -> T:
    if isinstance(x, ArrayLike):
        return tree_scale(tree, x)
    return jtu.tree_map(lambda a, b: a * b, tree, x)


def tree_shift[T](tree1: T, x: ArrayLike) -> T:
    return jtu.tree_map(lambda a: a + x, tree1)


def tree_add[T](tree1: T, tree2: T | ArrayLike) -> T:
    if isinstance(tree2, ArrayLike):
        return tree_shift(tree1, tree2)
    return jtu.tree_map(lambda a, b: a + b, tree1, tree2)


def tree_sub[T](tree1: T, tree2: T) -> T:
    return jtu.tree_map(lambda a, b: a - b, tree1, tree2)


def tree_dot[T](a: T, b: T) -> Array:
    return jtu.tree_reduce(
        jnp.add, jtu.tree_map(jnp.sum, jax.tree_map(jax.lax.mul, a, b))
    )


def tree_sum[T](tree: PyTree[ArrayLike]) -> Array:
    return jtu.tree_reduce(jnp.add, jtu.tree_map(jnp.sum, tree))


def tree_squared_norm[T](tree: PyTree[ArrayLike]) -> Array:
    return jtu.tree_reduce(
        jnp.add, jtu.tree_map(lambda x: jnp.einsum('...,...->', x, x), tree)
    )


def tree_concat[T](trees: Sequence[T], axis: int = 0) -> T:
    return jtu.tree_map(lambda *args: jnp.concatenate(args, axis=axis), *trees)


def tree_split[T](tree: T, sizes: tuple[int]) -> tuple[T, ...]:
    idx = 0
    result: list[T] = []
    for s in sizes:
        result.append(jtu.tree_map(lambda x: x[idx : idx + s], tree))
        idx += s
    result.append(jtu.tree_map(lambda x: x[idx:], tree))
    return tuple(result)


def tree_idx[T](tree: T, idx) -> T:
    return jtu.tree_map(lambda x: x[idx], tree)


def tree_expand[T](tree: T, axis) -> T:
    return jtu.tree_map(lambda x: jnp.expand_dims(x, axis), tree)


def tree_take[T](tree: T, idx, axis) -> T:
    def take(x):
        indices = idx
        if isinstance(indices, slice):
            slices = [slice(None)] * x.ndim
            slices[axis] = idx
            return x[tuple(slices)]
        return jnp.take(x, indices, axis)

    return jtu.tree_map(take, tree)


def tree_to_dtype[T](tree: T, dtype: DTypeLike) -> T:
    return jtu.tree_map(
        lambda x: x.astype(dtype) if isinstance(x, jax.Array) else x, tree
    )


def tree_stack[T](*trees: T) -> T:
    def stack(*args):
        return jnp.stack(args)

    return jtu.tree_map(stack, *trees)
