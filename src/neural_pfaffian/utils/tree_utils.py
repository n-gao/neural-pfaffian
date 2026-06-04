from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, DTypeLike, PyTree


def tree_scale[T](tree: T, x: ArrayLike) -> T:
    return jax.tree.map(lambda a: a * x, tree)


def tree_mul[T](tree: T, x: T | ArrayLike) -> T:
    if isinstance(x, ArrayLike):
        return tree_scale(tree, x)
    return jax.tree.map(lambda a, b: a * b, tree, x)


def tree_shift[T](tree1: T, x: ArrayLike) -> T:
    return jax.tree.map(lambda a: a + x, tree1)


def tree_add[T](tree1: T, tree2: T | ArrayLike) -> T:
    if isinstance(tree2, ArrayLike):
        return tree_shift(tree1, tree2)
    return jax.tree.map(lambda a, b: a + b, tree1, tree2)


def tree_sub[T](tree1: T, tree2: T) -> T:
    return jax.tree.map(lambda a, b: a - b, tree1, tree2)


def tree_dot[T](a: T, b: T) -> Array:
    return jax.tree.reduce(
        jnp.add,
        jax.tree.map(jnp.sum, jax.tree.map(jax.lax.mul, a, b)),
    )


def tree_sum[T](tree: PyTree[ArrayLike]) -> Array:
    return jax.tree.reduce(jnp.add, jax.tree.map(jnp.sum, tree))


def tree_squared_norm[T](tree: PyTree[ArrayLike]) -> Array:
    return jax.tree.reduce(
        jnp.add,
        jax.tree.map(lambda x: jnp.einsum('...,...->', x, x), tree),
    )


def tree_concat[T](trees: Sequence[T], axis: int = 0) -> T:
    return jax.tree.map(lambda *args: jnp.concatenate(args, axis=axis), *trees)


def tree_split[T](tree: T, sizes: tuple[int]) -> tuple[T, ...]:
    idx = 0
    result: list[T] = []
    for s in sizes:
        result.append(jax.tree.map(lambda x: x[idx : idx + s], tree))
        idx += s
    result.append(jax.tree.map(lambda x: x[idx:], tree))
    return tuple(result)


def tree_idx[T](tree: T, idx) -> T:
    return jax.tree.map(lambda x: x[idx], tree)


def tree_expand[T](tree: T, axis) -> T:
    return jax.tree.map(lambda x: jnp.expand_dims(x, axis), tree)


def tree_take[T](tree: T, idx, axis) -> T:
    def take(x):
        indices = idx
        if isinstance(indices, slice):
            slices = [slice(None)] * x.ndim
            slices[axis] = idx
            return x[tuple(slices)]
        return jnp.take(x, indices, axis)

    return jax.tree.map(take, tree)


def tree_to_dtype[T](tree: T, dtype: DTypeLike) -> T:
    return jax.tree.map(
        lambda x: x.astype(dtype) if isinstance(x, jax.Array) else x,
        tree,
    )


def tree_stack[T](*trees: T) -> T:
    def stack(*args):
        return jnp.stack(args)

    return jax.tree.map(stack, *trees)


def is_tree_finite(tree: PyTree) -> Bool[Array, '']:
    # Code from chex
    labeled_tree = jax.tree.map(
        lambda x: jax.lax.select(jnp.isfinite(x).all(), 0.0, jnp.nan),
        tree,
    )
    return jnp.all(
        jnp.isfinite(jnp.asarray(jax.tree_util.tree_leaves(labeled_tree))),
    )
