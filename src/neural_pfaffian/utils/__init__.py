from collections import defaultdict
from typing import Any, Callable, Generic, NamedTuple, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import numpy.typing as npt
from jaxtyping import ArrayLike

from neural_pfaffian.utils.jax_utils import jit

T = TypeVar('T')


def unique(
    items: Sequence[T],
) -> tuple[tuple[T, ...], tuple[list[int], ...], np.ndarray, tuple[int, ...]]:
    """
    Returns the unique items in a tuple and the indices of the first occurence of each item.
    """
    unique = defaultdict(list)
    for i, x in enumerate(items):
        unique[x].append(i)
    return (
        tuple(unique.keys()),
        tuple(unique.values()),
        np.concatenate(tuple(np.ones_like(c) * i for i, c in enumerate(unique.values()))),
        tuple(x[0] for x in unique.values()),
    )


def merge_slices(*slices: slice) -> tuple[slice, ...]:
    """
    Merges adjacent slices.
    Assumes the slices to be ordered by their starting index and to be non-overlapping.

    Args:
    - slices: slices to merge
    Returns:
    - list of slices
    """
    result = list(slices)
    i = 0
    while i < len(result) + 1:
        while i + 1 < len(result) and result[i].stop == result[i + 1].start:
            result[i] = slice(result[i].start, result[i + 1].stop)
            del result[i + 1]
        i += 1
    return tuple(result)


def adj_idx(
    a_sizes: tuple[int, ...],
    b_sizes: tuple[int, ...] | None = None,
    drop_diagonal: bool = False,
    drop_off_block: bool = False,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Computes the indices of the adjacency matrix of a block matrix.

    Args:
    - a_sizes: sizes of the blocks in the first dimension
    - b_sizes: sizes of the blocks in the second dimension
    - drop_diagonal: whether to drop the diagonal of each block
    - drop_off_block: whether to drop the off-diagonal blocks
    Return:
    - i: row indices of the adjacency matrix
    - j: column indices of the adjacency matrix
    - m: indices of the blocks
    """
    if b_sizes is None:
        b_sizes = a_sizes
    assert np.allclose(a_sizes, b_sizes) or not drop_diagonal
    i, j, m = [], [], []
    off_a, off_b = 0, 0
    for k, (a, b) in enumerate(zip(a_sizes, b_sizes)):
        adj = np.ones((a, b))
        if drop_off_block:
            adj = np.triu(adj)
        if drop_diagonal:
            adj -= np.eye(a)
        _i, _j = np.where(adj)
        i.append(_i + off_a)
        j.append(_j + off_b)
        m.append(np.ones(_i.size, dtype=int) * k)
        off_a += a
        off_b += b
    return (
        np.concatenate(i, axis=0),
        np.concatenate(j, axis=0),
        np.concatenate(m, axis=0),
    )


T = TypeVar('T')


class EMAState(NamedTuple, Generic[T]):
    data: T
    weight: jax.Array


# EMA for usage in JAX
def ema_make(tree: T) -> EMAState[T]:
    """
    Creates an EMA state for a pytree.

    Args:
    - tree: pytree to create the EMA state for
    Return:
    - EMA state
    """
    return EMAState(jtu.tree_map(jnp.zeros_like, tree), jnp.zeros((), dtype=jnp.float32))


@jit
def ema_update(data: EMAState[T], value: T, decay: ArrayLike = 0.9) -> EMAState[T]:
    """
    Updates an EMA state with a new value.

    Args:
    - data: EMA state
    - value: value to update the EMA state with
    - decay: decay rate
    Return:
    - updated EMA state
    """
    tree, weight = data
    return EMAState(
        jtu.tree_map(lambda a, b: a * decay + b, tree, value), weight * decay + 1
    )


@jit
def ema_value(data: EMAState[T], backup: T | None = None) -> T:
    """
    Computes the EMA value of an EMA state.

    Args:
    - data: EMA state
    - backup: backup value to use if the weight is 0
    Return:
    - EMA value
    """
    tree, weight = data
    if backup is None:
        backup = tree
    is_nan = weight == 0
    return jtu.tree_map(lambda x, y: jnp.where(is_nan, y, x / weight), tree, backup)


def batch(data: Sequence[T], n: int) -> list[Sequence[T]]:
    """
    Batches data into chunks of size n.

    Args:
    - data: data to batch
    - n: size of the chunks
    Return:
    - batched data
    """
    return [data[i : i + n] for i in range(0, len(data), n)]


def itemgetter(*items: Any) -> Callable[[Sequence[T]], tuple[T, ...]]:
    """
    Implementation of itemgetter that always returns a tuple.

    Args:
    - items: items to get
    Return:
    - function that returns a tuple of the items
    """

    def g(obj: Sequence[T]) -> tuple[T, ...]:
        return tuple(obj[item] for item in items)

    return g


T = TypeVar('T')


class Modules(dict[str, type[T]], Generic[T]):
    def init(self, module: str, args: dict[str, dict[str, Any]], **kwargs) -> T:
        module = module.lower()
        return self[module](**args[module], **kwargs)

    def init_many(self, modules: Sequence[tuple[str, dict[str, Any]]]) -> tuple[T, ...]:
        return tuple(self[module.lower()](**args) for module, args in modules)
