import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Generic, Self, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax.struct import PyTreeNode
from jaxtyping import Array, ArrayLike, Float

from .jax_utils import SerializeablePyTree, jit


def unique[T](
    items: Sequence[T],
) -> tuple[tuple[T, ...], tuple[list[int], ...], np.ndarray, tuple[int, ...]]:
    """
    Identify unique items in a sequence, group their indices, and map each item to a group.

    Args:
        items (Sequence[T]): A sequence of items.

    Returns:
        tuple:
            A 4-element tuple containing:

            1. tuple[T, ...]:
                All unique items in the order they appear in `items`.

            2. tuple[list[int], ...]:
                A tuple of lists, each list containing the indices at which a unique item appears.

            3. np.ndarray:
                An array of integer group labels, where each element indicates
                the group index of the corresponding element in the original sequence.

            4. tuple[int, ...]:
                The first occurrence index of each unique item, in the same order
                as the tuple of unique items.
    """
    unique = defaultdict(list)
    for i, x in enumerate(items):
        unique[x].append(i)
    group_ids = np.empty(len(items), dtype=int)
    for i, indices in enumerate(unique.values()):
        group_ids[indices] = i
    return (
        tuple(unique.keys()),  # unique items
        tuple(unique.values()),  # tuple of lists of indices
        group_ids,
        tuple(x[0] for x in unique.values()),  # first occurence of each item
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
    for k, (a, b) in enumerate(zip(a_sizes, b_sizes, strict=False)):
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


class EMA(Generic[T], SerializeablePyTree):
    data: T
    weight: Float[Array, '']
    """"The accumulated weight used for normalization, ensuring that the EMA is corrected
    for the bias introduced by the initial state."""

    @classmethod
    def init(cls, data: T, initial_bias_strength: float = 0.0) -> 'EMA[T]':
        if initial_bias_strength > 0:
            return cls(
                jax.tree.map(lambda x: jnp.full_like(x, jnp.nan, x.dtype), data),
                jnp.ones((), dtype=jnp.float32) * initial_bias_strength,
            )
        return cls(jax.tree.map(jnp.zeros_like, data), jnp.zeros((), dtype=jnp.float32))

    @jit
    def update(self, value: T, decay: ArrayLike) -> Self:
        init_update = jax.tree.map(lambda x: jnp.isnan(x), self.data)

        def _select_data(cond, data, value):
            return jnp.where(cond, value * self.weight, data * decay + value)

        return self.replace(
            data=jax.tree.map(_select_data, init_update, self.data, value),
            weight=jnp.where(
                jax.tree.reduce(jnp.logical_and, jax.tree.map(jnp.all, init_update)),
                self.weight,
                self.weight * decay + 1,
            ),
        )

    @jit
    def value(self, backup: T | None = None) -> T:
        if backup is None:
            backup = self.data
        is_nan = self.weight == 0
        return jax.tree.map(
            lambda x, y: jnp.where(is_nan, y, x / self.weight),
            self.data,
            backup,
        )


class RollingAverage(Generic[T], PyTreeNode):
    data: T

    @classmethod
    def init(cls, data: T, window_size: int = 5_000) -> 'RollingAverage[T]':
        return cls(
            jax.tree.map(
                lambda x: jnp.full((*x.shape, window_size), jnp.nan, x.dtype),
                data,
            ),
        )

    @jit
    def update(self, value: T, *_) -> Self:
        def _update_arr(val, arr):
            return jnp.roll(arr, 1, axis=-1).at[..., 0].set(val)

        return self.replace(
            data=jax.tree.map(_update_arr, value, self.data),
        )

    @jit
    def value(self) -> T:
        return jax.tree.map(lambda x: jnp.nanmean(x, axis=-1), self.data)


def batch[T](data: Sequence[T], n: int) -> list[Sequence[T]]:
    """
    Batches data into chunks of size n.

    Args:
    - data: data to batch
    - n: size of the chunks
    Return:
    - batched data
    """
    return [data[i : i + n] for i in range(0, len(data), n)]


def itemgetter[T](*items: Any) -> Callable[[Sequence[T]], tuple[T, ...]]:
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


class Modules[T](dict[str, type[T]]):
    def init_or_none(self, module: str | None, **kwargs) -> T | None:
        if module is None:
            return None
        return self.init(module, **kwargs)

    def init(self, module: str, args: dict[str, dict[str, Any]], **kwargs) -> T:
        module = module.lower()
        try:
            # Try a 'factory' initializer
            return self[module].create(**args.get(module, {}), **kwargs)  # type: ignore
        except AttributeError:
            # Try a 'constructor' initializer
            return self[module](**args.get(module, {}), **kwargs)

    def init_many(
        self,
        modules: Sequence[tuple[str, dict[str, Any]]] | dict[str, dict[str, Any]],
    ) -> tuple[T, ...]:
        if isinstance(modules, dict):
            return tuple(self[k.lower()](**kwargs) for k, kwargs in modules.items())
        return tuple(self[module.lower()](**args) for module, args in modules)

    def try_init_many(
        self,
        modules: Sequence[tuple[str, dict[str, Any]]] | dict[str, dict[str, Any]],
    ) -> tuple[T, ...]:
        result = []
        if isinstance(modules, dict):
            for k, kwargs in modules.items():
                try:
                    result.append(self[k.lower()](**kwargs))
                except Exception:
                    logging.warning(f'Failed to initialize {k}', exc_info=True)
            return tuple(result)
        for module, args in modules:
            try:
                result.append(self[module.lower()](**args))
            except Exception:
                logging.warning(f'Failed to initialize {module}', exc_info=True)
        return tuple(result)
