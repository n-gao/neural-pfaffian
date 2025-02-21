from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Callable, Generic, Self, Sequence, TypeVar

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import numpy.typing as npt
from flax.serialization import from_bytes, to_bytes
from flax.struct import PyTreeNode
from jaxtyping import Array, ArrayLike, Float

from .jax_utils import jit

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


class EMA(Generic[T], PyTreeNode):
    data: T
    weight: Float[Array, '']

    @classmethod
    def init(cls, data: T) -> 'EMA[T]':
        return cls(jtu.tree_map(jnp.zeros_like, data), jnp.zeros((), dtype=jnp.float32))

    @jit
    def update(self, value: T, decay: ArrayLike) -> Self:
        return self.replace(
            data=jtu.tree_map(lambda a, b: a * decay + b, self.data, value),
            weight=self.weight * decay + 1,
        )

    @jit
    def value(self, backup: T | None = None) -> T:
        if backup is None:
            backup = self.data
        is_nan = self.weight == 0
        return jtu.tree_map(
            lambda x, y: jnp.where(is_nan, y, x / self.weight), self.data, backup
        )


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
    def init_or_none(self, module: str | None, **kwargs) -> T | None:
        if module is None:
            return None
        return self.init(module, **kwargs)

    def init(self, module: str, args: dict[str, dict[str, Any]], **kwargs) -> T:
        module = module.lower()
        return self[module](**args[module], **kwargs)

    def init_many(
        self, modules: Sequence[tuple[str, dict[str, Any]]] | dict[str, dict[str, Any]]
    ) -> tuple[T, ...]:
        if isinstance(modules, dict):
            return tuple(self[k.lower()](**kwargs) for k, kwargs in modules.items())
        return tuple(self[module.lower()](**args) for module, args in modules)

    def try_init_many(
        self, modules: Sequence[tuple[str, dict[str, Any]]] | dict[str, dict[str, Any]]
    ) -> tuple[T, ...]:
        result = []
        if isinstance(modules, dict):
            for k, kwargs in modules.items():
                try:
                    result.append(self[k.lower()](**kwargs))
                except Exception:
                    logging.warn(f'Failed to initialize {k}')
            return tuple(result)
        for module, args in modules:
            try:
                result.append(self[module.lower()](**args))
            except Exception:
                logging.warn(f'Failed to initialize {module}')
        return tuple(result)


class SerializeablePyTree(PyTreeNode):
    serialize = to_bytes
    deserialize = from_bytes

    def to_file(self, path: str | Path):
        Path(path).open('wb').write(self.serialize())

    def from_file(self, path: str | Path):
        return self.deserialize(Path(path).read_bytes())
