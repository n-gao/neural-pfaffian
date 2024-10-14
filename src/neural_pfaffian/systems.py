from typing import (
    Generator,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
    TypeVarTuple,
    overload,
)

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import numpy.typing as npt
from flax.struct import PyTreeNode, field
from jaxtyping import Array, ArrayLike, Float, Integer

from neural_pfaffian.utils import adj_idx, merge_slices, unique

Electrons = Float[Array, '... n_elec 3']
Nuclei = Float[Array, '... n_nuc 3']

Spins = tuple[int, int]
Charges = tuple[int, ...] | Array

T_Array = TypeVar('T_Array', bound=Array)
ElecElecDistances = Float[Array, 'electrons_electrons 4']
ElecNucDistances = Float[Array, 'electrons_nuclei 4']
NucNucDistances = Float[Array, 'nuclei_nuclei 4']


class ChunkSizeFunction(Protocol):
    def __call__(self, s: Spins, c: Charges) -> int: ...


def chunk_molecule(s: Spins, c: Charges) -> int:
    return 1


def chunk_electron(s: Spins, c: Charges) -> int:
    return sum(s)


def chunk_nuclei(s: Spins, c: Charges) -> int:
    return len(c)


def chunk_nuclei_nuclei(s: Spins, c: Charges) -> int:
    return len(c) ** 2


def chunk_electron_nuclei(s: Spins, c: Charges) -> int:
    return sum(s) * len(c)


def chunk_electron_electron(s: Spins, c: Charges) -> int:
    return sum(s) ** 2


T = TypeVar('T')
Ts = TypeVarTuple('Ts')


class Systems(PyTreeNode):
    spins: tuple[Spins, ...] = field(pytree_node=False)
    charges: tuple[Charges, ...] = field(pytree_node=False)
    electrons: Electrons
    nuclei: Nuclei

    @property
    def n_elec_by_mol(self):
        return tuple(sum(s) for s in self.spins)

    @property
    def n_nuc_by_mol(self):
        return tuple(len(c) for c in self.charges)

    @property
    def n_elec(self):
        return sum(self.n_elec_by_mol)

    @property
    def n_nuc(self):
        return sum(self.n_nuc_by_mol)

    @property
    def n_ee(self):
        return sum(a**2 - a for a in self.n_elec_by_mol)

    @property
    def n_nn(self):
        return sum(a**2 for a in self.n_nuc_by_mol)

    @property
    def n_en(self):
        return sum(a * b for a, b in zip(self.n_elec_by_mol, self.n_nuc_by_mol))

    @property
    def n_mols(self):
        return len(self.spins)

    @property
    def flat_charges(self) -> npt.NDArray[np.int64]:
        return np.concatenate(self.charges)

    @property
    def spin_mask(self) -> npt.NDArray[np.int64]:
        return np.array(
            [idx for n_up, n_down in self.spins for idx in ([0] * n_up + [1] * n_down)]
        )

    @property
    def elec_elec_idx(self):
        return sort_by_same_spin(
            adj_idx(self.n_elec_by_mol, drop_diagonal=True),
            self.spins,
            drop_diagonal=True,
        )

    @property
    def elec_nuc_idx(self):
        return adj_idx(self.n_elec_by_mol, self.n_nuc_by_mol)

    @property
    def nuc_nuc_idx(self):
        return adj_idx(self.n_nuc_by_mol)

    @property
    def elec_elec_dists(self) -> ElecElecDistances:
        i, j, _ = self.elec_elec_idx
        dists = self.electrons[..., j, :] - self.electrons[..., i, :]
        return jnp.concatenate(
            [dists, jnp.linalg.norm(dists, axis=-1, keepdims=True)],
            axis=-1,
        )

    @property
    def elec_nuc_dists(self) -> ElecNucDistances:
        i, j, _ = self.elec_nuc_idx
        dists = self.electrons[..., i, :] - self.nuclei[..., j, :]
        return jnp.concatenate(
            [dists, jnp.linalg.norm(dists, axis=-1, keepdims=True)], axis=-1
        )

    @property
    def nuc_nuc_dists(self) -> NucNucDistances:
        i, j, _ = self.nuc_nuc_idx
        dists = self.nuclei[..., j, :] - self.nuclei[..., i, :]
        return jnp.concatenate(
            [dists, jnp.linalg.norm(dists, axis=-1, keepdims=True)],
            axis=-1,
        )

    @property
    def sub_configs(self):
        result: list[Systems] = []
        e_idx = n_idx = 0
        for s, c in zip(self.spins, self.charges):
            n_elec = sum(s)
            n_nuc = len(c)
            e = self.electrons[..., e_idx : e_idx + n_elec, :]
            n = self.nuclei[..., n_idx : n_idx + n_nuc, :]
            e_idx += n_elec
            n_idx += n_nuc
            result.append(Systems((s,), (c,), e, n))
        return tuple(result)

    @property
    def spins_and_charges(self):
        return tuple((s, c) for s, c in zip(self.spins, self.charges))

    @property
    def unique_spins_and_charges(self):
        return unique(self.spins_and_charges)[0]

    @property
    def unique_indices(self):
        return unique(self.spins_and_charges)[1]

    @property
    def inverse_unique_indices(self):
        _, inv_idx = np.unique(np.concatenate(self.unique_indices), return_index=True)
        return inv_idx

    @overload
    def group(
        self,
        data: T_Array,
        size_fn: ChunkSizeFunction,
        axis: int,
        return_config: Literal[True],
    ) -> Generator[tuple[T_Array, tuple[Spins, Charges]], None, None]: ...

    @overload
    def group(
        self,
        data: T_Array,
        size_fn: ChunkSizeFunction,
        axis: int = 0,
        return_config: Literal[False] = False,
    ) -> Generator[T_Array, None, None]: ...

    def group(
        self,
        data: T_Array,
        size_fn: ChunkSizeFunction,
        axis: int = 0,
        return_config: bool = False,
    ) -> (
        Generator[T_Array, None, None]
        | Generator[tuple[T_Array, tuple[Spins, Charges]], None, None]
    ):
        confs, idx, _, _ = unique(self.spins_and_charges)

        chunks = [size_fn(s, c) for s, c in zip(self.spins, self.charges)]
        offsets = np.cumsum([0] + chunks)[:-1]
        chunks = np.array(chunks)

        slice_off = (slice(None),) * axis

        for (spins, charges), m in zip(confs, idx):
            n = size_fn(spins, charges)
            slices = merge_slices(*[slice(o, o + n) for o in offsets[m]])
            if len(slices) == 1:
                result = jtu.tree_map(
                    lambda x: x[(*slice_off, slices[0])].reshape(
                        x.shape[:axis] + (len(m), n) + x.shape[axis + 1 :]
                    ),
                    data,
                )
            else:
                result = jtu.tree_map(
                    lambda x: jnp.concatenate(
                        [x[(*slice_off, s)] for s in slices], axis=axis
                    ).reshape(x.shape[:axis] + [len(m)] + x.shape[axis + 1 :]),
                    data,
                )
            if return_config:
                yield result, (spins, charges)
            else:
                yield result

    def iter_grouped_molecules(self):
        yield from zip(
            self.group(self.electrons, lambda s, c: sum(s), axis=-2),
            self.group(self.nuclei, lambda s, c: len(c), axis=-2),
            self.unique_spins_and_charges,
        )

    @property
    def n_elec_pair_same(self):
        return n_pair_same(self.spins, drop_diagonal=True)

    def elec_pair_mask(self, diag: bool):
        return elec_pair_mask(self.spins, diag=diag, drop_diagonal=True)

    @property
    def nuclei_molecule_mask(self):
        return np.repeat(np.arange(self.n_mols), self.n_nuc_by_mol)


T = TypeVar('T')
SystemSpins = Sequence[Spins] | Integer[ArrayLike, 'n_mols 2']


def sort_by_same_spin(
    pairs: T,
    spins: SystemSpins,
    drop_diagonal: bool = False,
    drop_off_block: bool = False,
) -> T:
    """
    Rearranges pairwise terms such that the block diagonals are first.

    Args:
    pairs: A 1D array representing the pairwise terms.
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    drop_diagonal: A boolean indicating whether to drop diagonal elements from the mask.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    A 1D array representing the sorted pairwise terms.
    """
    idx = np.argsort(
        ~pair_block_mask(spins, drop_diagonal, drop_off_block), kind='stable'
    )
    return jtu.tree_map(lambda x: x[idx], pairs)


def pair_graph_mask(
    spins: SystemSpins,
    diag: bool,
    drop_diagonal: bool = False,
    drop_off_block: bool = False,
) -> npt.NDArray[np.int64]:
    """
    Computes a index mask indicating for the pairwise terms to which graph they belong.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    diag: A boolean indicating whether to include diagonal elements in the mask.
    drop_diagonal: A boolean indicating whether to drop diagonal elements from the mask.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    A 1D array representing the pair graph mask.
    """
    _, _, m = adj_idx(np.sum(spins, -1))
    result = sort_by_same_spin(m, spins, drop_diagonal, drop_off_block)
    n_same = n_pair_same(spins, drop_diagonal, drop_off_block)
    if diag:
        return result[:n_same]
    else:
        return result[n_same:]


def n_pair_same(
    spins: SystemSpins,
    drop_diagonal: bool = False,
    drop_off_block: bool = False,
) -> int:
    """
    Computes the number of same-spin pairs.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    drop_diagonal: A boolean indicating whether to drop diagonal elements from the mask.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    An integer representing the number of same pairs.
    """
    spins = np.array(spins)
    n_items = spins**2
    if drop_diagonal:
        n_items -= spins
        if drop_off_block:
            n_items = n_items / 2
    elif drop_off_block:
        n_items = (n_items - spins) / 2 + spins
    return int(n_items.sum())


def n_pair_diff(spins: SystemSpins, drop_off_block: bool = False) -> int:
    """
    Computes the number of pairs with different spins.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    An integer representing the number of different pairs.
    """
    spins = np.array(spins)
    result = 2 * (spins[:, 0] * spins[:, 1]).sum()
    if drop_off_block:
        result //= 2
    return result


def pair_block_mask(
    spins: SystemSpins, drop_diagonal: bool = False, drop_off_block: bool = False
) -> npt.NDArray[np.bool]:
    """
    Computes a index mask that indicates to which block a pairwise term belongs.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    drop_diagonal: A boolean indicating whether to drop diagonal elements from the mask.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    A 1D array representing the pair block mask.
    """
    result = []
    spins = np.array(spins)
    for a, b in spins:
        mask = np.block(
            [[np.ones((a, a)), np.zeros((a, b))], [np.zeros((b, a)), np.ones((b, b))]]
        )
        if drop_diagonal:
            # set diagonal to -1
            mask -= 2 * np.eye(a + b)
        if drop_off_block:
            mask[np.tril_indices(a + b)] = -1
        mask = mask.reshape(-1)
        # Remove potential diagonal elements; also reshapes to 1D array
        result.append(mask[mask >= 0].astype(bool))
    return np.concatenate(result)


def elec_pair_mask(
    spins: SystemSpins,
    diag: bool,
    drop_diagonal: bool = False,
    drop_off_block: bool = False,
) -> npt.NDArray[np.int64]:
    """
    Compute a index mask for segment sums over pairwise terms.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    diag: A boolean indicating whether to select from the diagonal elements or offdiagonal.

    Returns:
    A 1D array representing the batched pair mask.
    """
    i, j, _ = adj_idx(
        np.sum(spins, -1), drop_diagonal=drop_diagonal, drop_off_block=drop_off_block
    )
    result = sort_by_same_spin(i, spins, drop_diagonal, drop_off_block)
    n_same = n_pair_same(spins, drop_diagonal, drop_off_block)
    if diag:
        return result[:n_same]
    else:
        return result[n_same:]
