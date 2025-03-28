import functools
from typing import (
    Generator,
    Literal,
    Protocol,
    Self,
    Sequence,
    TypeVar,
    TypeVarTuple,
    overload,
    override,
)

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import numpy.typing as npt
import pyscf
from flax.struct import field
from jaxtyping import Array, ArrayLike, Float, Integer, PyTree

from neural_pfaffian.hf import HFOrbitalFn, make_hf_orbitals
from neural_pfaffian.utils import adj_idx, merge_slices, unique
from neural_pfaffian.utils.jax_utils import (
    BATCH_SPEC,
    REPLICATE_SPEC,
    SerializeablePyTree,
)
from neural_pfaffian.utils.tree_utils import tree_take

Electrons = Float[Array, '... n_elec 3']
Nuclei = Float[Array, '... n_nuc 3']

Spins = tuple[int, int]
Charges = tuple[int, ...]

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
S = TypeVar('S', bound='Systems')


def assign_spins_to_atoms(R: Nuclei, Z: Charges):
    Z_np = np.array(Z)
    n_el = np.sum(Z_np)

    # Assign equal nr of up and down spins to all atoms.
    # If the nuclear charge is odd, we'll redistribute the reamining spins below
    n_up_per_atom = Z_np // 2
    n_el_remaining = n_el - 2 * np.sum(n_up_per_atom)

    if n_el_remaining > 0:
        # Get the indices of the atoms with "open shells"
        ind_open_shell = np.where(Z_np % 2)[0]
        R_open = R[ind_open_shell]
        dist = np.linalg.norm(R_open[:, None, :] - R_open[None, :, :], axis=-1)
        kernel = np.exp(-dist * 0.5)

        # Loop over all remaining electrons
        spins = np.zeros(n_el_remaining)
        n_dn_left = n_el_remaining // 2
        n_up_left = n_el_remaining - n_dn_left
        for _ in range(n_el_remaining):
            is_free = spins == 0
            spin_per_site = kernel[is_free, :] @ spins

            # Compute the loss loss_i = sum_j kernel_ij * spin_j
            # and add another spin such that the loss is minimal (ie. as much anti-parallel as possible)
            ind_atom = np.arange(n_el_remaining)[is_free]
            loss_up = spin_per_site
            loss_dn = -spin_per_site
            if (n_up_left > 0) and (np.min(loss_up) < np.min(loss_dn)):
                ind = ind_atom[np.argmin(loss_up)]
                spins[ind] = 1
                n_up_left -= 1
            else:
                ind = ind_atom[np.argmin(loss_dn)]
                spins[ind] = -1
                n_dn_left -= 1

        # Add spins to the atoms with open shells
        n_up_per_atom[ind_open_shell] += spins == 1

    n_dn_per_atom = Z - n_up_per_atom
    # Collect a list of atom indices: first all up spins, then all down spins
    ind_atom = []
    for i, n_up in enumerate(n_up_per_atom):
        ind_atom += [i] * n_up
    for i, n_dn in enumerate(n_dn_per_atom):
        ind_atom += [i] * n_dn
    return np.array(ind_atom)


def init_electrons(
    key: jax.Array, nuclei: Nuclei, charges: Charges, spins: Spins, batch_size: int
) -> Electrons:
    n_el = sum(spins)
    key, subkey = jax.random.split(key)
    electrons = jax.random.normal(subkey, (batch_size, n_el, 3), dtype=jnp.float32)

    R = np.array(nuclei, dtype=jnp.float32)
    n_atoms = len(R)
    if n_atoms > 1:
        if n_el == sum(charges):
            # We can assign spins with the least stress
            ind_atom = assign_spins_to_atoms(nuclei, charges)
        else:
            # We randomly pick atoms based on their charge as probability.
            key, subkey = jax.random.split(key)
            ind_atom = np.asarray(
                jax.random.choice(
                    subkey,
                    np.arange(n_atoms),
                    shape=(batch_size, n_el),
                    p=np.array(charges) / sum(charges),
                )
            )
        electrons += R[ind_atom]
    if spins[0] - spins[1] != 0:
        # We randomly shuffle the electron which gets moved to the majority spin channel
        up_electrons = electrons[:, : n_el // 2]
        down_electrons = electrons[:, n_el // 2 :]
        key, key_up, key_dn = jax.random.split(key, 3)
        up_electrons = jax.random.permutation(key_up, up_electrons, axis=1)
        down_electrons = jax.random.permutation(key_dn, down_electrons, axis=1)
        electrons = jnp.concatenate([up_electrons, down_electrons], axis=1)
    return electrons


@functools.total_ordering
class Systems(Sequence['Systems'], SerializeablePyTree):
    spins: tuple[Spins, ...] = field(pytree_node=False)
    charges: tuple[Charges, ...] = field(pytree_node=False)
    electrons: Electrons
    nuclei: Nuclei
    mol_data: dict[str, PyTree[Float[Array, 'n_mols ...']]]

    def set_mol_data(self, key: str, data: PyTree[Float[Array, 'n_mols ...']]):
        return self.replace(mol_data=self.mol_data | {key: data})

    def get_mol_data(self, key: str) -> PyTree[Float[Array, 'n_mols ...']]:
        return self.mol_data[key]

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
            [spin for n_up, n_down in self.spins for spin in ([0] * n_up + [1] * n_down)]
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
        return tuple(self[i] for i in range(self.n_mols))

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

    @property
    def spins_are_identical(self):
        return all(s == self.spins[0] for s in self.spins)

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
        axis = axis % data.ndim
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
    def electron_molecule_mask(self):
        return np.repeat(np.arange(self.n_mols), self.n_elec_by_mol)

    @property
    def nuclei_molecule_mask(self):
        return np.repeat(np.arange(self.n_mols), self.n_nuc_by_mol)

    @property
    def molecule_vmap(self):
        return self.replace(electrons=0, nuclei=0, mol_data=0)

    @property
    def electron_vmap(self):
        return self.replace(electrons=0, nuclei=None, mol_data=None)

    @property
    def partition_spec(self):
        return self.replace(
            electrons=BATCH_SPEC,  # electons are batched per molecule
            nuclei=REPLICATE_SPEC,  # nuclei are replicated
            mol_data=REPLICATE_SPEC,  # molecule data is replicated
        )

    def pyscf_molecules(self, basis: str):
        # Only works unjitted
        return tuple(
            pyscf.gto.M(
                atom=[
                    (c, np.asarray(pos)) for c, pos in zip(mol.flat_charges, mol.nuclei)
                ],
                charge=int(sum(mol.flat_charges) - mol.n_elec),
                spin=mol.spins[0][0] - mol.spins[0][1],
                basis=basis,
                unit='bohr',
            )
            for mol in self.sub_configs
        )

    def hf_functions(self, basis: str):
        # Only works unjitted
        return tuple(make_hf_orbitals(mol) for mol in self.pyscf_molecules(basis))

    def with_hf(self, basis: str) -> 'SystemsWithHF':
        return SystemsWithHF(
            self.spins,
            self.charges,
            self.electrons,
            self.nuclei,
            self.mol_data,
            self.hf_functions(basis),
            tuple([None] * self.n_mols),
        )

    def __eq__(self, other):
        if not isinstance(other, Systems):
            return False
        # TODO: Sort first
        return self.spins == other.spins and self.charges == other.charges

    def __lt__(self, other):
        if not isinstance(other, Systems):
            return NotImplemented
        # TODO: Sort first
        if self.spins == other.spins:
            return self.charges < other.charges
        else:
            return self.spins < other.spins

    def get_nth_molecule(self, idx: int) -> Self:
        e_idx = np.cumsum((0,) + self.n_elec_by_mol)[idx]
        n_idx = np.cumsum((0,) + self.n_nuc_by_mol)[idx]
        return Systems(
            (self.spins[idx],),
            (self.charges[idx],),
            self.electrons[..., e_idx : e_idx + self.n_elec_by_mol[idx], :],
            self.nuclei[..., n_idx : n_idx + self.n_nuc_by_mol[idx], :],
            tree_take(self.mol_data, slice(idx, idx + 1), 0),
        )  # type: ignore

    def __getitem__(self, idx) -> Self:
        cls = self.__class__
        if isinstance(idx, int):
            return self.get_nth_molecule(idx)
        elif isinstance(idx, slice):
            return cls.merge([self[i] for i in range(*idx.indices(self.n_mols))])
        else:
            raise NotImplementedError

    def __len__(self):
        return self.n_mols

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            raise ValueError(f'Cannot add {self.__class__} with {type(other)}')
        return self.__class__(
            self.spins + other.spins,
            self.charges + other.charges,
            jnp.concatenate([self.electrons, other.electrons], axis=-2),
            jnp.concatenate([self.nuclei, other.nuclei], axis=-2),
            jtu.tree_map(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                self.mol_data,
                other.mol_data,
            ),
        )

    def __radd__(self, other: Self) -> Self:
        return self + other

    @classmethod
    def merge(cls, systems: Sequence[Self]) -> Self:
        # each system is now a single molecule
        flat_systems = [s for sys in systems for s in sys.sub_configs]
        flat_systems = sorted(flat_systems)
        return functools.reduce(cls.__add__, flat_systems)

    @classmethod
    def from_pyscf(cls, mol: pyscf.gto.Mole) -> Self:
        nuclei = jnp.array(mol.atom_coords(), dtype=jnp.float32)
        charges = tuple(int(c) for c in mol.atom_charges())
        spins = mol.nelec
        return cls.create(spins, charges, nuclei)

    @classmethod
    def create(
        cls,
        spins: Spins,
        charges: Charges,
        nuclei: Nuclei,
    ) -> Self:
        n_elec = np.sum(spins)
        electrons = jnp.zeros((n_elec, 3), dtype=jnp.float32)
        return cls((spins,), (charges,), electrons, nuclei, {})

    @property
    def example_input(self):
        # A dummy input without any batch dimensions
        return Systems(
            self.spins,
            self.charges,
            jnp.zeros((self.n_elec, 3), dtype=self.electrons.dtype),
            self.nuclei,
            self.mol_data,
        )

    def init_electrons(self, key: jax.Array, batch_size: int) -> Self:
        electrons = []
        for s in self:
            electrons.append(
                init_electrons(key, s.nuclei, s.charges[0], s.spins[0], batch_size)
            )
        return self.replace(electrons=jnp.concatenate(electrons, axis=-2))


class SystemsWithHF(Systems):
    hf_functions: tuple[HFOrbitalFn, ...] = field(pytree_node=False)
    cache: tuple[PyTree[Array], ...]

    @property
    def to_systems(self):
        return Systems(
            self.spins, self.charges, self.electrons, self.nuclei, self.mol_data
        )

    @property
    def hf_orbitals(self):
        return tuple(
            hf_fn(sys.electrons)
            for hf_fn, sys in zip(self.hf_functions, self.sub_configs)
        )

    @property
    def molecule_vmap(self):
        return super().molecule_vmap.replace(cache=0)

    @property
    def electron_vmap(self):
        return super().electron_vmap.replace(cache=None)

    @property
    def partition_spec(self):
        return super().partition_spec.replace(cache=REPLICATE_SPEC)

    @override
    def get_nth_molecule(self, idx: int) -> Self:
        return SystemsWithHF(
            **vars(super().get_nth_molecule(idx)),
            hf_functions=(self.hf_functions[idx],),
            cache=(self.cache[idx],),
        )  # type: ignore

    @override
    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'Cannot add {self.__class__} with {type(other)}')

        return SystemsWithHF(
            **vars(self.to_systems + other.to_systems),
            hf_functions=self.hf_functions + other.hf_functions,
            cache=self.cache + other.cache,
        )


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
