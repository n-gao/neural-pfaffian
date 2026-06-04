import functools
from collections import Counter
from collections.abc import Callable, Generator, Sequence
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Protocol,
    Self,
    TypeVar,
    TypeVarTuple,
    overload,
    override,
)

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pyscf
from flax.struct import field
from jaxtyping import Array, ArrayLike, Float, Integer, PyTree
from pyscf.lib.logger import WARNING as PYSCF_WARNING

from neural_pfaffian.hf import (
    HFOrbitals,
    PretrainingTarget,
    PretrainingTargetFn,
    make_hf_fns,
)
from neural_pfaffian.utils import adj_idx, batch, itemgetter, merge_slices, unique
from neural_pfaffian.utils.constants import ELEMENT_BY_ATOMIC_NUM
from neural_pfaffian.utils.jax_utils import (
    BATCH_SPEC,
    REPLICATE_SPEC,
    SerializeablePyTree,
)
from neural_pfaffian.utils.tree_utils import tree_take

if TYPE_CHECKING:  # Avoid circular imports
    from neural_pfaffian.nn.wave_function import (
        SignedLogAmplitude,
    )
Electrons = Float[Array, '... n_elec 3']
Nuclei = Float[Array, '... n_nuc 3']

Spins = tuple[int, int]
Charges = tuple[int, ...]

T_Array = TypeVar('T_Array', bound=Array)
ElecElecDistances = Float[Array, 'electrons_electrons 4']
ElecNucDistances = Float[Array, 'electrons_nuclei 4']
NucNucDistances = Float[Array, 'nuclei_nuclei 4']


class PseudopotentialProperties(NamedTuple):
    r_grid: np.ndarray
    v_loc: np.ndarray
    v_nonloc: np.ndarray
    label: str | None = None

    def __eq__(self, other):
        if not isinstance(other, PseudopotentialProperties):
            return NotImplemented
        return (
            np.array_equal(self.r_grid, other.r_grid)
            and np.array_equal(self.v_loc, other.v_loc)
            and np.array_equal(self.v_nonloc, other.v_nonloc)
            and self.label == other.label
        )

    def __hash__(self):
        return hash(
            (
                self.r_grid.tobytes(),
                self.v_loc.tobytes(),
                self.v_nonloc.tobytes(),
                self.label,
            ),
        )

    @classmethod
    def create_empty(cls, n_nuc: int, dtype) -> Self:
        np_dtype = np.dtype(dtype)
        r_grid = np.array([0.0], dtype=np_dtype)
        v_loc = np.zeros((n_nuc, 1), dtype=np_dtype)
        v_nonloc = np.zeros((n_nuc, 0, 1), dtype=np_dtype)
        return cls(r_grid, v_loc, v_nonloc, None)


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
    key: jax.Array,
    nuclei: Nuclei,
    charges: Charges,
    spins: Spins,
    batch_size: int,
    init_width: float,
) -> Electrons:
    n_el = sum(spins)
    key, subkey = jax.random.split(key)
    electrons = (
        jax.random.normal(subkey, (batch_size, n_el, 3), dtype=jnp.float32) * init_width
    )

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
                ),
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
    mol_ids: tuple[int, ...] = field(pytree_node=False)
    """Uniquely identifies a molecule. Used to unite all excited states of a molecule."""
    excitations: tuple[int, ...] = field(pytree_node=False)
    """Excitation index. Discerns between different excited states of a molecule."""
    effective_charges: tuple[Charges, ...] = field(pytree_node=False)
    """Effective nuclear charges per molecule after pseudopotential projection."""
    pp_data: tuple[PseudopotentialProperties, ...] = field(pytree_node=False)
    """Per-molecule pseudopotential tables (radial grids, local/nonlocal channels)."""

    def set_mol_data(self, key: str, data: PyTree[Float[Array, 'n_mols ...']]):
        return self.replace(mol_data=self.mol_data | {key: data})

    def get_mol_data(self, key: str) -> PyTree[Float[Array, 'n_mols ...']]:
        return self.mol_data[key]

    @property
    def n_elec_by_mol(self):
        return tuple(sum(s) for s in self.spins)

    @property
    def n_up_by_mol(self):
        return tuple(s[0] for s in self.spins)

    @property
    def n_down_by_mol(self):
        return tuple(s[1] for s in self.spins)

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
        return sum(
            a * b for a, b in zip(self.n_elec_by_mol, self.n_nuc_by_mol, strict=False)
        )

    @property
    def n_mols(self):
        return len(self.spins)

    @property
    def flat_charges(self) -> npt.NDArray[np.int64]:
        return np.concatenate(self.charges)

    @property
    def max_num_states(self):
        return max(self.excitations) + 1

    @property
    def n_unique_mols(self):
        """Different mols can describe different states of the same molecule.
        This returns the number of unique molecules.
        `system.n_mols` = `system.n_unique_mols` * `max_num_states`"""
        return len(set(self.mol_ids))

    @property
    def mol_id_groups(self) -> np.ndarray:
        """Dense group ids for mol_ids in order of first appearance."""
        return unique(self.mol_ids)[2]

    @property
    def has_pseudopotentials(self) -> bool:
        """Return True if effective charges differ from bare charges."""
        return not np.array_equal(self.flat_charges, self.flat_effective_charges)

    @property
    def flat_effective_charges(self) -> npt.NDArray[np.int64]:
        return np.concatenate(self.effective_charges)

    @property
    def ecp_mask(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of the same shape as `self.flat_charges` indicating
        whether a pseudopotential is applied to the respective nucleus."""
        if not self.has_pseudopotentials:
            return np.zeros_like(self.flat_charges, dtype=bool)
        return np.abs(self.flat_charges - self.flat_effective_charges) > 1e-6

    @property
    def spin_mask(self) -> npt.NDArray[np.int64]:
        return np.array(
            [spin for n_up, n_down in self.spins for spin in ([0] * n_up + [1] * n_down)],
        )

    @property
    def total_spins(self):
        return tuple(s[0] - s[1] for s in self.spins)

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
            [dists, jnp.linalg.norm(dists, axis=-1, keepdims=True)],
            axis=-1,
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
        return tuple((s, c) for s, c in zip(self.spins, self.charges, strict=True))

    @property
    def spins_and_charges_and_excitations(self):
        return tuple(
            (s, c, e)
            for s, c, e in zip(self.spins, self.charges, self.excitations, strict=True)
        )

    @property
    def unique_spins_and_charges(self):
        return unique(self.spins_and_charges)[0]

    @property
    def unique_indices(self):
        return unique(self.spins_and_charges)[1]

    @property
    def unique_systems(self):
        ids = [idx[0] for idx in unique(self.mol_ids)[1]]
        return self[ids]

    @property
    def unique_systems_indices(self):
        return unique(self.mol_ids)[1]

    @property
    def inverse_unique_systems_indices(self):
        return np.argsort(np.concatenate(self.unique_systems_indices), stable=True)

    def replicate_unique_array_per_mol(self, x: Array, chunk_fn: ChunkSizeFunction):
        unique_sys = self.unique_systems
        indices = []
        offset = 0
        for spins, charges in zip(unique_sys.spins, unique_sys.charges, strict=True):
            n = chunk_fn(spins, charges)
            indices.append(np.arange(offset, offset + n))
            offset += n
        all_indices = np.concatenate(
            itemgetter(*self.inverse_unique_systems_indices)(
                [
                    idx
                    for idx, sys_idx in zip(
                        indices,
                        self.unique_systems_indices,
                        strict=True,
                    )
                    for _ in range(len(sys_idx))
                ],
            ),
        )
        return x[all_indices]

    @property
    def inverse_unique_indices(self):
        _, inv_idx = np.unique(
            np.concatenate(self.unique_indices),
            return_index=True,
        )
        return inv_idx

    @property
    def spins_are_identical(self):
        return all(s == self.spins[0] for s in self.spins)

    @property
    def grouped_excitations(self) -> Generator[Integer[Array, ' n_groups'], None, None]:
        """Applies the same grouping as `Systems.group` to the excitations."""
        for excitation in self.group(jnp.array(self.excitations), lambda s, c: 1):
            yield excitation.squeeze(-1).astype(jnp.int32)

    def group_molecule_ids(self) -> Generator['Systems', None, None]:
        """Returns a generator of Systems grouped by their molecule ids.
        Each group contains only systems with the same molecule id."""
        for indices in unique(self.mol_ids)[1]:
            yield self[indices]

    def set_global_excitation(self, excitation: int):
        return self.replace(excitations=(excitation,) * self.n_mols)

    @overload
    def group(
        self,
        data: T_Array,
        size_fn: ChunkSizeFunction,
        axis: int,
        *,
        return_config: Literal[True],
        include_excitation: Literal[False] = False,
    ) -> Generator[tuple[T_Array, tuple[Spins, Charges]], None, None]: ...

    @overload
    def group(
        self,
        data: T_Array,
        size_fn: ChunkSizeFunction,
        axis: int,
        *,
        return_config: Literal[True],
        include_excitation: Literal[True],
    ) -> Generator[tuple[T_Array, tuple[Spins, Charges, int]], None, None]: ...

    @overload
    def group(
        self,
        data: T_Array,
        size_fn: ChunkSizeFunction,
        axis: int = 0,
        *,
        return_config: Literal[False] = False,
        include_excitation: bool = False,
    ) -> Generator[T_Array, None, None]: ...

    def group(
        self,
        data: T_Array,
        size_fn: ChunkSizeFunction,
        axis: int = 0,
        *,
        return_config: bool = False,
        include_excitation: bool = False,
    ) -> Generator[
        T_Array | tuple[T_Array, tuple[Spins, Charges] | tuple[Spins, Charges, int]],
        None,
        None,
    ]:
        """
        Group `data` by unique (Spins, Charges) or (Spins, Charges, Excitation) configurations.

        This method partitions the input `data` (an array or PyTree of arrays) along
        the specified `axis` according to unique spin-charge configurations (or
        spin-charge-excitation configurations if `include_excitation=True`).
        It produces slices of `data` that correspond to each unique configuration.

        Args:
            data (T_Array):
                An array or a PyTree of arrays to be grouped.

            size_fn (ChunkSizeFunction):
                A function that takes `(Spins, Charges)` and returns the
                chunk size (for example, the number of electrons to group).

            axis (int, optional):
                The axis along which to slice the data. Defaults to 0.

            return_config (bool, optional):
                If `True`, the generator yields `(group_data, config)` tuples,
                where `config` is `(Spins, Charges)` or `(Spins, Charges, Excitation)`.
                If `False`, only `group_data` is yielded. Defaults to False.

            include_excitation (bool, optional):
                If `True`, grouping is performed using `(Spins, Charges, Excitation)`.
                If `False`, grouping is performed using `(Spins, Charges)`. Defaults
                to False.

        Yields:
            Union[T_Array, tuple[T_Array, tuple[Spins, Charges]],
            tuple[T_Array, tuple[Spins, Charges, int]]]:
                - If `return_config=False`, yields only the grouped data (array or PyTree).
                - If `return_config=True` and `include_excitation=False`,
                  yields `(group_data, (spins, charges))`.
                - If `return_config=True` and `include_excitation=True`,
                  yields `(group_data, (spins, charges, excitation))`.
        """
        axis = axis % data.ndim
        confs, idx, _, _ = (
            unique(self.spins_and_charges)
            if not include_excitation
            else unique(self.spins_and_charges_and_excitations)
        )

        chunks = [size_fn(s, c) for s, c in zip(self.spins, self.charges, strict=False)]
        offsets = np.cumsum([0, *chunks])[:-1]
        chunks = np.array(chunks)

        slice_off = (slice(None),) * axis

        for conf, m in zip(confs, idx, strict=False):
            spins, charges = conf[:2]
            n = size_fn(spins, charges)
            slices = merge_slices(*[slice(o, o + n) for o in offsets[m]])
            if len(slices) == 1:
                result = jax.tree.map(
                    lambda x, slices=slices, m=m, n=n: x[(*slice_off, slices[0])].reshape(
                        (*x.shape[:axis], len(m), n, *x.shape[axis + 1 :]),
                    ),
                    data,
                )
            else:
                result = jax.tree.map(
                    lambda x, slices=slices, m=m: jnp.concatenate(
                        [x[(*slice_off, s)] for s in slices],
                        axis=axis,
                    ).reshape([*x.shape[:axis], len(m), *x.shape[axis + 1 :]]),
                    data,
                )
            if return_config:
                yield (result, conf)
            else:
                yield result

    @overload
    def iter_grouped_molecules(
        self,
        *,
        include_excitation: Literal[False] = False,
    ) -> Generator[
        tuple[
            Float[Array, 'chunk_size ...'],
            Float[Array, 'chunk_size ...'],
            tuple[Spins, Charges],
        ],
        None,
        None,
    ]: ...

    @overload
    def iter_grouped_molecules(
        self,
        *,
        include_excitation: Literal[True],
    ) -> Generator[
        tuple[
            Float[Array, 'chunk_size ...'],
            Float[Array, 'chunk_size ...'],
            tuple[Spins, Charges, int],
        ],
        None,
        None,
    ]: ...

    def iter_grouped_molecules(
        self,
        *,
        include_excitation: bool = False,
    ) -> Generator[
        tuple[
            Float[Array, 'chunk_size ...'],
            Float[Array, 'chunk_size ...'],
            tuple[Spins, Charges] | tuple[Spins, Charges, int],
        ],
        None,
        None,
    ]:
        """
        Iterate over unique spin-charge (and optionally excitation) configurations,
        returning grouped electrons and nuclei arrays for each configuration.

        This method effectively organizes the system into groups such that each group
        contains all molecules with the same `(Spins, Charges)` or `(Spins, Charges, Excitation)`.
        For each group, it yields a 3-tuple containing:

        1. A chunked subset of `electrons` corresponding to the configuration.
        2. A chunked subset of `nuclei` corresponding to the configuration.
        3. The unique configuration itself.

        Args:
            include_excitation (bool, optional):
                If `True`, group by `(Spins, Charges, Excitation)`.
                Otherwise, group by `(Spins, Charges)`. Defaults to `False`.

        Yields:
            tuple:
                A 3-tuple `(electrons_chunk, nuclei_chunk, config)` where:

                1. `electrons_chunk`: A chunked section of the electrons array,
                  grouped along `axis=-2`.

                2. `nuclei_chunk`: A chunked section of the nuclei array,
                  grouped along `axis=-2`.

                3. `config`: Either `(Spins, Charges)` or `(Spins, Charges, Excitation)`,
                  depending on `include_excitation`.
        """
        system_props = (
            self.spins_and_charges
            if not include_excitation
            else self.spins_and_charges_and_excitations
        )  # properties to group by
        yield from zip(
            self.group(
                self.electrons,
                chunk_electron,
                axis=-2,
                include_excitation=include_excitation,
            ),
            self.group(
                self.nuclei,
                chunk_nuclei,
                axis=-2,
                include_excitation=include_excitation,
            ),
            unique(system_props)[0],
            strict=False,
        )

    def iter_stacked_sub_systems(self) -> Generator['Systems', None, None]:
        """Yield `Systems` grouped by unique (spins, charges) with stacked data.

        The returned `Systems` each represent a single spin/charge confguration
        (`n_mols == 1`) but the electron and nuclei arrays carry an additional leading
        dimension enumerating the individual molecules that share that configuration.
        This is useful for vmapping computation over *identical molecules with different geometries*.

        `excitations` and `mol_ids` are preserved per group, i.e., each returned `Systems`
        contains all excited states of the corresponding geometry.
        **Caution: `len(system.excitations) != system.n_mols` in the yielded systems!**

        Similar to spins and charges, `effective_charges` and `pp_data` are identical
        across all molecules in a group, so only the first entry is kept.

        **Caution: `mol_data` is dropped completely from the yielded `Systems`.
        Do not use this method if `mol_data` is required.**
        """

        electrons_grouped = self.group(self.electrons, chunk_electron, axis=-2)
        nuclei_grouped = self.group(self.nuclei, chunk_nuclei, axis=-2)
        configs, config_indices, _, _ = unique(self.spins_and_charges)

        for (
            electrons_chunk,
            nuclei_chunk,
            (spins, charges),
            group_indices,
        ) in zip(electrons_grouped, nuclei_grouped, configs, config_indices, strict=True):
            getter = itemgetter(*group_indices)
            excitations = getter(self.excitations)
            mol_ids = getter(self.mol_ids)
            # All molecules in a group share the same effective charges and pp_data
            effective_charges = self.effective_charges[group_indices[0]]
            pp_data = self.pp_data[group_indices[0]]
            yield Systems(
                (spins,),
                (charges,),
                electrons_chunk,
                nuclei_chunk,
                {},  # HACK: `mol_data` is simply dropped
                mol_ids,
                excitations,
                (effective_charges,),
                (pp_data,),
            )

    @property
    def split_by_spin(self):
        up_idx = np.where(self.spin_mask == 0)[0]
        down_idx = np.where(self.spin_mask == 1)[0]
        up_elecs = self.electrons[..., up_idx, :]
        down_elecs = self.electrons[..., down_idx, :]
        new_up_spins = tuple((s[0], 0) for s in self.spins)
        new_down_spins = tuple((0, s[1]) for s in self.spins)
        return (
            self.replace(
                electrons=up_elecs,
                spins=new_up_spins,
            ),
            self.replace(
                electrons=down_elecs,
                spins=new_down_spins,
            ),
            up_idx,
            down_idx,
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
    def partition_spec(self) -> Self:
        return self.replace(
            electrons=BATCH_SPEC,  # electons are batched per molecule
            nuclei=REPLICATE_SPEC,  # nuclei are replicated
            mol_data=REPLICATE_SPEC,  # molecule data is replicated
        )

    def pyscf_molecules(self, basis: str):
        # Only works unjitted

        def _ecp_arg(mol: Systems):
            if not mol.has_pseudopotentials:
                return None
            label = mol.pp_data[0].label
            mask = mol.ecp_mask
            symbols = [ELEMENT_BY_ATOMIC_NUM[z].symbol for z in mol.flat_charges]
            return {sym: label for sym, use in zip(symbols, mask, strict=True) if use}

        return tuple(
            pyscf.gto.M(
                atom=[
                    (c, np.asarray(pos))
                    for c, pos in zip(mol.flat_charges, mol.nuclei, strict=True)
                ],
                charge=int(sum(mol.flat_effective_charges) - mol.n_elec),
                spin=mol.spins[0][0] - mol.spins[0][1],
                basis=basis,
                ecp=_ecp_arg(mol),
                unit='bohr',
                verbose=PYSCF_WARNING,
            )
            for mol in self.sub_configs
        )

    # TODO: Add options to configure pretraining target factory
    def with_hf(self, basis: str, **kwargs) -> 'SystemsWithPretrainTarget':
        return SystemsWithPretrainTarget(
            self.spins,
            self.charges,
            self.electrons,
            self.nuclei,
            self.mol_data,
            self.mol_ids,
            self.excitations,
            self.effective_charges,
            self.pp_data,
            *make_hf_fns(self, basis, **kwargs),
            tuple([None] * self.n_mols),
        )

    @property
    def _sort_key(self):
        return tuple(
            zip(
                self.spins,
                (tuple(sorted(c)) for c in self.charges),
                self.mol_ids,
                self.excitations,
                strict=True,
            ),
        )

    def __eq__(self, other):
        if not isinstance(other, Systems):
            return False
        # TODO: At the moment two mols with charges (1, 2) and (2, 1) are considered
        # equal. This could be dangerous if we relied on __eq__ for grouping molecules
        # into "computations", but desired behavior for sorting.
        # Not a problem rn, but a footgun waiting to go off in the future.
        # Charges should probably be forced into some canonical order.
        return self._sort_key == other._sort_key

    def __lt__(self, other):
        if not isinstance(other, Systems):
            return NotImplemented
        if self.n_mols != 1 or other.n_mols != 1:
            # TODO: Figure out how we want to handle ordering of multi-mol systems.
            raise NotImplementedError(
                'Comparison operators are only implemented for single-molecule Systems.',
            )
        return self._sort_key < other._sort_key

    def get_nth_molecule(self, idx: int) -> Self:
        e_idx = np.cumsum((0, *self.n_elec_by_mol))[idx]
        n_idx = np.cumsum((0, *self.n_nuc_by_mol))[idx]
        return Systems(
            (self.spins[idx],),
            (self.charges[idx],),
            self.electrons[..., e_idx : e_idx + self.n_elec_by_mol[idx], :],
            self.nuclei[..., n_idx : n_idx + self.n_nuc_by_mol[idx], :],
            tree_take(self.mol_data, slice(idx, idx + 1), 0),
            (self.mol_ids[idx],),
            (self.excitations[idx],),
            (self.effective_charges[idx],),
            (self.pp_data[idx],),
        )  # type: ignore

    def __getitem__(self, idx) -> Self:
        cls = self.__class__
        if isinstance(idx, int):
            return self.get_nth_molecule(idx)
        if isinstance(idx, ArrayLike | list):
            idx = np.asarray(idx)
            if not np.issubdtype(idx.dtype, np.integer):
                raise NotImplementedError(
                    'Indexing with non-integer arrays is not supported.',
                )
            if idx.ndim == 0:
                return self.get_nth_molecule(idx.item())
            if idx.ndim == 1:
                return cls.merge([self[i] for i in idx.tolist()], sort=False)
            raise NotImplementedError(
                'Indexing with arrays of dimension > 1 is not supported.',
            )
        if isinstance(idx, slice):
            return cls.merge(
                [self[i] for i in range(*idx.indices(self.n_mols))],
                sort=False,
            )
        raise NotImplementedError(f'Indexing with {type(idx)} is not supported.')

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
            jax.tree.map(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                self.mol_data,
                other.mol_data,
            ),
            self.mol_ids + other.mol_ids,
            self.excitations + other.excitations,
            self.effective_charges + other.effective_charges,
            self.pp_data + other.pp_data,
        )

    def __radd__(self, other: Self) -> Self:
        return self + other

    def __str__(self):
        """Gives a name to the systems. Name should include:
        - maximum excitation
        - whether it represents a PES
        - whether there are multiple structures
        - if single structure, the molecule name
        """
        name = ''
        # Excitation
        if self.max_num_states > 1:
            name += f'{self.max_num_states}ex_'
        # Different molecules
        if len(set(self.charges)) > 1:
            name += 'multi_'
            smallest_charge = min(self.flat_charges)
            largest_charge = max(self.flat_charges)
            name += f'{ELEMENT_BY_ATOMIC_NUM[smallest_charge].symbol}-{ELEMENT_BY_ATOMIC_NUM[largest_charge].symbol}_'
        else:
            charge_counts = Counter(self.charges[0])
            for charge, count in charge_counts.items():
                name += f'{count}{ELEMENT_BY_ATOMIC_NUM[charge].symbol}'
            name += '_'
            if (n_geometries := len(set(self.mol_ids))) > 1:
                name += f'{n_geometries}PES_'

        return name[:-1]

    def __repr__(self) -> str:
        if self.n_mols == 1:
            return (
                f'Systems(spins={self.spins[0]}, charges={self.charges[0]}, '
                f'excitation={self.excitations[0]}, mol_id={self.mol_ids[0]}, '
                f'electrons={self.electrons.shape}, nuclei={self.nuclei.shape})'
            )
        unique_mols = Counter(self.spins_and_charges)
        spin_charge_str = ', '.join(
            f'({count} x ({spins}, {charges}))'
            for (spins, charges), count in unique_mols.items()
        )
        return (
            f'Systems(n_mols={self.n_mols}, max_num_states={self.max_num_states}, '
            f'spins_and_charges={spin_charge_str}, electrons={self.electrons.shape}, nuclei={self.nuclei.shape})'
        )

    @classmethod
    def merge(cls, systems: Sequence[Self], *, sort: bool = True) -> Self:
        # TODO: sort defaults to true for backward compatibility. This should be changed.
        # each system is now a single molecule
        flat_systems = [s for sys in systems for s in sys.sub_configs]
        if sort:
            flat_systems = sorted(flat_systems)
        return functools.reduce(cls.__add__, flat_systems)

    def sorted(self) -> Self:
        """Return a copy with subsystems sorted by spins, charges, mol_ids, and excitation."""
        return self.merge(self.sub_configs, sort=True)

    @classmethod
    def safe_batch(cls, systems: 'Systems', n: int) -> list['Systems']:
        """Batches the systems into chunks of size n.
        Takes precaution not to separate identical mol_ids into different batches."""
        batches = batch(systems, n)
        # Check if different batches contain the same mol_ids
        if (
            set.intersection(*[set(batch.mol_ids) for batch in batches])  # type: ignore
            and len(batches) > 1
        ):
            raise ValueError(
                'Batching would separate excited states of the same molecule.'
                f'Please adjust the batch size {n} to be divisible by '
                f'{systems.max_num_states}.',
            )
        return batches  # type: ignore

    @classmethod
    def from_pyscf(cls, mol: pyscf.gto.Mole) -> Self:
        nuclei = jnp.array(mol.atom_coords(), dtype=jnp.float32)
        charges = tuple(mol.atom_charges().tolist())
        spins = mol.nelec
        return cls.create(spins, charges, nuclei)

    @classmethod
    def create(
        cls,
        spins: Spins,
        charges: Charges,
        nuclei: Nuclei,
        mol_ids: tuple[int, ...] = (0,),
        excitations: tuple[int, ...] = (0,),
        effective_charges: Charges | None = None,
        pp_data: PseudopotentialProperties | None = None,
    ) -> Self:
        n_elec = np.sum(spins)
        electrons = jnp.zeros((n_elec, 3), dtype=jnp.float32)
        eff_charges = (charges if effective_charges is None else effective_charges,)
        pp_props = (
            pp_data
            if pp_data is not None
            else PseudopotentialProperties.create_empty(len(charges), electrons.dtype),
        )
        return cls(
            (spins,),
            (charges,),
            electrons,
            nuclei,
            {},
            mol_ids,
            excitations,
            eff_charges,
            pp_props,
        )

    @property
    def example_input(self):
        # A dummy input without any batch dimensions
        return self.replace(
            electrons=jnp.zeros((self.n_elec, 3), dtype=self.electrons.dtype),
        )

    def init_electrons(
        self,
        key: jax.Array,
        batch_size: int,
        init_width: float,
    ) -> Self:
        electrons = []
        for s in self:
            key, subkey = jax.random.split(key)
            electrons.append(
                init_electrons(
                    subkey,
                    s.nuclei,
                    s.effective_charges[0],
                    s.spins[0],
                    batch_size,
                    init_width,
                ),
            )
        return self.replace(electrons=jnp.concatenate(electrons, axis=-2))

    def update_batch_size(
        self,
        key: jax.Array,
        num_walker_per_mol: int,
    ):
        """Change the batch size of the system by creating additional walkers."""
        # Let's make sure nothing's sharded
        jax.device_get(self)

        current_batch_size = self.electrons.shape[0]
        if current_batch_size == num_walker_per_mol:
            return self

        def _update_number_of_walkers(
            key: jax.Array,
            system: Systems,
        ):
            if current_batch_size < num_walker_per_mol:
                # Create new walkers
                key, subkey = jax.random.split(key)
                walker_indices = jax.random.choice(
                    subkey,
                    current_batch_size,
                    (num_walker_per_mol,),
                )
                return system.electrons[walker_indices]
            return system.electrons[:num_walker_per_mol]

        electrons = []
        for s in self:
            key, subkey = jax.random.split(key)
            electrons.append(_update_number_of_walkers(subkey, s))

        new_electrons = jnp.concatenate(electrons, axis=-2)
        return self.replace(electrons=new_electrons)


class SystemsWithPretrainTarget(Systems):
    target_fns: tuple[PretrainingTargetFn, ...] = field(pytree_node=False)
    logamplitude_fns: tuple[
        Callable[[Electrons], 'SignedLogAmplitude'],
        ...,
    ] = field(
        pytree_node=False,
    )
    cache: tuple[PyTree[Array], ...]
    """Saves the pretraining state of each molecule."""

    @property
    def to_systems(self):
        return Systems(
            self.spins,
            self.charges,
            self.electrons,
            self.nuclei,
            self.mol_data,
            self.mol_ids,
            self.excitations,
            self.effective_charges,
            self.pp_data,
        )

    @property
    def targets(self) -> tuple[PretrainingTarget, ...]:
        return tuple(
            hf_fn(sys.electrons)
            for hf_fn, sys in zip(self.target_fns, self.sub_configs, strict=False)
        )

    @property
    def slogamplitudes(self) -> 'SignedLogAmplitude':
        slogamps = [
            fn(sys.electrons)
            for fn, sys in zip(
                self.logamplitude_fns,
                self.sub_configs,
                strict=False,
            )
        ]
        return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *slogamps)

    @property
    def hf_orbitals(self) -> tuple[HFOrbitals, ...]:
        hf_orbitals = []
        for t, (nup, _) in zip(self.targets, self.spins, strict=False):
            m = t.molecular_orbitals @ t.molecular_orbital_permutation
            hf_orbitals.append((m[..., :nup, :nup], m[..., nup:, nup:]))
        return tuple(hf_orbitals)

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
        return SystemsWithPretrainTarget(
            **vars(super().get_nth_molecule(idx)),
            target_fns=(self.target_fns[idx],),
            logamplitude_fns=(self.logamplitude_fns[idx],),
            cache=(self.cache[idx],),
        )  # type: ignore

    @override
    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'Cannot add {self.__class__} with {type(other)}')

        return SystemsWithPretrainTarget(
            **vars(self.to_systems + other.to_systems),
            target_fns=self.target_fns + other.target_fns,
            logamplitude_fns=self.logamplitude_fns + other.logamplitude_fns,
            cache=self.cache + other.cache,
        )


T = TypeVar('T')
SystemSpins = Sequence[Spins] | Integer[ArrayLike, 'n_mols 2']


def sort_by_same_spin[T](
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
        ~pair_block_mask(spins, drop_diagonal, drop_off_block),
        kind='stable',
    )
    return jax.tree.map(lambda x: x[idx], pairs)


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
    spins: SystemSpins,
    drop_diagonal: bool = False,
    drop_off_block: bool = False,
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
            [[np.ones((a, a)), np.zeros((a, b))], [np.zeros((b, a)), np.ones((b, b))]],
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
    i, _j, _ = adj_idx(
        np.sum(spins, -1),
        drop_diagonal=drop_diagonal,
        drop_off_block=drop_off_block,
    )
    result = sort_by_same_spin(i, spins, drop_diagonal, drop_off_block)
    n_same = n_pair_same(spins, drop_diagonal, drop_off_block)
    if diag:
        return result[:n_same]
    return result[n_same:]
