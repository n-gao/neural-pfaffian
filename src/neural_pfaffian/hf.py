import heapq
import itertools
import logging
from typing import TYPE_CHECKING, Protocol

import einops
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pyscf
from flax.struct import PyTreeNode
from jaxtyping import Array, Float
from pyscf.scf import RHF, ROHF, UHF
from scipy.signal import argrelextrema
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KernelDensity

from neural_pfaffian.utils import itemgetter
from neural_pfaffian.utils.constants import ELEMENT_BY_ATOMIC_NUM, PERIODS
from neural_pfaffian.utils.gto import Mol

if TYPE_CHECKING:  # Avoid circular imports
    from neural_pfaffian.nn.wave_function import (
        SignedLogAmplitude,
    )
    from neural_pfaffian.systems import Systems

Electrons = Float[Array, '... n_elec 3']
HFOrbitals = tuple[Float[Array, '... n_up n_up'], Float[Array, '... n_down n_down']]
MOCoeffs = Float[Array, 'n_ao 2*n_ao'] | np.ndarray[Float, np.dtype[np.float64]]
"""Molecular orbital coefficients. Columns are MOs, rows are AOs."""
MolOrbitals = Float[Array, '... n_elec n_mo'] | np.ndarray[Float, np.dtype[np.float64]]
"""Molecular orbitals evaluated at electron positions. Columns are MOs, rows are electrons."""
MolOrbSelector = Float[Array, '... n_mo n_elec'] | np.ndarray[Float, np.dtype[np.float64]]
"""Orbital selector matrix for excited states."""
SCF = pyscf.scf.rhf.RHF | pyscf.scf.uhf.UHF | pyscf.scf.rohf.ROHF


class PretrainingTarget(PyTreeNode):
    molecular_orbitals: MolOrbitals
    molecular_orbital_permutation: MolOrbSelector


class PretrainingTargetFn(Protocol):
    def __call__(self, electrons: Electrons) -> PretrainingTarget: ...


class LogAmplitudeFn(Protocol):
    def __call__(self, electrons: Electrons) -> 'SignedLogAmplitude': ...


def make_hf_fns(
    systems: 'Systems',
    basis: str,
    *,
    n_tuple_excitation: int = 2,
    exclude_core: bool = False,
    use_smearing: bool = False,
    hf_method: str | None = None,
    minimal_spin_only: bool = False,
    inactive_orbital_truncation: int = 10,
) -> tuple[tuple[PretrainingTargetFn, ...], tuple[LogAmplitudeFn, ...]]:
    _targets: list[tuple[PretrainingTargetFn, LogAmplitudeFn]] = []
    for system in [systems[idx] for idx in systems.unique_indices]:
        # All configs in `system` belong to the same mol type (N2, LiH, etc.) and
        # have the same total spin
        # The excitations of all subconfigs must match
        # Using the nuclei positions, we build a MST to connect similar geometries
        # via a minimum spanning tree
        # As root node we pick the geometry closest to the average position
        molecules = list(system.group_molecule_ids())
        nuc_pos = np.stack([np.asarray(m.nuclei) for m in molecules], axis=0).reshape(
            len(molecules),
            -1,
        )
        dists = squareform(pdist(nuc_pos))
        mst = minimum_spanning_tree(dists)
        dag = nx.from_numpy_array(mst)
        node_heights: dict[int, float] = nx.eccentricity(dag, weight='weight')  # type: ignore
        root = min(node_heights, key=node_heights.__getitem__)
        logging.info(
            f'Using molecule index {root} as root for HF pretraining target generation.',
        )
        logging.info(str(mst))
        # We iterate over the molecules in BFS order starting from the root
        # We add the root explicitly since bfs_edges only yields edges.
        edges = [(None, root), *nx.bfs_edges(dag, source=root)]
        new_targets: dict[int, list[tuple[PretrainingTargetFn, LogAmplitudeFn]]] = {}
        previous_targets: dict[int, tuple[SCF, list[MolOrbSelector]]] = {}
        # There should be one edge per molecule
        assert len(edges) == len(molecules)
        for prev_idx, new_idx in edges:
            logging.info(f'Processing molecule index {new_idx} (from {prev_idx})')
            previous_target = previous_targets.get(prev_idx)
            molecule = molecules[new_idx]
            # Each molecule is a single config with different excitations
            n_excitations = len(molecule.sub_configs)

            pyscf_mol = molecule[0].pyscf_molecules(basis=basis)[0]
            scf_cls = _get_scf_solver(hf_method)
            mf = scf_cls(pyscf_mol)
            if previous_target is not None:
                mf.max_cycle = 0
                # Populate fields with some stuff - we anyway overwrite the MO coeffs later
                mf.kernel(dm0=previous_target[0].make_rdm1())
            else:
                mf.max_cycle = 100
                if use_smearing:
                    hf = pyscf.scf.addons.smearing_(mf, sigma=0.1, method='fermi')
                    hf.max_cycle = 5
                    hf.kernel()
                    mf.sigma = 0  # type: ignore
                    mf.max_cycle = 100
                    mf.kernel(dm0=hf.make_rdm1())
                else:
                    mf.kernel()
            logging.info(f'HF energy: {mf.e_tot}')

            excitations = _ordered_hf_excitations(
                mf,
                n_excitations,
                n_tuple_excitation,
                exclude_core=exclude_core,
                minimal_spin_only=minimal_spin_only,
                inactive_orbital_truncation=inactive_orbital_truncation,
            )

            if previous_target is not None:
                # Rotate the active and inactive orbitals to match the previous geometry as closely as possible
                mo_coeffs = _get_aligned_scf_mo_coeffs(
                    previous_target[0],
                    mf,
                )
                mf = mf.set(mo_coeff=mo_coeffs)
                excitations = previous_target[1]

            previous_targets[new_idx] = (mf, excitations)

            _log_hf_excitations(mf, excitations)

            # Save pretraining target fns
            # The lambda fn adheres to PretrainingTargetFn
            # targets is now sorted by the unique indices and not by the original batch!
            new_targets[new_idx] = [
                _make_hf_pretraining_target_fn(mf, ex) for ex in excitations
            ]
        # Collect targets in original order
        for i in range(len(molecules)):
            _targets.extend(new_targets[i])

    targets = itemgetter(*systems.inverse_unique_indices)(_targets)
    o_targets: tuple[PretrainingTargetFn, ...]
    a_targets: tuple[LogAmplitudeFn, ...]
    o_targets, a_targets = zip(*targets, strict=False)
    return o_targets, a_targets


def _get_scf_solver(hf_method: str | None):
    match hf_method.lower() if hf_method is not None else None:
        case 'rhf' | None:
            return RHF
        case 'rohf':
            return ROHF
        case 'uhf':
            return UHF
        case _:
            raise ValueError(f'Unknown hf_method: {hf_method}')


def _log_hf_excitations(
    mf: SCF,
    excitations: list[MolOrbSelector],
) -> None:
    energy = mf.mo_energy
    assert energy is not None, 'HF calculation must have converged'
    if energy.ndim == 1:  # RHF: duplicate for both spins
        energy = np.stack([energy, energy], axis=0)
    energy = energy.reshape(-1)

    gs_perm = _get_permutation_from_mf(mf)
    perms = np.stack(excitations, axis=0)
    total_energies = np.einsum('n,knm->k', energy, perms)
    ground_energy = np.einsum('n,nm->', energy, gs_perm)
    deltas = total_energies - ground_energy

    ground_occ = gs_perm.sum(-1).astype(np.int32).reshape(2, -1)
    ground_idx = [np.nonzero(ground_occ[s])[0] for s in range(2)]
    ground_maps = [{int(orb): idx for idx, orb in enumerate(idx)} for idx in ground_idx]

    def _spin_vec(exc_occ: np.ndarray, s: int) -> np.ndarray:
        occ_idx = ground_idx[s]
        vec = np.zeros(occ_idx.size, dtype=int)
        if occ_idx.size:
            exc_idx = np.nonzero(exc_occ[s])[0]
            holes = np.setdiff1d(occ_idx, exc_idx, assume_unique=True)
            particles = np.setdiff1d(exc_idx, occ_idx, assume_unique=True)
            for hole, particle in zip(holes, particles, strict=False):
                vec[ground_maps[s][int(hole)]] = int(particle - occ_idx.size + 1)
        return vec

    def _format(ex) -> str:
        exc_occ = ex.sum(-1).reshape(2, -1).astype(np.int32)
        alpha = _spin_vec(exc_occ, 0)
        beta = _spin_vec(exc_occ, 1)
        return (
            f'alpha=[{" ".join(str(val) for val in alpha.tolist())}] '
            f'beta=[{" ".join(str(val) for val in beta.tolist())}]'
        )

    logging.info(
        'Selected excitations (idx: ΔE [Ha] -> virtual index per electron):\n%s',
        '\n'.join(
            f'  {idx: 4}: {float(delta):.6f} -> {_format(ex)}'
            for idx, (delta, ex) in enumerate(zip(deltas, excitations, strict=False))
        ),
    )


def _is_minimal_spin_occupation(
    occupation: np.ndarray | Array,
    nocc: np.ndarray | Array,
) -> bool:
    occ = np.asarray(occupation).reshape(2, -1).astype(np.int32)
    nocc_arr = np.asarray(nocc).astype(np.int32)
    if nocc_arr[0] == nocc_arr[1]:
        # Ensure closed shell excitation if gs is closed shell
        return bool(np.array_equal(occ[0], occ[1]))
    if nocc_arr[0] > nocc_arr[1]:
        # If alpha > beta, we want all beta occupied orbitals to be a subset of alpha
        return bool(np.all(occ[1] <= occ[0]))
    return bool(np.all(occ[0] <= occ[1]))


def _ordered_hf_excitations(
    mf: SCF,
    n_states: int,
    n_tuple_excitation: int = 2,
    *,
    exclude_core: bool = False,
    preserve_spin: bool = True,
    inactive_orbital_truncation: int = 10,
    minimal_spin_only: bool = False,
) -> list[MolOrbSelector]:
    """Computes all excitations up to n-tuple excitations, estimates their energies and orders
    them in ascending order.

    Args:
        mf: The pyscf mean field object.
        n_states: The number of states to return
        n_tuple_excitation: Up to how many electrons to excite at once.
            Defaults to double excitations. Beware of combinatorial explosion.
        exclude_core: Whether to exclude core electrons from the excitation.
            Core electrons are determined automatically as the largest atom from the previous period.
            E.g. Mg has 10 core electrons, only its 2 valence electrons are considered for excitation.
        inactive_orbital_truncation: Upper bound for virtual orbitals to consider. Larger values
            produce more excitations but can explode combinatorics.
        minimal_spin_only: If True, only keep determinants with minimal-spin occupation
            patterns for the given M_s (closed-shell (i.e. double occupied orbitals only) for M_s=0,
            minority-spin subset (i.e. double occupied for all beta electrons) otherwise).
    """
    gs_permutation = _get_permutation_from_mf(mf)
    energy = mf.mo_energy
    assert energy is not None, 'HF calculation must have converged'

    # Deal with RHF
    if energy.ndim == 1:
        energy = np.stack([energy, energy], axis=0)
    energy = energy.reshape(-1)

    core_electrons = _get_core_electrons(mf) if exclude_core else 0

    active_idx, inactive_idx, nocc, orbital_occupation = _select_active_inactive_indices(
        gs_permutation,
        core_electrons,
        inactive_orbital_truncation,
    )
    if minimal_spin_only and not preserve_spin:
        raise ValueError('minimal_spin_only requires preserve_spin=True')
    n_orbitals_per_spin = orbital_occupation.size // 2
    occupied_orbitals = np.flatnonzero(orbital_occupation)
    identity = np.eye(orbital_occupation.size)
    if minimal_spin_only and not _is_minimal_spin_occupation(orbital_occupation, nocc):
        raise ValueError('Ground-state occupation does not match minimal-spin pattern.')

    excitations: list[
        tuple[
            float,
            int,
            tuple[tuple[int, ...], tuple[int, ...]],
        ]
    ] = []
    tie_break_counter = 0

    def maybe_add(
        energy_delta: float,
        active_combo: tuple[int, ...],
        inactive_combo: tuple[int, ...],
    ) -> None:
        if minimal_spin_only:
            ex_occ = orbital_occupation.copy()
            # Deactivate active orbitals and activate inactive orbitals for the current swap
            if active_combo:
                ex_occ[list(active_combo)] = 0
            if inactive_combo:
                ex_occ[list(inactive_combo)] = 1
            # Check if the resulting occupation matches the minimal-spin pattern
            if not _is_minimal_spin_occupation(ex_occ, nocc):
                return
        nonlocal tie_break_counter
        candidate = (-energy_delta, tie_break_counter, (active_combo, inactive_combo))
        tie_break_counter += 1
        if len(excitations) < n_states:
            heapq.heappush(excitations, candidate)
        elif candidate[0] > excitations[0][0]:
            heapq.heapreplace(excitations, candidate)

    maybe_add(0.0, (), ())

    def _spin_signature(combo: tuple[int, ...]) -> tuple[int, int] | None:
        if not preserve_spin:
            return None
        n_up = sum(idx < n_orbitals_per_spin for idx in combo)
        return (n_up, len(combo) - n_up)

    def _group_combinations(
        indices: np.ndarray,
        n: int,
    ) -> dict[tuple[int, int] | None, tuple[list[tuple[int, ...]], np.ndarray]]:
        grouped: dict[
            tuple[int, int] | None,
            list[tuple[tuple[int, ...], float]],
        ] = {}
        # Group all possible combinations of n indices by their spin signature
        for combo in itertools.combinations(indices.tolist(), n):
            signature = _spin_signature(combo)
            combo_energy = float(sum(energy[idx] for idx in combo))
            grouped.setdefault(signature, []).append((combo, combo_energy))

        # For each possible spin signature we have a list of index combinations and the
        # corresponding sum of orbital energies
        return {
            key: (
                [combo for combo, _ in value],
                np.fromiter((e for _, e in value), dtype=float, count=len(value)),
            )
            for key, value in grouped.items()
        }

    for n in range(1, n_tuple_excitation + 1):
        active_combos = _group_combinations(active_idx, n)
        inactive_combos = _group_combinations(inactive_idx, n)
        for signature, (active_combination, active_energies) in active_combos.items():
            # Throw away all combinations where spin would not be preserved
            if signature not in inactive_combos:
                continue
            inactive_combination, inactive_energies = inactive_combos[signature]
            if not active_combination or not inactive_combination:
                continue

            for inactive_combo, inactive_energy in zip(
                inactive_combination,
                inactive_energies,
                strict=False,
            ):
                # For a single combination of inactive orbitals, compute deltas for all
                # possible active orbital combinations: If we want to occupy these inactive
                # orbitals, what's the energy change for each possible source of those electrons?
                deltas = inactive_energy - active_energies
                if len(excitations) == n_states:
                    worst_delta = -excitations[0][0]
                    better_mask = deltas < worst_delta
                    if not np.any(better_mask):
                        continue
                    candidate_idx = np.nonzero(better_mask)[0]
                    candidate_count = min(n_states, candidate_idx.size)
                    if candidate_count == 0:
                        continue
                    # argpartition is an efficient way to get the smallest `candidate_count` elements
                    # without fully sorting the array
                    candidate_deltas = deltas[candidate_idx]
                    candidate_idx = candidate_idx[
                        np.argpartition(
                            candidate_deltas,
                            candidate_count - 1,
                        )[:candidate_count]
                    ]
                    # candidate idx are now indices of the smallest deltas
                else:
                    candidate_count = min(n_states, deltas.size)
                    if candidate_count == 0:
                        continue
                    candidate_idx = np.argpartition(
                        deltas,
                        candidate_count - 1,
                    )[:candidate_count]

                candidate_idx = candidate_idx[np.argsort(deltas[candidate_idx])]

                for idx in candidate_idx:
                    maybe_add(
                        float(deltas[idx]),
                        active_combination[idx],
                        inactive_combo,
                    )

    if n_states is not None and len(excitations) < n_states:
        raise ValueError(
            f'Not enough excitations found. Found {len(excitations)}, requested {n_states}.'
            'Try increasing `n_tuple_excitation`, disabling `exclude_core` or a larger basis set.',
        )

    orbital_to_col = {orb: idx for idx, orb in enumerate(occupied_orbitals)}

    def _build_permutation(
        active_combo: tuple[int, ...],
        inactive_combo: tuple[int, ...],
    ) -> MolOrbSelector:
        if not active_combo and not inactive_combo:
            return gs_permutation

        excited_orbitals = occupied_orbitals.copy()
        for src, dest in zip(active_combo, inactive_combo, strict=True):
            excited_orbitals[orbital_to_col[src]] = dest  # type: ignore
        return identity[:, excited_orbitals]

    excitations.sort(key=lambda entry: -entry[0])
    return [
        _build_permutation(active_combo, inactive_combo)
        for _, _, (active_combo, inactive_combo) in excitations
    ]


def _get_core_electrons(mf):
    core_electrons = 0
    # Logic from ferminet repo
    for atom in mf.mol.atom_charges():
        period = ELEMENT_BY_ATOMIC_NUM[atom].period
        if period > 1 and atom:
            # Largest atom from previous period
            # Everything up to valence electrons is considered core
            core_electrons += PERIODS[period - 1][-1].atomic_number
    core_electrons //= 2
    return core_electrons


def _select_active_inactive_indices(
    permutation: MolOrbSelector,
    core_electrons: int,
    inactive_orbital_truncation: int,
):
    """Returns occupied and virtual orbital indices with core and truncation filters applied."""
    orbital_occupation = permutation.sum(-1).astype(np.int32)

    # per spin channel occupation
    nocc = orbital_occupation.reshape(2, -1).sum(-1).astype(np.int32)

    active_idx = np.nonzero(orbital_occupation)[0]

    # Filter out core_electrons
    active_idx = active_idx[
        list(range(core_electrons, nocc[0]))
        + list(range(nocc[0] + core_electrons, active_idx.size))
    ]

    inactive_idx = np.nonzero(1 - orbital_occupation)[0]

    # Filter out exceedingly high excitations
    excitation_threshold = int(max(nocc) * inactive_orbital_truncation)
    n_orbs_per_channel = len(orbital_occupation) // 2
    if n_orbs_per_channel > excitation_threshold:
        inactive_idx = inactive_idx[
            np.nonzero(
                inactive_idx % n_orbs_per_channel < excitation_threshold,
            )[0]
        ]
    return active_idx, inactive_idx, nocc, orbital_occupation


def _get_permutation_from_mf(
    mf: SCF,
) -> MolOrbSelector:
    """Converts the RHF or UHF occupation vector to a permutation matrix."""
    occ = mf.mo_occ
    assert occ is not None, 'HF calculation must have converged'

    if occ.ndim == 1:
        occ = _split_rhf_occ(occ)
    occ = occ.reshape(-1)
    p = np.eye(occ.shape[0])
    return p[:, occ > 0]


def _split_rhf_occ(occ: np.ndarray) -> np.ndarray:
    """Splits the RHF occupation vector into two spin channels.
    Returns a UHF style occupation vector of shape (2, n_ao)."""
    up = np.zeros_like(occ)
    down = np.zeros_like(occ)
    up[occ >= 1] = 1
    down[occ == 2] = 1
    return np.stack([up, down])


def _procrustes_block(C1_block, S12, C2_block):
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem#Solution
    # C1_block: (n_ao1, k), C2_block: (n_ao2, k)
    M = C1_block.T @ S12 @ C2_block  # (k, k)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = Vt.T @ U.T  # (k, k) in O(k)
    return C2_block @ R  # aligned block


def _get_aligned_scf_mo_coeffs(mf_target, mf_align):
    S12 = pyscf.gto.mole.intor_cross('int1e_ovlp', mf_target.mol, mf_align.mol)
    spin_stacked_coeffs = _aligned_mo_coeffs_by_procrustes(
        mf_target,
        mf_align,
        S12,
    )  # (n_ao, 2*n_ao)
    coeffs = einops.rearrange(
        spin_stacked_coeffs,
        'ao (spin mo) -> spin ao mo',
        spin=2,
    )

    # Reshape back to RHF / UHF format
    if mf_align.mo_coeff.ndim == 3:
        # UHF
        return coeffs
    return coeffs[0]  # RHF


def _split_coeff_occ_vir(mf):
    # coeff: (spin, n_ao, n_mo_spin)
    coeff = mf.mo_coeff
    if coeff.ndim == 2:
        coeff = np.stack([coeff, coeff], axis=0)
    return coeff


def _split_by_kde(data, bandwidth=0.5):
    # Ensure data is 1D and sorted for KDE evaluation
    data = np.array(data).flatten()
    data_reshaped = data.reshape(-1, 1)

    # 1. Fit KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data_reshaped)

    # 2. Evaluate density over a range
    # We create a grid from min to max data points
    x_grid = np.linspace(
        data.min() - (bandwidth * 3),
        data.max() + (bandwidth * 3),
        4096,
    ).reshape(-1, 1)
    log_dens = kde.score_samples(x_grid)
    dens = np.exp(log_dens)

    # 3. Find boundaries (valleys/local minima)
    valleys_idx = argrelextrema(dens, np.less)[0]
    boundaries = x_grid[valleys_idx].flatten()

    # 4. Assign clusters
    # If no valleys are found, every point belongs to cluster 0
    if len(boundaries) == 0:
        return np.zeros_like(data, dtype=int)

    # Use digitize to assign points to bins defined by boundaries
    labels = np.digitize(data, boundaries)
    idx_boundaries = [0, *(np.where(np.diff(labels))[0] + 1), None]
    return [
        slice(int(s), int(e) if e is not None else None)
        for s, e in itertools.pairwise(idx_boundaries)
    ]


def _aligned_mo_coeffs_by_procrustes(mf1, mf2, S12):
    C1 = _split_coeff_occ_vir(mf1)
    C2 = _split_coeff_occ_vir(mf2)

    slices = _split_by_kde(mf1.mo_energy, 0.25)
    logging.info(f'Procrustes alignment using {len(slices)} energy blocks.')
    logging.info(str(slices))

    aligned = []
    for s in (0, 1):  # alpha, beta
        C1s, C2s = C1[s], C2[s]  # (n_ao, n_mo_spin)
        new_C = np.concatenate(
            [_procrustes_block(C1s[:, sl], S12, C2s[:, sl]) for sl in slices],
            axis=1,
        )
        aligned.append(new_C)
    C2_aligned_spin = np.stack(aligned, axis=0)  # (spin, n_ao, n_mo_spin)
    mf2.max_cycle = 0
    mf2.mo_coeff = C2_aligned_spin[0] if mf2.mo_coeff.ndim == 2 else C2_aligned_spin
    mf2.kernel(dm0=mf2.make_rdm1(mf2.mo_coeff, mf2.mo_occ))
    logging.info(f'Energy after: {mf2.e_tot}')
    return einops.rearrange(C2_aligned_spin, 'spin ao mo -> ao (spin mo)')


def _make_hf_pretraining_target_fn(
    mf: SCF,
    permutation: MolOrbSelector,
) -> tuple[PretrainingTargetFn, LogAmplitudeFn]:
    assert mf.mo_coeff is not None, 'HF calculation must have converged'
    mo_coeff = _convert_scf_mo_coeffs(mf.mo_coeff)
    n_up, _ = mf.mol.nelec
    n_ao = mo_coeff.shape[0]

    jax_mol = Mol.from_pyscf_mol(mf.mol)

    def atomic_orbitals(electrons: jnp.ndarray) -> jnp.ndarray:
        batch_shape = electrons.shape[:-1]
        ao_values = jax_mol.eval_gto(electrons.reshape(-1, 3))
        return ao_values.reshape(*batch_shape, mf.mol.nao)

    @jax.jit
    def mo_orbitals(electrons: Electrons) -> PretrainingTarget:
        ao_orbitals = atomic_orbitals(electrons).astype(electrons.dtype)
        mo_values = jnp.array(ao_orbitals @ mo_coeff, electrons.dtype)
        # Set off-diagonal elements to zero
        # (a spin-up electron shouldn't be in a spin-down orbital)
        mo_values = mo_values.at[..., :n_up, n_ao:].set(0.0)
        mo_values = mo_values.at[..., n_up:, :n_ao].set(0.0)

        return PretrainingTarget(mo_values, jnp.array(permutation, electrons.dtype))

    @jax.jit
    def logamplitude(electrons: Electrons) -> 'SignedLogAmplitude':
        target = mo_orbitals(electrons)
        hf_orbitals = target.molecular_orbitals @ target.molecular_orbital_permutation
        return jnp.linalg.slogdet(hf_orbitals)

    return mo_orbitals, logamplitude


def _convert_scf_mo_coeffs(mo_coeff: np.ndarray) -> MOCoeffs:
    if mo_coeff.ndim == 2:
        mo_coeff = np.stack([mo_coeff, mo_coeff], axis=0)
    return einops.rearrange(mo_coeff, 'spin ao mo -> ao (spin mo)')
