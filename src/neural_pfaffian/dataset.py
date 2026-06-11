import functools
import inspect
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pyscf
import yaml

from neural_pfaffian.systems import Systems

_src_dir = Path(__file__).parent


def atomic(charge: int, spin: int | None = None, unit: str = 'bohr', **kwargs):
    return Systems.from_pyscf(
        pyscf.gto.M(atom=[(charge, (0, 0, 0))], unit=unit, spin=spin, **kwargs),
    )


def diatomic(
    charge1: int,
    charge2: int,
    distance: float,
    spin: int | None = None,
    unit: str = 'bohr',
    **kwargs,
):
    return Systems.from_pyscf(
        pyscf.gto.M(
            atom=[(charge1, (0, 0, -distance / 2)), (charge2, (0, 0, distance / 2))],
            unit=unit,
            spin=spin,
            **kwargs,
        ),
    )


def polyatomic(
    charges: Sequence[int],
    positions: Sequence[tuple[float, float, float]],
    spin: int | None = None,
    unit: str = 'bohr',
    **kwargs,
):
    return Systems.from_pyscf(
        pyscf.gto.M(
            atom=[(charge, pos) for charge, pos in zip(charges, positions, strict=True)],
            unit=unit,
            spin=spin,
            **kwargs,
        ),
    )


def rotate_subsystem(
    stationary_charges: Sequence[int],
    stationary_positions: Sequence[tuple[float, float, float]],
    rotating_charges: Sequence[int],
    rotating_positions: Sequence[tuple[float, float, float]],
    angle_degrees: float,
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    unit: str = 'angstrom',
    spin: int | None = None,
    **kwargs,
):
    """
    Build a molecule from two sets of atoms:
      - stationary: remain fixed,
      - rotating:   rotated by `angle_degrees` around `axis`,
    about a specified `pivot` point.

    Parameters
    ----------
    stationary_charges  : charges (atomic numbers) for the stationary atoms
    stationary_positions: positions (x,y,z) for the stationary atoms
    rotating_charges    : charges (atomic numbers) for the rotating atoms
    rotating_positions  : positions (x,y,z) for the rotating atoms
    angle_degrees       : rotation angle in degrees
    axis                : (ax, ay, az) - rotation axis (need not be unit length)
    pivot               : (px, py, pz) - pivot point for the rotation
    unit                : 'angstrom' or 'bohr' for PySCF
    spin                : total spin of the system (e.g. 0 for singlet)
    kwargs              : additional keyword arguments passed to pyscf.gto.M

    Returns
    -------
    A Systems.from_pyscf(...) object (assuming you have that function).
    """
    # Convert to NumPy arrays for easier manipulation
    axis_np = np.array(axis, dtype=float)
    pivot_np = np.array(pivot, dtype=float)
    angle_radians = math.radians(angle_degrees)

    # Build the stationary atom list: (charge, (x, y, z))
    stationary_atoms = [
        (ch, pos)
        for ch, pos in zip(stationary_charges, stationary_positions, strict=True)
    ]

    # Build the rotating atom list similarly
    rotating_atoms = [
        (ch, pos) for ch, pos in zip(rotating_charges, rotating_positions, strict=True)
    ]

    # Separate out the rotating coords so we can rotate them
    rotating_coords = np.array([pos for _, pos in rotating_atoms], dtype=float)

    # Perform the rotation
    for i in range(len(rotating_coords)):
        # Translate so pivot is at origin
        vec = rotating_coords[i] - pivot_np
        # Rotate
        vec_rot = rotate_vector(vec, axis_np, angle_radians)
        # Translate back
        rotating_coords[i] = pivot_np + vec_rot

    # Reassemble the final list of (charge, (x,y,z)) for PySCF
    final_atom_list = []
    final_atom_list.extend(stationary_atoms)  # unchanged
    for i, (ch, _) in enumerate(rotating_atoms):
        x, y, z = rotating_coords[i]
        final_atom_list.append((ch, (x, y, z)))

    # Build the PySCF Mole object
    mol = pyscf.gto.M(
        atom=final_atom_list,
        unit=unit,
        spin=spin,
        **kwargs,
    )

    # Wrap it in your Systems.from_pyscf(...) or however you do it
    return Systems.from_pyscf(mol)


@functools.cache
def _deeperwin_datasets():
    with open(_src_dir / 'data/deeperwin/datasets.json') as inp:
        return json.load(inp)


@functools.cache
def _deeperwin_geometries():
    with open(_src_dir / 'data/deeperwin/geometries.json') as inp:
        return json.load(inp)


def deeperwin_molecule(mol_hash: str | None = None, name: str | None = None) -> Systems:
    assert mol_hash is not None or name is not None
    if mol_hash is not None:
        geometry = _deeperwin_geometries()[mol_hash]
    else:
        matches = [
            g for g in _deeperwin_geometries().values() if g.get('name', '') == name
        ]
        assert len(matches) == 1, f'No unique match for {name}'
        geometry = matches[0]
    charges = tuple(geometry['Z'])
    n_elec = sum(charges) - geometry.get('charge', 0)
    spin = geometry.get('spin', 0)
    n_up, n_down = (n_elec + spin) // 2, (n_elec - spin) // 2
    nuclei = jnp.asarray(geometry['R'], dtype=jnp.float32)
    return Systems.create((n_up, n_down), charges, nuclei)


def deeperwin_dataset(name: str) -> Systems:
    dataset = _deeperwin_datasets()[name]
    result = []
    for geometry in dataset['geometries']:
        mol_hash, mol_name = geometry.split('__')
        result.append(deeperwin_molecule(mol_hash, mol_name))
    for subset in dataset['datasets']:
        result += deeperwin_dataset(subset)
    return Systems.merge(result)


def excited(
    total_spins: int | Sequence[int],
    n_states: int,
    molecules: Sequence[tuple[str, dict[str, Any]]],
) -> Systems:
    """System factory for creating systems with excited states.

    Sample configuration:
    ```yaml
        systems:
        molecules:
            -   - "excited"
                - total_spins: 0 # if int, all states will have the same total spin
                  n_states: 2
                  molecules:
                        - ["diatomic", { charge1: 7, charge2: 7, distance: 2.02858 }]
        # or equivalently
            - ["excited",
                {
                    total_spins: 0,
                    n_states: 2,
                    molecules: [
                        ["diatomic", { charge1: 7, charge2: 7, distance: 2.02858 }]
                    ]
                }
            ]
    ```

    Args:
        total_spins: Total spin for each molecule, i.e. `2S` or `no. of up electrons` - `no. of down electrons`.
            When an integer is provided, all states will have the same total spin. Otherwise, the length of the
            sequence must be `n_states`.
        n_states: Number of excited states to create for each molecule.
        molecules: List of molecule configurations as would be passed to non-excited system factories.
    """

    if isinstance(total_spins, int):
        total_spins = [total_spins] * n_states
    else:
        assert len(total_spins) == n_states, 'total_spins must have length `n_states`'

    total_spins = sorted(total_spins)

    assert n_states > 0, 'n_states must be greater than 0'
    systems = []
    molecule_idx = 0
    for spin, count in Counter(total_spins).items():
        for name, config in molecules:
            system_factory = globals()[name]
            assert inspect.signature(system_factory).parameters.get('spin') is not None, (
                f'system factory {name} must have a `spin` parameter to be used with `excited`'
            )

            system = system_factory(spin=spin, **config).replace(
                mol_ids=(molecule_idx,),
            )
            system = Systems.merge(
                [system.replace(excitations=(i,)) for i in range(count)],
            )
            systems.append(system)
            molecule_idx += 1
    return Systems.merge(systems)


def create_systems(
    key: jax.Array,
    molecules: Sequence[tuple[str, dict[str, Any]]],
    num_walker_per_mol: int,
    init_electron_width: float = 1.0,
    pseudopotential: dict[str, Any] | None = None,
) -> Systems:
    _systems: list[Systems] = []
    molecule_idx = 0
    for name, config in molecules:
        system = globals()[name](**config)

        # Offset molecule indices
        system: Systems = system.replace(
            mol_ids=tuple(
                sum(offset_id)
                for offset_id in zip(
                    system.mol_ids,
                    (molecule_idx,) * len(system),
                    strict=True,
                )
            ),
        )
        molecule_idx = max(system.mol_ids) + 1
        _systems.append(system)

    systems = Systems.merge(_systems)
    if pseudopotential is not None and pseudopotential.get('enable', False):
        from neural_pfaffian.pseudopotential import attach_pseudopotentials

        systems = attach_pseudopotentials(systems, **pseudopotential)

    systems = systems.sorted()

    return systems.init_electrons(
        key,
        num_walker_per_mol,
        init_electron_width,
    )


def rotate_vector(
    vector: np.ndarray,
    axis: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    """
    Rotate `vector` around the specified `axis` (both 3D) by `angle_radians`.
    Axis should be a 3D unit vector. Uses Rodrigues' rotation formula.
    """
    axis = axis / np.linalg.norm(axis)
    v = vector
    k = axis
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    return (v * c) + (np.cross(k, v) * s) + (k * np.dot(k, v) * (1.0 - c))


def load_systems_from_config_and_msgpack(
    config: dict[str, Any] | Path,
    msgpack_path: str | Path,
):
    """Load Systems from a msgpack file given a configuration dictionary."""
    if isinstance(config, Path | str):
        _config: dict[str, Any] = yaml.safe_load(Path(config).read_text())
    elif isinstance(config, dict):
        _config = config
    else:
        raise ValueError('config must be a dict or a Path/str to a JSON file.')

    # If we get a top-level config
    if 'systems' in _config:
        _config = _config['systems']

    if not ('molecules' in _config and 'num_walker_per_mol' in _config):
        raise ValueError("Configuration must contain 'systems' or 'molecules' key.")

    systems = create_systems(jax.random.PRNGKey(0), **_config)
    systems = systems.from_file(msgpack_path)

    # Treat mol_data separately as otherwise it would require setting up
    # all other components
    raw_bytes = Path(msgpack_path).read_bytes()
    data = flax.serialization.msgpack_restore(raw_bytes)
    systems = systems.replace(
        mol_data=data['mol_data'],
    )
    return systems
