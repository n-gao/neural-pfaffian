import functools
import json
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import pyscf

from neural_pfaffian.systems import Systems

_src_dir = Path(__file__).parent


def atomic(symbol: int, **kwargs):
    return Systems.from_pyscf(
        pyscf.gto.M(atom=[(symbol, (0, 0, 0))], unit='bohr', **kwargs)
    )


def diatomic(charge1: int, charge2: int, distance: float, **kwargs):
    return Systems.from_pyscf(
        pyscf.gto.M(
            atom=[(charge1, (0, 0, 0)), (charge2, (0, 0, distance))],
            unit='bohr',
            **kwargs,
        )
    )


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
    return Systems.create(
        (n_up, n_down), charges, jnp.asarray(geometry['R'], dtype=jnp.float32)
    )


def deeperwin_dataset(name: str) -> Systems:
    dataset = _deeperwin_datasets()[name]
    result = []
    for geometry in dataset['geometries']:
        mol_hash, mol_name = geometry.split('__')
        result.append(deeperwin_molecule(mol_hash, mol_name))
    for subset in dataset['datasets']:
        result += deeperwin_dataset(subset)
    return Systems.merge(result)


def create_systems(
    key: jax.Array,
    molecules: Sequence[tuple[str, dict[str, Any]]],
    num_walker_per_mol: int,
) -> Systems:
    return Systems.merge(
        [globals()[name](**config) for name, config in molecules]
    ).init_electrons(key, num_walker_per_mol)
