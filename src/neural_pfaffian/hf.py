from enum import Enum
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from jaxtyping import Array, Float
from neural_pfaffian.utils.gto import Mol

Electrons = Float[Array, '... n_elec 3']
HFOrbitals = tuple[Float[Array, '... n_up n_up'], Float[Array, '... n_down n_down']]  # noqa: F722


class AOImplementation(Enum):
    PYSCF = 'pyscf'
    JAX = 'jax'


class HFOrbitalFn(Protocol):
    def __call__(self, electrons: Electrons) -> HFOrbitals: ...


def make_hf_orbitals(
    mol: pyscf.gto.Mole, implemenation: AOImplementation = AOImplementation.JAX
) -> HFOrbitalFn:
    coeffs = jnp.zeros((mol.nao, mol.nao))
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    coeffs = jnp.asarray(mf.mo_coeff)
    n_up, n_down = mol.nelec
    jax_mol = Mol.from_pyscf_mol(mol)

    def pyscf_atomic_orbitals(electrons: np.ndarray):
        batch_shape = electrons.shape[:-1]
        ao_values = mol.eval_gto('GTOval_sph', electrons.reshape(-1, 3)).astype(
            electrons.dtype
        )
        return ao_values.reshape(*batch_shape, mol.nao)

    def jax_atomic_orbitals(electrons: Electrons):
        return jnp.vectorize(jax_mol.eval_gto, signature='(n,d)->(n,m)')(electrons)

    def hf_orbitals(electrons: Electrons) -> HFOrbitals:
        match implemenation:
            case AOImplementation.PYSCF:
                ao_orbitals = jax.pure_callback(  # type: ignore
                    pyscf_atomic_orbitals,
                    jax.ShapeDtypeStruct(
                        (*electrons.shape[:-1], mol.nao), electrons.dtype
                    ),
                    electrons,
                    vmap_method='expand_dims',
                )
            case AOImplementation.JAX:
                ao_orbitals = jax_atomic_orbitals(electrons)
            case _:
                raise ValueError(f'Unknown implementation: {implemenation}')
        mo_values = jnp.array(ao_orbitals @ coeffs, electrons.dtype)

        up_orbitals = mo_values[..., :n_up, :n_up]
        down_orbitals = mo_values[..., n_up:, :n_down]
        return up_orbitals, down_orbitals

    return hf_orbitals


def make_hf_logpsi(hf_orbitals: HFOrbitalFn):
    def logpsi(electrons: Electrons):
        up_orbitals, dn_orbitals = hf_orbitals(electrons)
        up_logdet = jnp.linalg.slogdet(up_orbitals)[1]
        dn_logdet = jnp.linalg.slogdet(dn_orbitals)[1]
        logpsi = up_logdet + dn_logdet
        return logpsi

    return logpsi
