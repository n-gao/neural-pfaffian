import numpy as np
import pytest
from pyscf.scf import RHF, UHF, gto

from neural_pfaffian.hf import (
    _get_permutation_from_mf,
    _split_rhf_occ,
)


@pytest.fixture
def sample_scf_mol():
    mol = gto.Mole()
    mol.atom = """
        Li 0.0 0.0 0.0
        H 0.0 0.0 3.0
    """
    mol.basis = 'sto-6g'
    mol.build()
    return mol


@pytest.fixture
def expected_groundstate_permutation(sample_scf_mol):
    nao = sample_scf_mol.nao
    nelec = sample_scf_mol.nelectron
    nup, ndown = sample_scf_mol.nelec

    expected_perm = np.zeros(shape=(2 * nao, nelec))
    expected_perm[:nup, :nup] = np.eye(nup)
    expected_perm[nao : nao + ndown, nup:] = np.eye(ndown)
    return expected_perm


def test_split_rhf_occ():
    uhf_occ = np.array(
        [
            [1, 1, 1, 0],  # alpha
            [1, 0, 0, 0],  # beta
        ],
    )
    rhf_occ = uhf_occ.sum(axis=0)

    test_occ = _split_rhf_occ(rhf_occ)
    assert np.array_equal(test_occ, uhf_occ)


def test_get_permutation_from_hf(sample_scf_mol, expected_groundstate_permutation):
    expected_perm = expected_groundstate_permutation

    # Test with rhf
    rhf = RHF(sample_scf_mol)
    rhf.kernel()
    rhf_perm = _get_permutation_from_mf(rhf)

    assert np.array_equal(rhf_perm, expected_perm)

    # Test with uhf
    uhf = UHF(sample_scf_mol)
    uhf.kernel()
    uhf_perm = _get_permutation_from_mf(uhf)

    assert np.array_equal(uhf_perm, expected_perm)
