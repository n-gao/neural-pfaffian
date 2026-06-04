from collections.abc import Sequence
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from jaxtyping import Array, Float, Key

from neural_pfaffian.systems import Charges, PseudopotentialProperties, Spins, Systems
from neural_pfaffian.utils.constants import ELEMENT_BY_ATOMIC_NUM

# These helpers follow the structure/ideas of Ferminet's pseudopotential module
# (https://github.com/google-deepmind/ferminet/blob/main/ferminet/pseudopotential.py),
# adapted to our data structures.

# Default: pseudize elements above the first row unless symbols is specified explicitly.
DEFAULT_ECP_Z_THRESHOLD = 10

PYSCF_ECP_COEFFS = Sequence[Sequence[tuple[float, float]]]
"""For each r-exponent, a sequence of (exp_coeff, linear_coeff) pairs."""
PYSCF_ECP_FORMAT = tuple[int, Sequence[tuple[int, PYSCF_ECP_COEFFS]]]
"""Tuple of (n_core, ecp_terms), where ecp_terms is a sequence of (l, coeffs) pairs."""


def _gaussian(
    r: jnp.ndarray,
    a: float,
    b: float,
    n: int,
) -> jnp.ndarray:
    """Gaussian r^n e^{-b r^2} scaled by ``a``."""
    return a * r**n * jnp.exp(-b * r**2)


def _eval_ecp_channel(
    r: jnp.ndarray,
    coeffs: PYSCF_ECP_COEFFS,
) -> jnp.ndarray:
    """Evaluate r^2 U_l(r) for a single angular channel on a radial grid."""
    val = jnp.zeros_like(r)
    for r_exponent, terms in enumerate(coeffs):
        for exp_coeff, linear_coeff in terms:
            val = val + _gaussian(r, linear_coeff, exp_coeff, r_exponent)
    return val


def _eval_ecp_on_grid(
    ecp_all: dict[int, PYSCF_ECP_FORMAT],
    r_grid: jnp.ndarray | None = None,
    *,
    log_r0: float = -5.0,
    log_rn: float = 10.0,
    n_grid: int = 10_001,
) -> tuple[dict[int, int], dict[int, jnp.ndarray], jnp.ndarray, int]:
    """Tabulate ECP channels on a logarithmic radial grid.

    Args:
        ecp_all: Mapping from atomic number to PySCF ECP data
            as returned by ``pyscf.gto.basis.load_ecp``.
        r_grid: Optional precomputed radial grid.
        log_r0: log10 of the smallest radius if ``r_grid`` is None.
        log_rn: log10 of the largest radius if ``r_grid`` is None.
        n_grid: Number of radial grid points if ``r_grid`` is None.

    Returns:
        n_cores: Mapping from atomic number to number of core electrons.
        v_grid_dict: Mapping from atomic number to an array of shape
            ``(n_channels, n_grid)`` containing ``U_l(r)`` for each channel.
        r_grid: The radial grid as a 1D JAX array.
        n_channels: Maximum number of angular channels across all species.
    """
    if r_grid is None:
        r_grid = jnp.logspace(log_r0, log_rn, n_grid)
    else:
        n_grid = int(r_grid.shape[0])

    # PySCF ECP format: (n_core, ecp_terms)
    # ecp_terms is a sequence of (l, coeffs) pairs.
    n_channels = max(len(ecp_val[1]) for ecp_val in ecp_all.values())
    n_cores: dict[int, int] = {}
    v_grid_dict: dict[int, jnp.ndarray] = {}

    for z, (n_core, ecp_val) in ecp_all.items():
        n_cores[z] = int(n_core)
        v_grid = jnp.zeros((n_channels, n_grid), dtype=r_grid.dtype)
        # The local channel is stored at l=-1 in PySCF; Here we place it last (at index n_channels-1).
        for l, coeffs in ecp_val:
            channel = _eval_ecp_channel(r_grid, coeffs) / (r_grid**2)
            v_grid = v_grid.at[l].set(channel)
        v_grid_dict[z] = v_grid

    return n_cores, v_grid_dict, r_grid, n_channels


def attach_pseudopotentials(
    systems: Systems,
    *,
    enable: bool = False,
    ecp: str = 'ccecp',
    symbols: Sequence[str | int] | int | None = None,
) -> Systems:
    """Attach pseudopotential data to a `Systems` instance.

    This populates `systems.effective_charges` (Z with core electrons removed)
    and `systems.pp_data` (radial grids plus local/non-local channels).
    Furthermore, the `Systems`' `spins` are adjusted accordingly and the
    electron array is resized to contain only valence electrons.

    **Caution: This function is meant to be called before electron initialization.
    The returned `Systems` contains an electron array of zeros with the correct number
    of valence electrons, but no meaningful positions.**

    Args:
        systems: The systems to augment. Returned instance shares all other fields.
        ecp: PySCF ECP identifier to load (e.g. `'ccecp'`).
        symbols: Optional list of element symbols or atomic numbers to pseudize.
            If an integer is provided, all elements with atomic number above this
            value are pseudized. If `None`, all elements with atomic number
            above `DEFAULT_ECP_Z_THRESHOLD` are pseudized.
    """
    if not enable:
        return systems

    # Table construction mirrors Ferminet's pseudopotential preprocessing:
    # https://github.com/google-deepmind/ferminet/blob/main/ferminet/pseudopotential.py

    # Determine which elements should use an ECP.
    unique_Z = np.unique(systems.flat_charges)

    if symbols is None or isinstance(symbols, int):
        threshold = symbols or DEFAULT_ECP_Z_THRESHOLD
        mask = (unique_Z > threshold) & np.isin(
            unique_Z,
            list(ELEMENT_BY_ATOMIC_NUM.keys()),
        )
        symbol_set = {ELEMENT_BY_ATOMIC_NUM[int(z)].symbol for z in unique_Z[mask]}
    else:
        symbol_list: list[str] = []
        for s in set(symbols):
            if isinstance(s, int):
                element = ELEMENT_BY_ATOMIC_NUM.get(s)
                if element is None:
                    raise ValueError(f'No element with atomic number {s}.')
                s = element.symbol
            symbol_list.append(s)
        symbol_set = set(symbol_list)

    ecp_all: dict[int, PYSCF_ECP_FORMAT] = {}
    for z in unique_Z:
        element = ELEMENT_BY_ATOMIC_NUM.get(z)
        if element is None:
            continue
        if element.symbol not in symbol_set:
            continue
        try:
            ecp_symb = cast(
                'PYSCF_ECP_FORMAT',
                pyscf.gto.basis.load_ecp(ecp, element.symbol),  # (n_core, [(l, coeffs)])
            )
        except KeyError:
            continue
        if not ecp_symb:
            continue
        ecp_all[z] = ecp_symb

    if not ecp_all:
        # No pseudopotentials available for the requested elements.
        return systems

    n_cores, v_grid_dict, r_grid, n_channels = _eval_ecp_on_grid(ecp_all)

    n_grid = int(r_grid.shape[0])
    n_l_nonloc = max(n_channels - 1, 0)

    dtype = systems.electrons.dtype

    effective_charges: list[Charges] = []
    pp_props: list[PseudopotentialProperties] = []

    for charges in systems.charges:
        n_nuc = len(charges)
        eff_mol: list[float] = []
        v_loc_mol = np.zeros((n_nuc, n_grid), dtype=dtype)
        v_nonloc_mol = np.zeros((n_nuc, n_l_nonloc, n_grid), dtype=dtype)

        for local_idx, z in enumerate(charges):
            Z = int(z)
            n_core = n_cores.get(Z, 0)
            eff_val = float(Z - n_core)
            eff_mol.append(eff_val)

            if Z not in v_grid_dict or n_core == 0:
                continue

            v_grid = np.asarray(v_grid_dict[Z], dtype=dtype)  # (n_channels, n_grid)
            if n_l_nonloc > 0:
                v_nonloc_mol[local_idx, :, :] = v_grid[:n_l_nonloc]
            v_loc_mol[local_idx, :] = v_grid[n_l_nonloc]

        effective_charges.append(tuple(int(x) for x in eff_mol))
        pp_props.append(
            PseudopotentialProperties(
                np.asarray(r_grid, dtype=dtype),
                v_loc_mol,
                v_nonloc_mol,
                ecp if symbol_set else '',
            ),
        )

    # Adjust spins
    valence_spins: list[Spins] = []
    for spins, eff_charges in zip(systems.spins, effective_charges, strict=True):
        total_spin = spins[0] - spins[1]
        n_valence_electrons = sum(eff_charges)
        n_up, n_down = (
            (n_valence_electrons + total_spin) // 2,
            (n_valence_electrons - total_spin) // 2,
        )
        valence_spins.append((n_up, n_down))

    return systems.replace(
        effective_charges=tuple(effective_charges),
        pp_data=tuple(pp_props),
        spins=tuple(valence_spins),
        electrons=jnp.zeros((np.array(valence_spins).sum(), 3), dtype=dtype),
    )


# ---- helpers for non-local pseudopotentials ----
def _unit_icosahedron_points() -> np.ndarray:
    """Return vertices of a unit icosahedron used for angular quadrature."""
    theta = np.arctan(2.0)  # polar angle of azimuthal rings
    dirs = []
    dirs.append(np.array([0.0, 0.0, 1.0]))  # north pole
    dirs.append(np.array([0.0, 0.0, -1.0]))  # south pole
    for j in range(5):
        phi = 2 * j * np.pi / 5  # azimuthal ring
        dirs.append(
            np.array(
                [
                    np.cos(phi) * np.sin(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(theta),
                ],
            ),
        )
        dirs.append(
            np.array(
                [
                    np.cos((2 * j - 1) * np.pi / 5) * np.sin(np.pi - theta),
                    np.sin((2 * j - 1) * np.pi / 5) * np.sin(np.pi - theta),
                    np.cos(np.pi - theta),
                ],
            ),
        )
    return np.stack(dirs, axis=0)


UNIT_ICOSAHEDRON_POINTS = _unit_icosahedron_points()
"""Array of shape (12, 3) containing unit icosahedron vertices in Cartesian coordinates."""


def precompute_legendre_on_dirs(dirs: jnp.ndarray, n_l: int) -> jnp.ndarray:
    """Evaluate Legendre polynomials for cos(theta) of given directions.

    Args:
        dirs: Array of shape (n_dirs, 3) with unit directions.
        n_l: Number of angular channels (l) to evaluate.

    Returns:
        Array of shape (n_dirs, n_l) containing P_l(cos theta_dir).
    """
    cos_theta = jnp.clip(dirs[:, 2], -1.0, 1.0)  # (n_dirs,)
    l_vals = jnp.arange(n_l, dtype=jnp.int32)
    return jax.vmap(lambda l: eval_legendre(cos_theta, l))(l_vals).T  # (n_dirs, n_l)


def icosahedron_quadrature_configs(
    electrons: Float[Array, 'walker n_elec 3'],
    nucleus: Float[Array, '3'],
    probe_directions: Float[Array, 'n_quad 3'],
    key: Key,
) -> tuple[Float[Array, 'n_elec n_quad n_elec 3'], Float[Array, ' n_elec']]:
    """Return quadrature electron configs and electron-nucleus norms.

    Shapes:
        electrons: (..., n_elec, 3)
        nucleus: (3,)
        probe_directions: (n_quad, 3)

    Returns:
        quad_configs: (n_elec, n_quad, n_elec, 3) electron configurations where one
            electron at a time is moved to each quadrature point.
        radius: (n_elec,) electron-nucleus distances.
    """
    n_elec = electrons.shape[-2]
    n_quad = probe_directions.shape[0]

    elec_nuc_vector = electrons - nucleus  # (n_elec, 3)
    radius = jnp.linalg.norm(elec_nuc_vector, axis=-1)  # (n_elec,)
    key, subkey = jax.random.split(key)
    rotation_matrices = jax.vmap(align_z_to_vector)(
        jax.random.split(subkey, electrons.shape[0]),
        elec_nuc_vector,
    )  # (n_elec, 3, 3)
    quadrature_directions = jnp.einsum(
        'eij,qj->eqi',
        rotation_matrices,
        probe_directions,
    )  # (n_elec, n_quad, 3)
    quadrature_points = quadrature_directions * radius[:, None, None] + nucleus

    # Construct an array of (n_elec, 3) electron configs for each quadrature point
    # and each electron being moved to the quadrature point, resulting in
    # (moved_electron, quadrature_point, n_elec, 3).
    quad_configs = jnp.broadcast_to(electrons, (n_elec, n_quad, n_elec, 3))
    quad_configs = quad_configs.at[
        jnp.arange(n_elec),
        :,
        jnp.arange(n_elec),
        :,
    ].set(
        quadrature_points,
    )

    return quad_configs, radius


def align_z_to_vector(key: Key, v: Float[Array, '3']) -> Float[Array, '3 3']:
    """Construct rotation matrix aligning z-axis to vector v."""
    v_norm = v / jnp.linalg.norm(v)
    e2 = jnp.cross(v_norm, jax.random.normal(key, shape=(3,)))
    e2 = e2 / jnp.linalg.norm(e2)
    e1 = jnp.cross(e2, v_norm)
    return jnp.stack([e1, e2, v_norm], axis=1)  # columns are basis vectors


def eval_legendre(
    x: Float[Array, '*input_shape'],
    l: int,
) -> Float[Array, '*input_shape']:
    """Evaluate low-order Legendre polynomials for degrees up to 3."""
    return jax.lax.switch(
        l,
        [
            lambda x: jnp.ones_like(x),
            lambda x: x,
            lambda x: 0.5 * (3 * x**2 - 1),
            lambda x: 0.5 * (5 * x**3 - 3 * x),
        ],
        x,
    )
