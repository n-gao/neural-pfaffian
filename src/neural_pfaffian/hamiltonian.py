import functools
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
from folx import (
    ForwardLaplacianOperator,
    LoopLaplacianOperator,
    batched_vmap,
)
from jaxtyping import Array, Float, Key, PyTree

from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    WaveFunctionParameters,
)
from neural_pfaffian.pseudopotential import (
    UNIT_ICOSAHEDRON_POINTS,
    icosahedron_quadrature_configs,
    precompute_legendre_on_dirs,
)
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils.jax_utils import jit, vmap
from neural_pfaffian.utils.segment_utils import segment_sum


class KineticEnergyOp(Enum):
    LOOP = 'loop'
    FORWARD = 'forward'


def make_kinetic_energy(
    wf: GeneralizedWaveFunction,
    operator: KineticEnergyOp = KineticEnergyOp.FORWARD,
):
    match operator:
        case KineticEnergyOp.LOOP:
            op = LoopLaplacianOperator()
            vmap_fn = functools.partial(batched_vmap, max_batch_size=1)
        case KineticEnergyOp.FORWARD:
            op = ForwardLaplacianOperator(0.25)
            vmap_fn = functools.partial(batched_vmap, max_batch_size=1)

    @jit
    def laplacian(
        params: WaveFunctionParameters,
        systems: Systems,
        reparams: PyTree[Array] | None = None,
    ) -> Float[Array, ' n_mols']:
        if reparams is None:
            reparams = wf.reparams(params, systems)

        result = []
        for sub_system, mol_params in zip(
            systems.iter_stacked_sub_systems(),
            wf.group_reparams(systems, reparams),
            strict=False,
        ):

            @vmap_fn  # vmap over n_mols
            def _laplacian(systems: Systems, reparams: PyTree[Array], excitation: Array):
                def f_closure(elec):
                    sys = systems.replace(
                        electrons=elec,
                        excitations=excitation[..., None],
                    )
                    return wf.apply(params, sys, reparams).squeeze()

                laplacian, quantum_force = op(f_closure)(systems.electrons)
                return -0.5 * (jnp.sum(laplacian) + jnp.sum(quantum_force**2))

            result.append(
                _laplacian(
                    sub_system,
                    mol_params,
                    jnp.asarray(sub_system.excitations),
                ),
            )

        return jnp.concatenate(result)[systems.inverse_unique_indices]

    return laplacian


def _potential_local_ecp(systems: Systems) -> Float[Array, ' n_mols']:
    """Local ECP contribution adapted from Ferminet's pseudopotential tables."""
    if not systems.has_pseudopotentials:
        return jnp.zeros((systems.n_mols,), dtype=systems.electrons.dtype)

    results = []
    for sub_system in systems.iter_stacked_sub_systems():
        pp = sub_system.pp_data[0]
        r_grid = jnp.asarray(pp.r_grid, dtype=sub_system.electrons.dtype)
        v_loc = jnp.asarray(pp.v_loc, dtype=sub_system.electrons.dtype)
        ecp_mask = jnp.asarray(sub_system.ecp_mask, dtype=sub_system.electrons.dtype)

        def _single_mol(nuclei, electrons):
            r_en = jnp.linalg.norm(
                electrons[:, None, :] - nuclei[None, :, :],
                axis=-1,
            )  # (n_elec, n_nuc)
            v_loc_pairs = jax.vmap(
                lambda dist_row, v_grid: jnp.interp(dist_row, r_grid, v_grid),
                in_axes=(1, 0),
                out_axes=1,
            )(r_en, v_loc)
            return jnp.sum(v_loc_pairs * ecp_mask)

        results.append(jax.vmap(_single_mol)(sub_system.nuclei, sub_system.electrons))

    return jnp.concatenate(results)[systems.inverse_unique_indices]


def _potential_nonlocal_ecp(
    wf: GeneralizedWaveFunction,
    params: WaveFunctionParameters,
    systems: Systems,
    key: Key,
    reparams: PyTree[Array] | None = None,
) -> jax.Array:
    if reparams is None:
        reparams = wf.reparams(params, systems)

    results: list[Float[Array, ' n_mols_in_subsystem']] = []

    for subsystem, mol_params in zip(
        systems.iter_stacked_sub_systems(),
        wf.group_reparams(systems, reparams),
        strict=True,
    ):
        dtype = subsystem.electrons.dtype
        zeros = jnp.zeros((subsystem.electrons.shape[0],), dtype=dtype)  # (n_mols, )

        # Retrieve pseudopotential data for all molecules in this subsystem (identical across mols)
        pp = subsystem.pp_data[0]

        # Precomputed radial channel potential for interpolation
        v_nonloc = pp.v_nonloc  # (n_nuc, n_l, n_grid)
        r_grid = pp.r_grid  # (n_grid,)

        # Number of (non local) angular momentum channels
        n_l = v_nonloc.shape[1]

        if not np.any(subsystem.ecp_mask) or n_l == 0:
            results.append(zeros)
            continue

        ecp_mask = subsystem.ecp_mask  # (n_nuc,)

        # Setting up quadrature for spherical integration
        probe_directions = jnp.asarray(UNIT_ICOSAHEDRON_POINTS, dtype=dtype)  # (12, 3)
        n_quad = probe_directions.shape[0]  # 12
        legendre_values = precompute_legendre_on_dirs(probe_directions, n_l)
        # (2 l + 1) * 1/12
        quad_weights = (2 * jnp.arange(n_l, dtype=dtype) + 1) / n_quad  # (n_l,)

        excitations = jnp.asarray(subsystem.excitations)  # (n_mols,)
        key, subkey = jax.random.split(key)
        mol_keys = jax.random.split(subkey, subsystem.electrons.shape[0])

        @vmap  # over molecules in subsystem
        def _molecule_contrib(
            systems: Systems,
            excitation: Float[Array, ''],
            reparams: PyTree[Array],
            mol_key: Key,
        ):
            # Reference wave function value at the original configuration for ratios.
            def _excitation_closure(ex: Float[Array, ''], sys: Systems = systems):
                sys = sys.replace(excitations=ex[..., None])
                sign, logpsi = wf.signed(params, sys, reparams=reparams)
                return sign.squeeze(), logpsi.squeeze()

            base_sign, base_log = _excitation_closure(excitation)  # scalar

            nuc_keys = jax.random.split(mol_key, systems.nuclei.shape[0])

            @vmap(out_axes=-1)  # over quadrature points -> (n_quad, n_elec, 3)
            def _electron_batch_closure(electrons: Float[Array, 'n_elec 3']):
                _system = systems.replace(electrons=electrons)
                return _excitation_closure(excitation, _system)  # scalar

            def _nucleus_contrib(carry, idx):
                quad_configs, radius = icosahedron_quadrature_configs(
                    systems.electrons,
                    systems.nuclei[idx],
                    probe_directions,
                    nuc_keys[idx],
                )

                # Interpolate radial potentials at electron-nucleus distances
                radial_vals = jax.vmap(
                    lambda channel: jnp.interp(radius, r_grid, channel),
                )(jnp.array(v_nonloc, dtype=dtype)[idx]).T  # (n_elec, n_l)

                @vmap  # over electrons -> (n_elec, ...)
                def _electron_contrib(
                    r_en: Float[Array, ''],
                    probe_points: Float[Array, 'n_quad n_elec 3'],
                    radial_coef: Float[Array, 'n_l'],
                ):
                    # Compute wave function ratios at all quadrature points.
                    wf_signs, wf_logs = _electron_batch_closure(probe_points)
                    # (12,)
                    wf_ratio = (
                        base_sign[..., None]
                        * wf_signs
                        * jnp.exp(wf_logs - base_log[..., None])
                    )  # (12,)
                    integral = jnp.sum(
                        wf_ratio[..., None] * legendre_values,
                        axis=-2,
                    )  # (n_l,)
                    weighted_radial = radial_coef * quad_weights
                    pot = jnp.sum(
                        weighted_radial * integral,
                        axis=-1,
                    )  # scalar
                    valid = jnp.isfinite(r_en) & (r_en > 0)
                    return jnp.where(valid, pot, jnp.zeros_like(pot))

                elec_pot = _electron_contrib(
                    radius,
                    quad_configs,
                    radial_vals,
                )
                return carry + jnp.sum(elec_pot, axis=0), None

            # Only scan nuclei that actually use an ECP channel.
            pseudized_nuc_idx = np.flatnonzero(np.any(v_nonloc, axis=(-1, -2)) & ecp_mask)
            pot = jax.lax.scan(
                _nucleus_contrib,
                jnp.zeros_like(base_log),
                pseudized_nuc_idx,
                length=pseudized_nuc_idx.shape[0],
            )[0]
            return pot

        results.append(
            _molecule_contrib(
                subsystem,
                excitations,
                mol_params,
                mol_keys,
            ),
        )

    potentials = jnp.concatenate(results)[systems.inverse_unique_indices]
    return potentials


@jit
def potential_energy_components(systems: Systems) -> tuple[jax.Array, ...]:
    """Compute individual Coulomb components.

    Returns electron-electron, electron-nuclear, and nuclear-nuclear terms.
    """
    charges = jnp.asarray(systems.flat_effective_charges, dtype=systems.electrons.dtype)

    # Electron-electron potential
    v_ee = 1 / systems.elec_elec_dists[..., -1]
    v_ee = segment_sum(v_ee, systems.elec_elec_idx[2], systems.n_mols)
    v_ee /= 2  # we counted twice here

    # Electron-nuclear potential
    v_ne = charges[systems.elec_nuc_idx[1]] / systems.elec_nuc_dists[..., -1]
    v_ne = -segment_sum(v_ne, systems.elec_nuc_idx[2], systems.n_mols)

    # Nuclear-nuclear potential
    nn_i, nn_j, nn_mask = systems.nuc_nuc_idx
    dists = systems.nuc_nuc_dists[..., -1]
    self_interaction = dists < 1e-6
    v_nn = charges[nn_i] * charges[nn_j] / dists
    v_nn = jnp.where(self_interaction, 0, v_nn)
    v_nn = segment_sum(v_nn, nn_mask, systems.n_mols)
    v_nn /= 2  # we counted twice here
    return v_ee, v_ne, v_nn


@jit
def potential_energy(systems: Systems):
    v_ee, v_ne, v_nn = potential_energy_components(systems)
    return v_ee + v_ne + v_nn


@jit
def local_pp_energy(systems: Systems):
    """Local pseudopotential term evaluated via radial interpolation."""
    v_loc = (
        _potential_local_ecp(systems)
        if systems.has_pseudopotentials
        else jnp.zeros((systems.n_mols,), dtype=systems.electrons.dtype)
    )
    return v_loc


@jit
def nonlocal_pp_energy(
    wf: GeneralizedWaveFunction,
    params: WaveFunctionParameters,
    systems: Systems,
    key: jax.Array,
    reparams: PyTree[Array] | None = None,
) -> Float[Array, ' n_mols']:
    """Non-local pseudopotential term using icosahedral quadrature."""
    if not systems.has_pseudopotentials:
        zeros = jnp.zeros((systems.n_mols,), dtype=systems.electrons.dtype)
        return zeros
    return _potential_nonlocal_ecp(
        wf,
        params,
        systems,
        key,
        reparams,
    )


def make_local_energy(
    wf: GeneralizedWaveFunction,
    operator: KineticEnergyOp,
):
    kinetic_energy = make_kinetic_energy(
        wf,
        operator,
    )

    @jit
    def local_energy(
        params: WaveFunctionParameters,
        systems: Systems,
        reparams: PyTree[Array],
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        e_kin = kinetic_energy(params, systems, reparams)
        v_ee, v_ne, v_nn = potential_energy_components(systems)
        v_loc = local_pp_energy(systems)
        v_nonloc = nonlocal_pp_energy(
            wf,
            params,
            systems,
            key,
            reparams,
        )
        e_pot_base_local = v_ee + v_ne + v_nn + v_loc
        e_pot_base = e_pot_base_local + v_nonloc
        e_total = e_kin + e_pot_base

        aux_data = {
            'energy/kinetic': e_kin,
            'energy/electron_electron': v_ee,
            'energy/electron_nuclear': v_ne,
            'energy/nuclear_nuclear': v_nn,
            'energy/pseudopotential_local': v_loc,
            'energy/pseudopotential_nonlocal': v_nonloc,
            'energy/potential_total': e_pot_base,
            'energy/total': e_total,
        }

        return e_total, aux_data

    return local_energy
