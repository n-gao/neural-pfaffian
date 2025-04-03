import functools
from enum import Enum

import jax.numpy as jnp
from folx import (
    ForwardLaplacianOperator,
    LoopLaplacianOperator,
    batched_vmap,
)
from jaxtyping import Array, PyTree

from neural_pfaffian.nn.ops import segment_sum
from neural_pfaffian.nn.wave_function import (
    GeneralizedWaveFunction,
    WaveFunctionParameters,
)
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils.jax_utils import jit, vmap


class KineticEnergyOp(Enum):
    LOOP = 'loop'
    FORWARD = 'forward'


def make_kinetic_energy(
    wf: GeneralizedWaveFunction, operator: KineticEnergyOp = KineticEnergyOp.FORWARD
):
    match operator:
        case KineticEnergyOp.LOOP:
            op = LoopLaplacianOperator()
            vmap_fn = vmap
        case KineticEnergyOp.FORWARD:
            op = ForwardLaplacianOperator(0.6)
            vmap_fn = functools.partial(batched_vmap, max_batch_size=1)
        case _:
            raise ValueError(f'Unsupported kinetic energy operator: {operator}.')

    @jit
    def laplacian(
        params: WaveFunctionParameters,
        systems: Systems,
        reparams: PyTree[Array] | None = None,
    ):
        if reparams is None:
            reparams = wf.reparams(params, systems)

        result = []
        for (electrons, nuclei, (spins, charges)), mol_params in zip(
            systems.iter_grouped_molecules(), wf.group_reparams(systems, reparams)
        ):
            sub_system = Systems(
                spins=(spins,),
                charges=(charges,),
                electrons=electrons,
                nuclei=nuclei,
                mol_data={},
            )

            @vmap_fn
            def _laplacian(systems: Systems, reparams: PyTree[Array]):
                def f_closure(elec):
                    return wf.apply(
                        params, systems.replace(electrons=elec), reparams
                    ).squeeze()

                laplacian, quantum_force = op(f_closure)(systems.electrons)
                return -0.5 * (jnp.sum(laplacian) + jnp.sum(quantum_force**2))

            result.append(_laplacian(sub_system, mol_params))
        return jnp.concatenate(result)[systems.inverse_unique_indices]

    return laplacian


@jit
def potential_energy(systems: Systems):
    charges = systems.flat_charges

    v_ee = 1 / systems.elec_elec_dists[..., -1]
    v_ee = segment_sum(v_ee, systems.elec_elec_idx[2], systems.n_mols)
    v_ee /= 2  # we counted twice here

    v_ne = charges[systems.elec_nuc_idx[1]] / systems.elec_nuc_dists[..., -1]
    v_ne = -segment_sum(v_ne, systems.elec_nuc_idx[2], systems.n_mols)

    nn_i, nn_j, nn_mask = systems.nuc_nuc_idx
    dists = systems.nuc_nuc_dists[..., -1]
    self_interaction = dists < 1e-6
    v_nn = charges[nn_i] * charges[nn_j] / dists
    v_nn = jnp.where(self_interaction, 0, v_nn)
    v_nn = segment_sum(v_nn, nn_mask, systems.n_mols)
    v_nn /= 2  # we counted twice here

    return v_ee + v_ne + v_nn


def make_local_energy(wf: GeneralizedWaveFunction, operator: KineticEnergyOp):
    kinetic_energy = make_kinetic_energy(wf, operator)

    @jit
    def local_energy(
        params: WaveFunctionParameters, systems: Systems, reparams: PyTree[Array]
    ):
        return kinetic_energy(params, systems, reparams) + potential_energy(systems)

    return local_energy
