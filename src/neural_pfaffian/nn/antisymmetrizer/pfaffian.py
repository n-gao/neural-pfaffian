import functools
from collections.abc import Sequence
from typing import Literal, cast

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode
from jaxtyping import Array, Float

from neural_pfaffian.hf import MolOrbitals, MolOrbSelector
from neural_pfaffian.linalg import (
    antisymmetric_block_diagonal,
    skewsymmetric_quadratic,
    slog_pfaffian,
    slog_pfaffian_skewsymmetric_quadratic,
)
from neural_pfaffian.nn.envelope import Envelope
from neural_pfaffian.nn.module import ParamMeta, ParamTypes, ReparamModule
from neural_pfaffian.nn.wave_function import AntisymmetrizerP
from neural_pfaffian.systems import Systems, SystemsWithPretrainTarget, chunk_electron
from neural_pfaffian.utils import itemgetter
from neural_pfaffian.utils.jax_utils import pad_along_axis, vmap
from neural_pfaffian.utils.segment_utils import unsegment_axis
from neural_pfaffian.utils.tree_utils import tree_stack


def max_orbitals(orb_per_charge: dict[str, int]):
    return max(orb_per_charge.values())


def orbital_mask(orb_per_charge: dict[str, int], charges: Sequence[int]):
    max_orb = max_orbitals(orb_per_charge)
    return np.concatenate(
        [
            [True] * orb + [False] * (max_orb - orb)
            for orb in map(orb_per_charge.__getitem__, map(str, charges))
        ],
    )


AxisLiteral = Literal['det', 'orb']
_AXIS_TO_INDEX: dict[AxisLiteral, int] = {'orb': 0, 'det': -1}


def _resolve_axis(axis: AxisLiteral | None, label: str) -> int | None:
    if axis is None:
        return None
    try:
        return _AXIS_TO_INDEX[axis]
    except KeyError as err:
        raise ValueError(f'Invalid {label}: {axis}') from err


class PerNucOrbitals(ReparamModule):
    determinants: int
    orb_per_charge: dict[str, int]
    envelope: Envelope
    param_sharing_axis: Literal['det', 'orb'] | None = 'det'

    @nn.compact
    def __call__(self, systems: Systems, elec_embeddings: Float[Array, 'electrons dim']):
        inp_dim = elec_embeddings.shape[-1]
        max_orb = max_orbitals(self.orb_per_charge)
        param_sharing_axis = _resolve_axis(
            self.param_sharing_axis,
            'parameter sharing axis',
        )

        W, W_meta = self.reparam(
            'projection',
            jax.nn.initializers.normal(1 / jnp.sqrt(inp_dim), dtype=jnp.float32),
            (systems.n_nuc, max_orb, inp_dim, self.determinants),
            param_type=ParamTypes.NUCLEI,
            param_sharing_axis=param_sharing_axis,
            keep_distr=False,
        )
        # Set envelopes output correctly
        env = self.envelope.copy(out_dim=self.determinants * max_orb, out_per_nuc=True)(
            systems,
        )

        result: list[Array] = []
        for emb, env, W, (spins, charges) in zip(
            systems.group(elec_embeddings, chunk_electron),
            env,
            systems.group(W, W_meta.param_type.value.chunk_fn),
            systems.unique_spins_and_charges,
            strict=True,
        ):
            n_nuc = len(charges)
            n_orb = max_orb * n_nuc
            norm = (max(spins) / n_orb) ** 0.5
            orb_mask = orbital_mask(self.orb_per_charge, charges)

            @jax.vmap  # vmap over different molecules
            def _orbitals(emb: Array, env: Array, W: Array):
                env = einops.rearrange(env, 'elec (orb det) -> elec orb det', orb=n_orb)
                env = env[:, orb_mask, :]
                W = einops.rearrange(W, 'nuc opa dim det -> (nuc opa) dim det')
                W = W[orb_mask, :, :]
                return norm * einops.einsum(
                    emb,
                    W,
                    env,
                    'elec dim, orb dim det, elec orb det -> elec orb det',
                )

            result.append(_orbitals(emb, env, W))
        return result


class PfaffianOrbitals(PyTreeNode):
    orbitals: Float[Array, 'mols det elec orbitals']
    antisymmetrizer: Float[Array, 'mols det orbitals orbitals']
    orb_A_orb_product: Float[Array, 'mols det elec elec']


CorePfaffianOrbital = tuple[
    list[Array],
    list[Array],
    list[Array],
    list[Array],
    tuple[Array, ParamMeta],
]


class UnexcitedPfaffian(ReparamModule):
    determinants: int
    orb_per_charge: dict[str, int]
    envelope: Envelope

    @nn.compact
    def __call__(
        self,
        systems: Systems,
        elec_embeddings: Float[Array, 'electrons dim'],
    ) -> CorePfaffianOrbital:
        n_det = self.determinants
        max_orb = max_orbitals(self.orb_per_charge)

        up_sys, down_sys, up_idx, down_idx = systems.split_by_spin
        up_emb = elec_embeddings[up_idx]
        down_emb = elec_embeddings[down_idx]
        up_orbs = PerNucOrbitals(
            n_det,
            self.orb_per_charge,
            self.envelope.copy(pi_init=1.0, keep_distr=False),
        )(up_sys, up_emb)
        up_down_orbs = PerNucOrbitals(
            n_det,
            self.orb_per_charge,
            self.envelope.copy(pi_init=1.0, keep_distr=False),
        )(up_sys, up_emb)
        down_orbs = PerNucOrbitals(
            n_det,
            self.orb_per_charge,
            self.envelope.copy(pi_init=1.0, keep_distr=False),
        )(down_sys, down_emb)
        down_up_orbs = PerNucOrbitals(
            n_det,
            self.orb_per_charge,
            self.envelope.copy(pi_init=1.0, keep_distr=False),
        )(down_sys, down_emb)

        # If n_elec is odd, we need an extra orbital
        fill_vec, fill_vec_meta = self.reparam(
            'fill_coeffs',
            jax.nn.initializers.normal(1, dtype=jnp.float32),
            (systems.n_nuc, 2 * max_orb, n_det),
            param_type=ParamTypes.NUCLEI,
        )

        return (
            up_orbs,
            up_down_orbs,
            down_orbs,
            down_up_orbs,
            (fill_vec, fill_vec_meta),
        )


class ExcitedPfaffian(ReparamModule):
    determinants: int
    orb_per_charge: dict[str, int]

    max_num_states: int | None = None

    @nn.compact
    def __call__(self, systems: Systems, core_orbitals: CorePfaffianOrbital):
        assert self.max_num_states is not None and self.max_num_states > 0, (
            'max_num_states must be set'
        )
        max_orb = max_orbitals(self.orb_per_charge)
        As = [
            self.reparam(
                f'antisymmetrizer_state_{i}',
                jax.nn.initializers.normal(1, dtype=jnp.float32),
                (
                    systems.n_nn,
                    2,
                    max_orb,
                    max_orb,
                    self.determinants,
                ),
                param_type=ParamTypes.NUCLEI_NUCLEI,
                bias=True,
                param_sharing_axis=-1,
            )
            for i in range(self.max_num_states)
        ]

        result: list[PfaffianOrbitals] = []
        ups, up_downs, downs, down_ups, (fill_vec, fill_meta) = core_orbitals
        for up, up_down, down, down_up, fill, (spins, charges), excitation, *A in zip(
            ups,
            up_downs,
            downs,
            down_ups,
            systems.group(fill_vec, fill_meta.param_type.value.chunk_fn),
            systems.unique_spins_and_charges,
            systems.grouped_excitations,
            *[systems.group(A, A_m.param_type.value.chunk_fn) for A, A_m in As],
            strict=True,
        ):
            n_elec, n_nuc = sum(spins), len(charges)
            orb_mask = orbital_mask(self.orb_per_charge, charges)

            @vmap  # vmap over different molecules
            # vmap over different determinants
            @vmap(in_axes=(-1,) * 6 + (None,), out_axes=0)
            def _orbitals(
                up: Array,
                up_down: Array,
                down: Array,
                down_up: Array,
                fill: Array,
                As: tuple[Array, ...],
                excitation: Array,
            ):
                orbitals = jnp.concatenate(
                    [
                        jnp.concatenate([up, up_down], axis=1),
                        jnp.concatenate([down_up, down], axis=1),
                    ],
                    axis=0,
                )  # (n_elec, 2*n_orbs)

                # Pad additional orbital if n_elec is odd
                full_mask = np.concatenate([orb_mask, orb_mask])
                if n_elec % 2 == 1:
                    fill = fill.reshape(1, -1)[:, full_mask]
                    orbitals = jnp.concatenate([orbitals, fill], axis=0)

                # select antisymmetrizer for excitation
                A = jax.lax.select_n(excitation, *As)

                # A: (2, 2*n_orbs, 2*n_orbs)
                A = einops.rearrange(
                    A,
                    '(n1 n2) two o1 o2 -> two (n1 o1) (n2 o2)',
                    n1=n_nuc,
                )
                A = A[:, orb_mask, :][:, :, orb_mask]  # remove unused orbitals
                A_offdiag, A_diag = A[0], A[1]
                A_uu, A_dd = (A_diag - A_diag.mT) / 2, (A_diag + A_diag.mT) / 2
                A_dd = jnp.triu(A_dd) - jnp.triu(A_dd).mT
                A = jnp.block([[A_uu, A_offdiag], [-A_offdiag.mT, A_dd]])
                A /= A.shape[-1]

                # Product
                orb_A_orb_product = skewsymmetric_quadratic(
                    orbitals.astype(jnp.float64),
                    A.astype(jnp.float64),
                )
                return PfaffianOrbitals(orbitals, A, orb_A_orb_product)

            result.append(
                _orbitals(up, up_down, down, down_up, fill, tuple(A), excitation),
            )

        return result


class Pfaffian(
    ReparamModule,
    AntisymmetrizerP[
        list[PfaffianOrbitals],
        CorePfaffianOrbital,
        None,
    ],
):
    determinants: int
    orb_per_charge: dict[str, int]
    envelope: Envelope

    hf_match_orbitals: float
    hf_match_antisymmetrizer: float
    hf_match_noise_std: float

    max_num_states: int | None = None

    def setup(self):
        self.unexcited_pfaffians = UnexcitedPfaffian(
            self.determinants,
            self.orb_per_charge,
            self.envelope,
        )
        self.excited_pfaffian = ExcitedPfaffian(
            self.determinants,
            self.orb_per_charge,
            self.max_num_states,
        )

    def __call__(
        self,
        systems: Systems,
        elec_embeddings: Float[Array, 'electrons dim'],
    ):
        assert self.max_num_states is not None and self.max_num_states > 0, (
            'max_num_states must be set'
        )

        core_orbitals = self.unexcited_pfaffians(systems, elec_embeddings)
        return self.excited_pfaffian(systems, core_orbitals)

    def core_orbitals(
        self,
        systems: Systems,
        elec_embeddings: Float[Array, 'electrons dim'],
    ):
        return self.unexcited_pfaffians(systems, elec_embeddings)

    def apply_excitation(self, systems: Systems, core_orbitals):
        return self.excited_pfaffian(systems, core_orbitals)

    def to_slog_psi(self, systems: Systems, orbitals: list[PfaffianOrbitals]):
        signs, logpsis = [], []
        for orb in orbitals:
            if orb.orbitals.shape[-2] == orb.orb_A_orb_product.shape[-2]:
                # We can use the fused version
                sign, logpsi = slog_pfaffian_skewsymmetric_quadratic(
                    orb.orbitals.astype(jnp.float64),
                    orb.antisymmetrizer.astype(jnp.float64),
                )
            else:
                # We need to act on the padded version
                sign, logpsi = slog_pfaffian(orb.orb_A_orb_product.astype(jnp.float64))
            logpsi, sign = jax.nn.logsumexp(logpsi, axis=1, b=sign, return_sign=True)
            signs.append(sign)
            logpsis.append(logpsi)
        order = systems.inverse_unique_indices
        sign = jnp.concatenate(signs)[order]
        log_psi = jnp.concatenate(logpsis)[order]
        return sign.astype(jnp.int32), log_psi.astype(systems.electrons.dtype)

    def match_hf_orbitals(
        self,
        systems: SystemsWithPretrainTarget,
        orbitals: list[PfaffianOrbitals],  # grouped by molecules
    ):
        targets = systems.targets

        loss_fn = functools.partial(
            pfaffian_pretraining_loss,
            orb_weight=self.hf_match_orbitals,
            antisym_weight=self.hf_match_antisymmetrizer,
            noise_std=self.hf_match_noise_std,
            skip_subproblem=max(systems.n_nuc_by_mol) == 1,
        )

        loss = jnp.zeros((), dtype=jnp.float32)
        n_diff_mols = jnp.zeros((), dtype=jnp.int32)

        for idx, pfaff_orbs in zip(
            systems.unique_indices,
            orbitals,
            strict=True,
        ):
            # for-loop separates different molecule types
            getter = itemgetter(*idx)
            target_i, mol_ids_i = getter(targets), getter(systems.mol_ids)

            # Data is split by molecules, which is a segmented axis
            # where the first n_states elements correspond to first geometry, etc.
            # We split this axis to VMAP over different geometries while keeping
            # each geometry's excitations grouped along the segment axis.
            n_segments = len(set(mol_ids_i))
            n_diff_mols += n_segments
            unsegment = functools.partial(
                unsegment_axis,
                segment_ids=np.array(mol_ids_i),
                indices_are_grouped=False,
                num_segments=n_segments,
            )

            # Stack the molecules in the first dimension
            target_i = tree_stack(*target_i)
            target_i = jax.tree.map(unsegment, target_i)

            # for pfaff_orbs, we expect to see the molecules in the -4 dim. Thus, we should move it to the front
            pfaff_orbs = jax.tree.map(lambda x: jnp.moveaxis(x, -4, 0), pfaff_orbs)
            pfaff_orbs = jax.tree.map(unsegment, pfaff_orbs)

            # Matching
            loss_i = jax.vmap(loss_fn)(
                target_i.molecular_orbital_permutation,
                target_i.molecular_orbitals,
                pfaff_orbs,
            )
            loss += loss_i.sum()

        return (
            loss / n_diff_mols,
            list(systems.cache),
            {},
        )

    def init_systems(self, key: Array, systems: SystemsWithPretrainTarget):
        for sub_sys in systems.sub_configs:
            n_el = sub_sys.n_elec
            n_orbs = (
                orbital_mask(self.orb_per_charge, tuple(sub_sys.flat_charges)).sum() * 2
            )
            target = sub_sys.targets[0]
            n_mo_orbs = target.molecular_orbitals.shape[-1]

            # Check for truncation of active orbitals
            # We don't care about orbital truncation, if we're not pretraining A
            if n_orbs < n_mo_orbs and self.hf_match_antisymmetrizer > 0:
                # only works unjitted
                spin_pf_orbs = n_orbs // 2
                spin_mo_orbs = n_mo_orbs // 2
                truncation = list(range(spin_pf_orbs, spin_mo_orbs)) + list(
                    range(spin_mo_orbs + spin_pf_orbs, 2 * spin_mo_orbs),
                )
                assert jnp.all(
                    target.molecular_orbital_permutation[..., truncation, :] == 0,
                ), 'Active orbitals must not be truncated'
            if n_el % 2 == 1 and self.hf_match_antisymmetrizer > 0:
                assert jnp.all(target.molecular_orbital_permutation[..., -1, :] == 0), (
                    'Active orbital would be replaced by dummy orbital'
                )
        return systems.replace(cache=tuple([None] * systems.n_mols))


def pfaffian_pretraining_loss(
    hf_orb_selector: MolOrbSelector,
    hf_orbs: MolOrbitals,
    pf_orbs: PfaffianOrbitals,
    orb_weight: float,
    antisym_weight: float,
    *,
    noise_std: float = 0.0,
    skip_subproblem: bool = False,
):
    hf_orb_selector = cast('Array', hf_orb_selector)
    hf_orbs = cast('Array', hf_orbs)

    n_pf_orbs = pf_orbs.orbitals.shape[-1]
    n_hf_orbs = hf_orbs.shape[-1]
    n_el = hf_orbs.shape[-2]

    if n_pf_orbs < n_hf_orbs:
        # If the number of pfaffian orbitals is smaller than the number of
        # molecular orbitals, we need to truncate the molecular orbitals
        spin_pf_orbs = n_pf_orbs // 2
        spin_mo_orbs = n_hf_orbs // 2
        truncation = list(range(0, spin_pf_orbs)) + list(
            range(spin_mo_orbs, spin_mo_orbs + spin_pf_orbs),
        )
        hf_orb_selector = hf_orb_selector[..., truncation, :]
        hf_orbs = hf_orbs[..., truncation]
    elif n_pf_orbs > n_hf_orbs:
        # Else we need to add dummy orbitals
        pad_n = n_pf_orbs - n_hf_orbs
        hf_orb_selector = jnp.pad(
            hf_orb_selector,
            ((0, 0),) * (hf_orb_selector.ndim - 2) + ((0, pad_n), (0, 0)),
        )
        hf_orbs = jnp.pad(
            hf_orbs,
            ((0, 0),) * (hf_orbs.ndim - 1) + ((0, pad_n),),
        )

    if n_el % 2 == 1:
        # If n_elec is odd, we need the last orbital to be a dummy orbital
        hf_orbs = pad_along_axis(hf_orbs, axis=-2, pad_width=(0, 1))
        hf_orbs = hf_orbs.at[..., -1].set(0)
        hf_orbs = hf_orbs.at[..., -1, -1].set(1)
        # And a dummy electron that occupies the dummy orbital
        hf_orb_selector = pad_along_axis(hf_orb_selector, axis=-1, pad_width=(0, 1))
        hf_orb_selector = hf_orb_selector.at[..., -1, -1].set(1)

    # Insert broadcast dimension over determinants
    hf_orbs = hf_orbs[..., None, :, :]
    # orb_selector needs a broadcast dim for det and walker
    hf_orb_selector = hf_orb_selector[..., None, None, :, :]

    nn_antisym = pf_orbs.antisymmetrizer
    nn_orb = pf_orbs.orbitals

    # Procrutes matching for orbitals
    if not skip_subproblem:
        U, _, V = jnp.linalg.svd(
            jnp.einsum('...i,...j->ij', hf_orbs, nn_orb),
            full_matrices=False,
        )
        # orb_matching is usually well-defined, gradients can flow here.
        orb_matching = U @ V
    else:
        orb_matching = jnp.eye(hf_orbs.shape[-1], dtype=hf_orbs.dtype)

    orb_loss = ((hf_orbs @ orb_matching - nn_orb) ** 2).mean()
    # The antisymmetrizer is copied for all electron samples
    nn_antisym = nn_antisym[:, :1]
    p = hf_orb_selector.mT @ orb_matching
    if not skip_subproblem:
        U, _, V = jnp.linalg.svd(
            skewsymmetric_quadratic(p, nn_antisym),
            full_matrices=False,
        )
        orb_rotation = U @ V
    else:
        orb_rotation = antisymmetric_block_diagonal(p.shape[-2] // 2, dtype=p.dtype)
    # For the antisymmetrizer, we do not have full rank and thus the gradient is not well defined. Also eigenvalues are repeated leading to non-well defined gradients.
    orb_rotation = jax.lax.stop_gradient(orb_rotation)
    hf_antisym = skewsymmetric_quadratic(p.mT, orb_rotation)
    if noise_std > 0.0:
        key = jax.random.PRNGKey(0)
        noise = (
            jax.random.normal(key, hf_antisym.shape, dtype=hf_antisym.dtype) * noise_std
        )
        noise /= jnp.sqrt(hf_antisym.shape[-1])
        noise -= noise.mT
        hf_antisym += noise
    asym_loss = ((hf_antisym - nn_antisym) ** 2).sum(axis=(-2, -1)).mean() / n_el**2
    total_loss = asym_loss * antisym_weight + orb_loss * orb_weight
    return total_loss
