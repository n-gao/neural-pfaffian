import jax.numpy as jnp
import numpy as np
from chex import assert_trees_all_equal_shapes_and_dtypes
from fixtures import *  # noqa: F403
from utils import assert_finite

from neural_pfaffian.clipping import NoneClipping, NoneMasking
from neural_pfaffian.spin_operator import _SPIN_EMA, SpinPenalty


def _spin_setup(wave_function, params, systems, *, penalty_type='minimize'):
    penalty = SpinPenalty(
        wave_function=wave_function,
        sample_masking=NoneMasking(),
        ratio_clipping=NoneClipping(),
        penalty_scale=lambda step: jnp.array(1.0, dtype=jnp.float32),
        max_grad_norm=10.0,
        penalty_type=penalty_type,
        spin_ema_decay=lambda step: jnp.array(0.0, dtype=jnp.float32),
    )
    systems = penalty.init_systems(systems)
    return penalty, params, systems


def _preset_spin_ema(systems, value):
    ema = systems.get_mol_data(_SPIN_EMA)
    ema = ema.update(jnp.full((systems.n_mols,), value, dtype=jnp.float32), 0.0)
    return systems.set_mol_data(_SPIN_EMA, ema)


def test_spin_penalty_returns_finite_gradient(
    generalized_wf,
    generalized_wf_params,
    batched_excited_systems,
):
    penalty, params, systems = _spin_setup(
        generalized_wf,
        generalized_wf_params,
        batched_excited_systems,
    )

    (gradient, aux_data), systems = penalty(
        params,
        systems,
        jnp.array(0, dtype=jnp.int32),
    )

    assert_finite(gradient)
    assert_finite(aux_data)
    assert_finite(systems)
    assert_trees_all_equal_shapes_and_dtypes(gradient, params)


def test_spin_penalty_minimize_and_snap_targets_differ(
    generalized_wf,
    generalized_wf_params,
    batched_excited_systems,
):
    minimize, params, systems = _spin_setup(
        generalized_wf,
        generalized_wf_params,
        batched_excited_systems,
        penalty_type='minimize',
    )
    systems = _preset_spin_ema(systems, 2.0)
    snap = minimize.replace(penalty_type='snap')

    (_, minimize_aux), _ = minimize(params, systems, jnp.array(0, dtype=jnp.int32))
    (_, snap_aux), _ = snap(params, systems, jnp.array(0, dtype=jnp.int32))

    assert np.allclose(np.asarray(minimize_aux['spin/P_plus_shift']), 0.0)
    assert not np.allclose(
        np.asarray(minimize_aux['spin/P_plus_shift']),
        np.asarray(snap_aux['spin/P_plus_shift']),
    )


def test_spin_ema_updates_each_call(
    generalized_wf,
    generalized_wf_params,
    batched_excited_systems,
):
    penalty, params, systems = _spin_setup(
        generalized_wf,
        generalized_wf_params,
        batched_excited_systems,
    )
    penalty = penalty.replace(
        spin_ema_decay=lambda step: jnp.array(0.5, dtype=jnp.float32),
    )

    ema0 = systems.get_mol_data(_SPIN_EMA)
    (_, _), systems = penalty(params, systems, jnp.array(0, dtype=jnp.int32))
    ema1 = systems.get_mol_data(_SPIN_EMA)
    (_, _), systems = penalty(params, systems, jnp.array(1, dtype=jnp.int32))
    ema2 = systems.get_mol_data(_SPIN_EMA)

    assert not np.allclose(np.asarray(ema0.weight), np.asarray(ema1.weight))
    assert not np.allclose(np.asarray(ema1.weight), np.asarray(ema2.weight))
