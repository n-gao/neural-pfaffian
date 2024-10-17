import jax.numpy as jnp
import jax.tree_util as jtu
from fixtures import *  # noqa: F403

from neural_pfaffian.utils.jax_utils import pmap
from test.utils import assert_finite, assert_shape_and_dtype


def test_preconditioner(preconditioner, neural_pfaffian_params, pmapped_systems):
    neural_pfaffian_params = jtu.tree_map(lambda x: x[None], neural_pfaffian_params)
    state = pmap(preconditioner.init)(neural_pfaffian_params)
    apply = pmap(
        preconditioner.apply,
        in_axes=(0, pmapped_systems.electron_vmap, 0, 0),
    )

    grad, new_state, aux_data = apply(
        neural_pfaffian_params,
        pmapped_systems,
        jnp.zeros(
            (*pmapped_systems.electrons.shape[:-2], pmapped_systems.n_mols),
            dtype=pmapped_systems.electrons.dtype,
        ),
        state,
    )
    assert_shape_and_dtype(grad, neural_pfaffian_params)
    assert_shape_and_dtype(new_state, state)
    assert_finite(grad)
    assert_finite(new_state)
    assert isinstance(aux_data, dict)
