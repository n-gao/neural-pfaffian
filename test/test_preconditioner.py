import jax
import jax.numpy as jnp
from fixtures import *  # noqa: F403
from utils import assert_finite, assert_shape_and_dtype

from neural_pfaffian.utils.jax_utils import BATCH_SPEC, REPLICATE_SPEC, shmap


def test_preconditioner(preconditioner, neural_pfaffian_params, batched_systems):
    state = preconditioner.init(
        jax.random.PRNGKey(0),
        neural_pfaffian_params,
        batched_systems,
    )
    apply = shmap(
        preconditioner.apply,
        in_specs=(
            REPLICATE_SPEC,
            batched_systems.partition_spec,
            BATCH_SPEC,
            REPLICATE_SPEC,
            REPLICATE_SPEC,
        ),
        out_specs=REPLICATE_SPEC,
        check_vma=False,
    )
    apply = jax.jit(apply)
    dE_dlogpsi = jax.random.normal(
        jax.random.PRNGKey(123),
        (*batched_systems.electrons.shape[:-2], batched_systems.n_mols),
        dtype=batched_systems.electrons.dtype,
    )

    grad, new_state, aux_data = apply(
        neural_pfaffian_params,
        batched_systems,
        dE_dlogpsi,
        state,
        jax.tree.map(lambda x: jnp.zeros_like(x), neural_pfaffian_params),
    )
    assert_shape_and_dtype(grad, neural_pfaffian_params)
    assert_shape_and_dtype(new_state, state)
    assert_finite(grad)
    assert_finite(new_state)
    assert isinstance(aux_data, dict)
