import jax.numpy as jnp
from jaxtyping import Array, Float


def hf_to_full(
    hf_up: Float[Array, '... n_up n_up'], hf_down: Float[Array, '... n_down n_down']
) -> Float[Array, '... n_up+n_down n_up+n_down']:
    n_up, n_down = hf_up.shape[-2], hf_down.shape[-2]
    return jnp.concatenate(
        [
            jnp.concatenate(
                [hf_up, jnp.zeros((*hf_up.shape[:-1], n_down), dtype=hf_up.dtype)],
                axis=-1,
            ),
            jnp.concatenate(
                [jnp.zeros((*hf_down.shape[:-1], n_up), dtype=hf_down.dtype), hf_down],
                axis=-1,
            ),
        ],
        axis=-2,
    )
