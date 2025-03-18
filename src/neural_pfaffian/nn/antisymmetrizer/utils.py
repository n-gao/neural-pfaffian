import jax.numpy as jnp
from jaxtyping import Array, Float


def hf_to_full(
    hf_up: Float[Array, '... n_up n_up'],
    hf_down: Float[Array, '... n_down n_down'],
    n_orbs: int | tuple[int, int] | None = None,
) -> Float[Array, '... n_up+n_down n_up+n_down']:
    n_up, n_down = hf_up.shape[-2], hf_down.shape[-2]
    match n_orbs:
        case None:
            n_up_target, n_down_target = n_up, n_down
        case int():
            n_up_target, n_down_target = n_orbs, n_orbs
        case (n_up_target, n_down_target):
            pass
        case _:
            raise ValueError(f'Invalid n_orbs: {n_orbs}')
    assert n_up_target >= n_up and n_down_target >= n_down
    hf_up = jnp.concatenate(
        [
            hf_up,
            jnp.zeros(
                (*hf_up.shape[:-1], n_up_target + n_down_target - n_up), dtype=hf_up.dtype
            ),
        ],
        axis=-1,
    )
    hf_down = jnp.concatenate(
        [
            jnp.zeros((*hf_down.shape[:-1], n_up_target), dtype=hf_down.dtype),
            hf_down,
            jnp.zeros(
                (*hf_down.shape[:-1], n_down_target - n_down),
                dtype=hf_down.dtype,
            ),
        ],
        axis=-1,
    )
    return jnp.concatenate([hf_up, hf_down], axis=-2)
