from typing import Any, Sequence

import jax.numpy as jnp
import jax.tree_util as jtu
import optax

TransformationConfig = Sequence[tuple[str, Sequence[Any], dict[str, Any]]]


def scale_by_trust_ratio_embeddings(
    min_norm: float = 0.0,
    trust_coefficient: float = 1.0,
    eps: float = 0.0,
) -> optax.GradientTransformation:
    """Scale by trust ratio but for embeddings were we don't want the norm
    over all parameters but just the last dimension.
    """

    def init_fn(params):
        del params
        return optax.ScaleByTrustRatioState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError('params cannot be None')

        def _scale_update(update, param):
            # Clip norms to minimum value, by default no clipping.
            param_norm = optax.safe_norm(param, min_norm, axis=-1, keepdims=True)
            update_norm = optax.safe_norm(update, min_norm, axis=-1, keepdims=True)
            trust_ratio = trust_coefficient * param_norm / (update_norm + eps)

            # If no minimum norm clipping is used
            # Set trust_ratio to 1 in case where parameters would never be updated.
            zero_norm = jnp.logical_or(param_norm == 0.0, update_norm == 0.0)
            safe_trust_ratio = jnp.where(
                zero_norm, jnp.array(1.0, dtype=param.dtype), trust_ratio
            )

            return update * safe_trust_ratio

        updates = jtu.tree_map(_scale_update, updates, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore


def get_transformations(transformations: TransformationConfig):
    def get_transform(name: str):
        if name in globals():
            return globals()[name]
        elif hasattr(optax, name):
            return getattr(optax, name)
        else:
            raise ValueError(f'Unknown transformation {name}')

    return [
        get_transform(name)(*args, **kwargs) for name, args, kwargs in transformations
    ]


def filter_by_path(name: str, transformations: TransformationConfig):
    def mask(params):
        def _mask(path, tensor):
            try:
                return name in path[-1].key
            except Exception:
                return False

        return jtu.tree_map_with_path(_mask, params)

    return optax.masked(optax.chain(*get_transformations(transformations)), mask)


def make_optimizer(
    transformations: TransformationConfig,
    learning_rate: float,
    delay: float,
):
    return optax.chain(
        *get_transformations(transformations),
        optax.scale_by_schedule(lambda x: -learning_rate / (1 + x / delay)),
    )
