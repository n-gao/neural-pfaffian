from typing import Any, Callable, Sequence, cast

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jaxtyping import Array, Float

from neural_pfaffian.utils.jax_utils import pmean_if_pmap

Transform = str | dict[str, Any] | tuple[str, *Sequence[Any]]
TransformationConfig = Sequence[Transform]


def scale_by_hyperbolic_schedule(learning_rate: float, delay: float):
    return optax.scale_by_schedule(lambda x: -learning_rate / (1 + x / delay))


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


def get_transformations(
    transformations: TransformationConfig,
) -> list[optax.GradientTransformation]:
    def get_transform(transform: Transform):
        if isinstance(transform, str):
            name = transform
            args, kwargs = [], {}
        elif isinstance(transform, dict):
            transform = transform.copy()
            name = transform.pop('transform')
            args, kwargs = [], transform
        else:
            name = transform[0]
            args, kwargs = transform[1:], {}

        if name in globals():
            constructor = globals()[name]
        elif hasattr(optax, name):
            constructor = getattr(optax, name)
        else:
            raise ValueError(f'Unknown transformation {name}')
        return constructor(*args, **kwargs)

    return [get_transform(transform) for transform in transformations]


def filter_by_param(name: str | Sequence[str], transformations: TransformationConfig):
    if isinstance(name, str):
        name = [name]

    def mask(params):
        def _mask(path, tensor):
            try:
                tensor_name = getattr(path[-1], 'name', getattr(path[-1], 'key', ''))
                return any(n in tensor_name for n in name)
            except Exception:
                return False

        return jtu.tree_map_with_path(_mask, params)

    return optax.masked(optax.chain(*get_transformations(transformations)), mask)


def make_optimizer(transformations: TransformationConfig):
    return optax.chain(*get_transformations(transformations))


Loss = Float[Array, '']


def optimize[T](
    target: Callable[[T], Loss],
    x: T,
    optimizer: optax.GradientTransformation,
    steps: int,
) -> tuple[T, Loss]:
    def avg_loss_and_grad(x: T) -> tuple[Loss, T]:
        return pmean_if_pmap(jax.value_and_grad(target)(x))

    def step(state: tuple[T, optax.OptState], i) -> tuple[tuple[T, optax.OptState], Loss]:
        params, opt_state = state
        value, grads = avg_loss_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)  # type: ignore
        params = cast(T, optax.apply_updates(params, updates))  # type: ignore
        return (params, opt_state), value

    (x, _), loss = jax.lax.scan(
        step,
        (x, optimizer.init(x)),  # type: ignore
        jnp.arange(steps),
    )
    return x, loss[-1]
