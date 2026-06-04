from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import optax

ScheduleConfig = str | dict[str, Any] | tuple[str, *Sequence[Any]] | int | float
"""Configures a schedule from optax or here. If numeric val (float | int), interpreted as constant schedule."""


def hyperbolic_decay(
    init_learning_rate: float,
    delay: float,
    asymptote: float = 0.0,
) -> optax.Schedule:
    """Hyperbolic decay schedule.
    Decays from `init_learning_rate` to `asymptote` as `1 / (1 + step / delay)`.
    In case `asymptote` > `init_learning_rate`, this is a growth schedule.
    """
    return lambda step: asymptote + (init_learning_rate - asymptote) / (1 + step / delay)


def clipped_hyperbolic_growth(
    init_value: float,
    delay: float,
    bound: float,
) -> optax.Schedule:
    """Hyperbolic growth schedule truncated at `bound`.
    Grows from `init_value` to `bound` as `1 - 1 / (1 + step / delay)`.
    Both `init_value` and `bound` are assumed to lie in (0, 1).
    """

    def schedule(step):
        base_term = 1.0 / (1.0 - init_value)
        value = 1.0 - 1.0 / (base_term + step / delay)
        return jnp.minimum(value, bound)

    return schedule


def piecewise_constant_schedule(
    init_value: float,
    boundaries_and_scales: dict[int, float] | None = None,
) -> optax.Schedule:
    """Took the function from optax and added a cast to threshold, because yaml parser"""
    if boundaries_and_scales is not None:
        all_positive = all(scale >= 0.0 for scale in boundaries_and_scales.values())
        if not all_positive:
            raise ValueError(
                '`piecewise_constant_schedule` expects non-negative scale factors',
            )

    def schedule(count):
        v = init_value
        if boundaries_and_scales is not None:
            for threshold, scale in sorted(boundaries_and_scales.items()):
                indicator = jnp.maximum(0.0, jnp.sign(int(threshold) - count))
                v = v * indicator + (1 - indicator) * scale * v
        return v

    return schedule


def sigmoid(
    init_value: float,
    end_value: float,
    midpoint_step: int,
    transition_steps: int,
) -> optax.Schedule:
    """Sigmoid schedule from `init_value` to `end_value` centered at `midpoint_step`.
    90% of the transition happens within `transition_steps`."""

    k = jnp.asarray(transition_steps / (2.0 * jnp.log(9.0)))

    def schedule(step):
        s = jax.nn.sigmoid((step - midpoint_step) / k)
        return init_value + (end_value - init_value) * s

    return schedule


def get_schedule(
    schedule: ScheduleConfig,
) -> optax.Schedule:
    if isinstance(schedule, str):
        name = schedule
        args, kwargs = [], {}
    elif isinstance(schedule, dict):
        schedule = schedule.copy()
        name = schedule.pop('schedule')
        args, kwargs = [], schedule
    elif isinstance(schedule, Sequence):
        name = schedule[0]
        args, kwargs = list(schedule[1:]), {}
    elif isinstance(schedule, int | float):
        name = 'constant_schedule'
        args, kwargs = [schedule], {}
    else:
        raise ValueError(f'Unknown schedule type: {type(schedule)}')

    if name in globals():
        constructor = globals()[name]
    elif hasattr(optax.schedules, name):
        constructor = getattr(optax.schedules, name)
    else:
        raise ValueError(f'Unknown schedule {name}')
    return constructor(*args, **kwargs)
