from typing import Any, Sequence

import optax

TransformationConfig = Sequence[tuple[str, Sequence[Any], dict[str, Any]]]


def make_optimizer(
    transformations: TransformationConfig,
    learning_rate: float,
    delay: float,
):
    return optax.chain(
        *[
            getattr(optax, name)(*args, **kwargs)
            for name, args, kwargs in transformations
        ],
        optax.scale_by_schedule(lambda x: -learning_rate / (1 + x / delay)),
    )
