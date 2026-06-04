from typing import Protocol

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Bool, Float

from neural_pfaffian.utils import Modules
from neural_pfaffian.utils.jax_utils import jit, pgather_if_pmap
from neural_pfaffian.utils.summary_stats import weighted_mean, weighted_quantile

ClipData = Float[Array, ' batch ...']
"""Generic tensor of data to be clipped. If running on multiple devices,
the first axis is expected to be the batch axis.
Data can be clipped over any number of additional axes."""
Mask = Bool[Array, ' batch ...']
Axis = int | tuple[int, ...] | None


class Clipping(Protocol):
    def __call__(
        self,
        data: ClipData,
        mask: Mask = jnp.ones((), dtype=bool),  # noqa: B008
        reweighting_factor: ClipData = jnp.ones((), dtype=jnp.float32),  # noqa: B008
        *,
        data_is_reweighted: bool = False,
    ) -> ClipData:
        """Clips data by computing summary statistics along the specified axis.

        data_is_reweighted: Indicates whether the input data is already multiplied with the `reweighting_factor`,
            i.e., whether the mean is computed as `jnp.mean(data)` [True] or `jnp.mean(data * reweighting_factor)` [False].
        """
        ...


class NoneClipping(Clipping, PyTreeNode):
    @jit(static_argnames=('data_is_reweighted',))
    def __call__(
        self,
        data: ClipData,
        mask: Mask = jnp.ones((), dtype=bool),  # noqa: B008
        reweighting_factor: ClipData = jnp.ones((), dtype=jnp.float32),  # noqa: B008
        *,
        data_is_reweighted: bool = False,
    ) -> ClipData:
        return data


class MeanClipping(Clipping, PyTreeNode):
    max_deviation: float = field(pytree_node=False)

    @jit(static_argnames=('data_is_reweighted',))
    def __call__(
        self,
        data: ClipData,
        mask: Mask = jnp.ones((), dtype=bool),  # noqa: B008
        reweighting_factor: ClipData = jnp.ones((), dtype=jnp.float32),  # noqa: B008
        *,
        data_is_reweighted: bool = False,
    ) -> ClipData:
        reweighting_factor = jnp.broadcast_to(reweighting_factor, data.shape)
        mask = jnp.broadcast_to(mask, data.shape)

        # We usually assume to get some data that is already multiplied with the reweighting factor.
        if not data_is_reweighted:
            data *= reweighting_factor

        center = weighted_mean(
            data,
            mask,
            reweighting_factor,
            keepdims=True,
            data_is_reweighted=True,
        )

        full_reweighting_factor = pgather_if_pmap(
            reweighting_factor,
            axis=0,
            tiled=True,
        )
        center = center * full_reweighting_factor
        dev = weighted_mean(
            jnp.abs(data - center),
            mask,
            reweighting_factor,
            keepdims=True,
            data_is_reweighted=True,
        )
        max_dev = self.max_deviation * dev * full_reweighting_factor
        data_clipped = jnp.clip(data, center - max_dev, center + max_dev)

        return (
            jnp.where(
                reweighting_factor > 0,
                data_clipped / reweighting_factor,
                0.0,
            )
            if not data_is_reweighted
            else data_clipped
        )


class MedianClipping(Clipping, PyTreeNode):
    max_deviation: float = field(pytree_node=False)

    @jit(static_argnames=('data_is_reweighted',))
    def __call__(
        self,
        data: ClipData,
        mask: Mask = jnp.ones((), dtype=bool),  # noqa: B008
        reweighting_factor: ClipData = jnp.ones((), dtype=jnp.float32),  # noqa: B008
        *,
        data_is_reweighted: bool = False,
    ) -> ClipData:
        reweighting_factor = jnp.broadcast_to(reweighting_factor, data.shape)
        mask = jnp.broadcast_to(mask, data.shape)

        # median is not a linear operation, we can't just move the reweighting factor around
        if data_is_reweighted:
            # safely undo w*x → x; wherever w==0, x=0 (won't matter)
            data = jnp.where(
                reweighting_factor > 0,
                data / reweighting_factor,
                0.0,
            )

        # weight = 0 --> doesn't contribute to the median; effectively giving nanmedian
        center = weighted_quantile(
            data,
            quantile=0.5,
            mask=mask,
            weights=reweighting_factor,
            keepdims=True,
        )
        dev = weighted_mean(
            jnp.abs(data - center),
            mask,
            reweighting_factor,
            keepdims=True,
        )
        max_dev = self.max_deviation * dev
        clipped_data = jnp.clip(data, center - max_dev, center + max_dev)

        return clipped_data * reweighting_factor if data_is_reweighted else clipped_data


class QuantileClipping(Clipping, PyTreeNode):
    max_deviation: float = field(pytree_node=False)
    quantile: float = field(pytree_node=False)

    @jit(static_argnames=('data_is_reweighted',))
    def __call__(
        self,
        data: ClipData,
        mask: Mask = jnp.ones((), dtype=bool),  # noqa: B008
        reweighting_factor: ClipData = jnp.ones((), dtype=jnp.float32),  # noqa: B008
        *,
        data_is_reweighted: bool = False,
    ) -> ClipData:
        reweighting_factor = jnp.broadcast_to(reweighting_factor, data.shape)
        mask = jnp.broadcast_to(mask, data.shape)

        if data_is_reweighted:
            # safely undo w*x → x; wherever w==0, x=0 (won't matter)
            data = jnp.where(
                reweighting_factor > 0,
                data / reweighting_factor,
                0.0,
            )

        center = weighted_quantile(
            data,
            quantile=0.5,
            mask=mask,
            weights=reweighting_factor,
            keepdims=True,
        )
        abs_diffs = jnp.abs(data - center)
        dev_threshold = weighted_quantile(
            abs_diffs,
            quantile=self.quantile,
            mask=mask,
            weights=reweighting_factor,
            keepdims=True,
        )
        max_dev = self.max_deviation * dev_threshold
        center, max_dev = jax.lax.stop_gradient((center, max_dev))
        clipped_data = jnp.clip(data, center - max_dev, center + max_dev)
        return clipped_data * reweighting_factor if data_is_reweighted else clipped_data


class Masking(Protocol):
    def __call__(
        self,
        data: ClipData,
        reweighting_factor: ClipData = jnp.ones((), dtype=jnp.float32),  # noqa: B008
        *,
        data_is_reweighted: bool = False,
    ) -> Mask: ...


class NoneMasking(Masking, PyTreeNode):
    def __call__(
        self,
        data: ClipData,
        reweighting_factor: ClipData = jnp.ones((), dtype=jnp.float32),  # noqa: B008
        *,
        data_is_reweighted: bool = False,
    ) -> Mask:
        return jnp.ones(data.shape, dtype=bool)


class QuantileMasking(Masking, PyTreeNode):
    max_deviation: float = field(pytree_node=False)
    quantile: float = field(pytree_node=False)

    @jit(static_argnames=('data_is_reweighted',))
    def __call__(
        self,
        data: ClipData,
        reweighting_factor: ClipData = jnp.ones((), dtype=jnp.float32),  # noqa: B008
        *,
        data_is_reweighted: bool = False,
    ) -> Mask:
        reweighting_factor = jnp.broadcast_to(reweighting_factor, data.shape)

        if data_is_reweighted:
            # safely undo w*x → x; wherever w==0, x=0 (won't matter)
            data = jnp.where(
                reweighting_factor > 0,
                data / reweighting_factor,
                0.0,
            )

        finite_mask = jnp.isfinite(data)
        center = weighted_quantile(
            data,
            quantile=0.5,
            mask=finite_mask,
            weights=reweighting_factor,
            keepdims=True,
        )
        abs_diffs = jnp.abs(data - center)
        dev_threshold = weighted_quantile(
            abs_diffs,
            quantile=self.quantile,
            mask=finite_mask,
            weights=reweighting_factor,
            keepdims=True,
        )
        max_dev = self.max_deviation * dev_threshold
        center, max_dev = jax.lax.stop_gradient((center, max_dev))
        lower = center - max_dev
        upper = center + max_dev
        return (data >= lower) & (data <= upper) & (jnp.isfinite(data))


class IterativeMeanMasking(Masking, PyTreeNode):
    max_deviation: float = field(pytree_node=False)
    iterations: int = field(pytree_node=False, default=5)

    @jit(static_argnames=('data_is_reweighted',))
    def __call__(
        self,
        data: ClipData,
        reweighting_factor: ClipData = jnp.ones((), dtype=jnp.float32),  # noqa: B008
        *,
        data_is_reweighted: bool = False,
    ) -> Mask:
        mask = jnp.isfinite(data)
        reweighting_factor = jnp.broadcast_to(reweighting_factor, data.shape)
        if not data_is_reweighted:
            data *= reweighting_factor

        if self.max_deviation > 0:
            for _ in range(self.iterations):
                clip_center = weighted_mean(
                    data,
                    mask,
                    reweighting_factor,
                    keepdims=True,
                    data_is_reweighted=True,
                )
                clip_center *= reweighting_factor
                mad = weighted_mean(
                    jnp.abs(data - clip_center),
                    mask,
                    reweighting_factor,
                    keepdims=True,
                    data_is_reweighted=True,
                )
                lower = clip_center - self.max_deviation * mad
                upper = clip_center + self.max_deviation * mad
                mask = (data >= lower) & (data <= upper) & (jnp.isfinite(data))
        return mask


CLIPPINGS = Modules[Clipping](
    {
        cls.__name__.lower().replace('clipping', ''): cls
        for cls in [
            NoneClipping,
            MeanClipping,
            MedianClipping,
            QuantileClipping,
        ]
    },
)

MASKINGS = Modules[Masking](
    {
        'iterative_mean': IterativeMeanMasking,
        'none': NoneMasking,
        'quantile': QuantileMasking,
    },
)
