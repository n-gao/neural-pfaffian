from typing import cast

import jax.numpy as jnp
from jax import Array

from neural_pfaffian.utils.jax_utils import jit, pgather_if_pmap, psum_if_pmap


def _validate_inputs(
    data: Array,
    mask: Array | None,
    weights: Array | None,
) -> tuple[Array, Array, Array]:
    data_array = jnp.asarray(data, dtype=jnp.result_type(data, jnp.float32))

    mask_array = mask if mask is not None else jnp.ones_like(data_array, dtype=jnp.bool_)
    mask_array = jnp.asarray(mask_array, dtype=jnp.bool_)
    mask_array = jnp.broadcast_to(mask_array, data_array.shape)

    if weights is None:
        weight_array = jnp.ones_like(data_array, dtype=data_array.dtype)
    else:
        weight_array = jnp.asarray(weights, dtype=jnp.result_type(weights, data_array))
        weight_array = jnp.broadcast_to(weight_array, data_array.shape)

    return data_array, mask_array, weight_array


def weighted_quantile(
    data: Array,
    quantile: float,
    mask: Array | None = None,
    weights: Array | None = None,
    keepdims: bool = False,
    *,
    data_is_reweighted: bool = False,
):
    """Computes the weighted quantile of the data."""
    data_array, mask_array, weight_array = _validate_inputs(data, mask, weights)
    data_array = cast('Array', pgather_if_pmap(data_array, axis=0, tiled=True))
    mask_array = cast('Array', pgather_if_pmap(mask_array, axis=0, tiled=True))
    weight_array = cast('Array', pgather_if_pmap(weight_array, axis=0, tiled=True))

    if data_is_reweighted:
        data_array = jnp.where(weight_array > 0, data_array / weight_array, 0.0)

    finite_mask = jnp.isfinite(data_array) & jnp.isfinite(weight_array)
    valid = mask_array & finite_mask & (weight_array > 0)

    fill_value = jnp.full_like(data_array, jnp.inf)
    data_for_sort = jnp.where(valid, data_array, fill_value)
    weights_for_sort = jnp.where(valid, weight_array, 0.0)

    sort_idx = jnp.argsort(data_for_sort, axis=0, stable=True)
    data_sorted = jnp.take_along_axis(data_for_sort, sort_idx, axis=0)
    weights_sorted = jnp.take_along_axis(weights_for_sort, sort_idx, axis=0)

    cum_w = jnp.cumsum(weights_sorted, axis=0)
    total_w = cum_w[-1]
    safe_total = jnp.where(total_w > 0.0, total_w, 1.0)
    q = jnp.clip(jnp.asarray(quantile, dtype=safe_total.dtype), 0.0, 1.0)
    target = q * safe_total

    ge = cum_w >= target[None, ...]
    first_idx = jnp.argmax(ge, axis=0)
    quantile_value = jnp.take_along_axis(
        data_sorted,
        first_idx[None, ...],
        axis=0,
    )[0]
    quantile_value = jnp.where(total_w > 0.0, quantile_value, jnp.zeros_like(total_w))

    if keepdims:
        quantile_value = quantile_value[jnp.newaxis, ...]
    return quantile_value


@jit(static_argnames=('keepdims', 'data_is_reweighted'))
def weighted_mean(
    data: Array,
    mask: Array | None = None,
    reweighting_factor: Array | None = None,
    *,
    keepdims: bool = False,
    data_is_reweighted: bool = False,
) -> Array:
    data_array, mask_array, weight_array = _validate_inputs(
        data,
        mask,
        reweighting_factor,
    )

    if not data_is_reweighted:
        data_array = data_array * weight_array

    masked_data = jnp.where(mask_array, data_array, 0.0)
    masked_weights = jnp.where(mask_array, weight_array, 0.0)

    numerator = psum_if_pmap(jnp.sum(masked_data, axis=0, keepdims=keepdims))
    denominator = psum_if_pmap(jnp.sum(masked_weights, axis=0, keepdims=keepdims))
    denominator = jnp.maximum(1.0, denominator)
    return numerator / denominator


@jit(static_argnames=('data_is_reweighted',))
def weighted_centering(
    data: Array,
    mask: Array | None = None,
    reweighting_factor: Array | None = None,
    *,
    data_is_reweighted: bool = False,
) -> Array:
    data_array, _mask_array, weight_array = _validate_inputs(
        data,
        mask,
        reweighting_factor,
    )
    if data_is_reweighted:
        data_unweighted = jnp.where(weight_array > 0, data_array / weight_array, 0.0)
    else:
        data_unweighted = data_array

    mean = weighted_mean(
        data_unweighted,
        mask=mask,
        reweighting_factor=reweighting_factor,
        keepdims=True,
        data_is_reweighted=False,
    )
    center = data_unweighted - mean
    return center * weight_array if data_is_reweighted else center


@jit(static_argnames=('keepdims', 'data_is_reweighted'))
def weighted_variance(
    data: Array,
    mask: Array | None = None,
    reweighting_factor: Array | None = None,
    *,
    keepdims: bool = False,
    data_is_reweighted: bool = False,
) -> Array:
    data_array, mask_array, weight_array = _validate_inputs(
        data,
        mask,
        reweighting_factor,
    )
    if data_is_reweighted:
        data_unweighted = jnp.where(weight_array > 0, data_array / weight_array, 0.0)
    else:
        data_unweighted = data_array

    mean = weighted_mean(
        data_unweighted,
        mask=mask_array,
        reweighting_factor=weight_array,
        keepdims=True,
        data_is_reweighted=False,
    )
    centered_sq = jnp.square(data_unweighted - mean)
    variance = weighted_mean(
        centered_sq,
        mask=mask_array,
        reweighting_factor=weight_array,
        keepdims=keepdims,
        data_is_reweighted=False,
    )
    return variance


@jit(static_argnames=('keepdims', 'data_is_reweighted'))
def weighted_std(
    data: Array,
    mask: Array | None = None,
    reweighting_factor: Array | None = None,
    *,
    keepdims: bool = False,
    data_is_reweighted: bool = False,
) -> Array:
    variance = weighted_variance(
        data,
        mask=mask,
        reweighting_factor=reweighting_factor,
        keepdims=keepdims,
        data_is_reweighted=data_is_reweighted,
    )
    return jnp.sqrt(jnp.maximum(variance, 0.0))
