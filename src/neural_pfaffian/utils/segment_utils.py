import einops
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, ArrayLike, Float

segment_sum = jax.ops.segment_sum


def unsegment_axis(
    data: Float[Array, ' ... n_segments*segment_size'],
    segment_ids: ArrayLike,
    axis: int = 0,
    indices_are_grouped: bool = False,
    num_segments: int | None = None,
) -> Float[Array, ' ... n_segments segment_size']:
    """Unsegments the data along the given axis.

    Args:
      data: the data to be unsegmented.
      segment_ids: the segment ids.
      axis: the axis to unsegment.
      indices_are_grouped: whether ``segment_ids`` is known to be grouped, i.e., [0, 0, 2, 2, 1, 1] vs [0, 1, 0, 1, 2, 2].
        If True, a more efficient algorithm is used, since no sorting is necessary.
        If False, the algorithm will first sort the data according to segment_ids while retaining
        the original segment order.
      num_segments: optional, an int with nonnegative value indicating the number of segments.
        The default is set to be the minimum number of segments that would support all
        indices in segment_ids, calculated as np.unique(segment_ids).size.
        Since num_segments determines the size of the output, a static value must be
        provided to use segment_sum in a JIT-compiled function.

    Returns:
      The unsegmented data.
    """
    if num_segments is None:
        concrete_segments = jax_core.concrete_or_error(
            lambda sid: int(np.unique(np.asarray(sid)).size),
            segment_ids,
            'num_segments must be provided when segment_ids is not statically known.',
        )
        num_segments_int = int(concrete_segments)
    else:
        try:
            num_segments_int = int(num_segments)
        except TypeError as exc:
            raise ValueError('num_segments must be a concrete integer.') from exc
    num_segments = num_segments_int

    segment_ids = jnp.asarray(segment_ids)
    total_elements = data.shape[axis]
    assert total_elements % num_segments == 0, (
        f'Total elements ({total_elements}) along axis {axis} must be divisible '
        f'by the number of segments ({num_segments})'
    )
    n = segment_ids.shape[0]
    positions = jnp.arange(n, dtype=jnp.int32)

    if not indices_are_grouped:
        ids = segment_ids
        eq = ids[:, None] == ids[None, :]
        le = positions[:, None] <= positions[None, :]
        first_idx = jnp.min(
            jnp.where(eq & le, positions[:, None], n),
            axis=0,
        )
        group_key = first_idx * n + positions
        sort_indices = jnp.argsort(group_key, stable=True)
    else:
        sort_indices = positions

    data = jnp.moveaxis(data, axis, -1)
    data = data[..., sort_indices]
    data = einops.rearrange(
        data,
        '... (n_segments segment_size) -> ... n_segments segment_size',
        n_segments=num_segments,
    )
    data = jnp.moveaxis(data, -1, axis)
    data = jnp.moveaxis(data, -1, axis)
    return data


def segment_argmin(
    data: jax.Array,
    segment_ids: ArrayLike,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> jax.Array:
    """
    For each segment, returns the index of the first minimal element in `data`.
    """
    segment_ids = jnp.asarray(segment_ids)
    n = data.shape[0]
    if num_segments is None:
        num_segments = int(jnp.max(segment_ids)) + 1

    # get per-segment min
    mins = jax.ops.segment_min(
        data,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
    )
    # mask those entries that equal the min
    eq_min = data == mins[segment_ids]
    # build a global index array
    idxs = jnp.arange(n)
    # pick only indices where eq_min, else set to n
    valid = jnp.where(eq_min, idxs, n)
    # segment-min over these indices → first occurrence
    argmins = jax.ops.segment_min(
        valid,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
    )
    return argmins


def segment_softmax(
    logits: jax.Array,
    segment_ids: ArrayLike,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
):
    """Computes a segment-wise softmax.

    For a given tree of logits that can be divded into segments, computes a
    softmax over the segments.

      logits = jnp.ndarray([1.0, 2.0, 3.0, 1.0, 2.0])
      segment_ids = jnp.ndarray([0, 0, 0, 1, 1])
      segment_softmax(logits, segments)
      >> DeviceArray([0.09003057, 0.24472848, 0.66524094, 0.26894142, 0.7310586],
      >> dtype=float32)

    Args:
      logits: an array of logits to be segment softmaxed.
      segment_ids: an array with integer dtype that indicates the segments of
        `data` (along its leading axis) to be maxed over. Values can be repeated
        and need not be sorted. Values outside of the range [0, num_segments) are
        dropped and do not contribute to the result.
      num_segments: optional, an int with positive value indicating the number of
        segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
        jnp.max(-segment_ids))`` but since ``num_segments`` determines the size of
        the output, a static value must be provided to use ``segment_sum`` in a
        ``jit``-compiled function.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted
      unique_indices: whether ``segment_ids`` is known to be free of duplicates

    Returns:
      The segment softmax-ed ``logits``.
    """
    # First, subtract the segment max for numerical stability
    maxs = jax.ops.segment_max(
        logits,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
    )
    logits = logits - maxs[segment_ids]
    # Then take the exp
    logits = jnp.exp(logits)
    # Then calculate the normalizers
    normalizers = segment_sum(
        logits,
        segment_ids,
        num_segments,
        indices_are_sorted,
        unique_indices,
    )
    normalizers = normalizers[segment_ids]
    softmax = logits / normalizers
    return softmax


def segment_mean(
    data: jax.Array,
    segment_ids: ArrayLike,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
):
    """Returns mean for each segment.

    Args:
      data: the values which are averaged segment-wise.
      segment_ids: indices for the segments.
      num_segments: total number of segments.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted.
      unique_indices: whether ``segment_ids`` is known to be free of duplicates.
    """
    nominator = segment_sum(
        data,
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    denominator = segment_sum(
        jnp.ones_like(data),
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    return nominator / jnp.maximum(
        denominator,
        jnp.ones(shape=[], dtype=denominator.dtype),
    )
