import jax
import jax.numpy as jnp
import numpy as np

from neural_pfaffian.utils.segment_utils import unsegment_axis
from neural_pfaffian.utils.summary_stats import (
    weighted_centering,
    weighted_quantile,
    weighted_std,
    weighted_variance,
)


def test_unsegment_axis():
    array = jnp.arange(12).reshape(2, 6)
    segment_ids = jnp.array([0, 1, 0, 1, 2, 2])

    result = unsegment_axis(array, segment_ids, axis=1)
    assert result.shape == (2, 3, 2)
    assert np.all(np.asarray(array[:, [0, 2]]) == np.asarray(result[:, 0]))

    # Test sorted
    segment_ids = jnp.array([0, 0, 1, 1, 2, 2])
    result = unsegment_axis(array.T, segment_ids, axis=0, indices_are_grouped=True)
    assert result.shape == (3, 2, 2)
    assert np.all(np.asarray(array.T[:2, :]) == np.asarray(result[0]))


def test_unsegment_axis_preserves_encounter_order():
    array = jnp.arange(12).reshape(2, 6)
    # Segment IDs appear in encounter order [1, 0, 2]
    segment_ids = jnp.array([1, 1, 0, 0, 2, 2])
    result = unsegment_axis(array, segment_ids, axis=1)
    # If encounter order were preserved, the first block would match the first-occurring segment (id=1)
    encounter_order_block = np.asarray(array[:, :2])
    assert np.array_equal(np.asarray(result[:, 0]), encounter_order_block)


def test_jitted_unsegment_axis():
    array = jnp.arange(12).reshape(2, 6)
    segment_ids = jnp.array([0, 1, 0, 1, 2, 2])

    jitted_unsegment = jax.jit(unsegment_axis, static_argnames=['axis', 'num_segments'])
    result = jitted_unsegment(array, segment_ids, axis=1, num_segments=3)
    assert result.shape == (2, 3, 2)
    assert np.all(np.asarray(array[:, [0, 2]]) == np.asarray(result[:, 0]))


def test_weighted_quantile_basic():
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    weights = jnp.array([1.0, 1.0, 2.0, 4.0])
    result = weighted_quantile(data, quantile=0.5, weights=weights)
    # Weighted median should be 3.0 because cumulative weights reach 0.5*total at 3.0
    np.testing.assert_allclose(np.asarray(result), 3.0)


def test_weighted_quantile_with_mask_and_keepdims():
    data = jnp.array([0.0, 10.0, 20.0, 30.0])
    weights = jnp.array([1.0, 1.0, 1.0, 1.0])
    mask = jnp.array([True, False, True, True])
    result = weighted_quantile(
        data,
        quantile=0.75,
        mask=mask,
        weights=weights,
        keepdims=True,
    )
    # Only entries 0,20,30 remain; 75th percentile is 30.
    assert result.shape == (1,)
    np.testing.assert_allclose(np.asarray(result), 30.0)


def test_weighted_quantile_reweighted_input():
    raw = jnp.array([1.0, 2.0, 5.0])
    weights = jnp.array([1.0, 3.0, 6.0])
    reweighted = raw * weights

    baseline = weighted_quantile(raw, quantile=0.8, weights=weights)
    reweighted_result = weighted_quantile(
        reweighted,
        quantile=0.8,
        weights=weights,
        data_is_reweighted=True,
    )

    np.testing.assert_allclose(np.asarray(reweighted_result), np.asarray(baseline))


def test_weighted_variance_and_std_basic():
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    weights = jnp.ones_like(data)
    var = weighted_variance(data, reweighting_factor=weights)
    std = weighted_std(data, reweighting_factor=weights)
    np.testing.assert_allclose(np.asarray(var), 1.25)
    np.testing.assert_allclose(np.asarray(std), np.sqrt(1.25))


def test_weighted_variance_with_mask_and_keepdims():
    data = jnp.array([0.0, 10.0, 20.0, 30.0])
    mask = jnp.array([True, False, True, True])
    weights = jnp.ones_like(data)
    var = weighted_variance(
        data,
        mask=mask,
        reweighting_factor=weights,
        keepdims=True,
    )
    std = weighted_std(
        data,
        mask=mask,
        reweighting_factor=weights,
        keepdims=True,
    )
    expected_values = np.array([0.0, 20.0, 30.0])
    mean = expected_values.mean()
    expected_var = np.mean((expected_values - mean) ** 2)
    np.testing.assert_allclose(np.asarray(var).squeeze(), expected_var)
    np.testing.assert_allclose(np.asarray(std).squeeze(), np.sqrt(expected_var))


def test_weighted_variance_reweighted_input():
    raw = jnp.array([1.0, 5.0, 7.0])
    weights = jnp.array([1.0, 2.0, 3.0])
    reweighted = raw * weights

    baseline_var = weighted_variance(raw, reweighting_factor=weights)
    baseline_std = weighted_std(raw, reweighting_factor=weights)

    reweighted_var = weighted_variance(
        reweighted,
        reweighting_factor=weights,
        data_is_reweighted=True,
    )
    reweighted_std = weighted_std(
        reweighted,
        reweighting_factor=weights,
        data_is_reweighted=True,
    )

    np.testing.assert_allclose(np.asarray(reweighted_var), np.asarray(baseline_var))
    np.testing.assert_allclose(np.asarray(reweighted_std), np.asarray(baseline_std))


def test_weighted_centering_basic():
    data = jnp.array([1.0, 2.0, 3.0])
    centered = weighted_centering(data)
    np.testing.assert_allclose(np.asarray(centered), np.array([-1.0, 0.0, 1.0]))


def test_weighted_centering_with_mask_and_weights():
    data = jnp.array([1.0, 10.0, 20.0, 30.0])
    mask = jnp.array([True, False, True, True])
    weights = jnp.array([1.0, 1.0, 2.0, 1.0])
    centered = weighted_centering(
        data,
        mask=mask,
        reweighting_factor=weights,
    )
    # Weighted mean excludes masked element and accounts for weights: mean = (1*1 + 2*20 + 1*30)/(1+2+1)=17.75
    expected = data - 17.75
    np.testing.assert_allclose(np.asarray(centered), expected)


def test_weighted_centering_reweighted_input():
    raw = jnp.array([1.0, 2.0, 6.0])
    weights = jnp.array([1.0, 2.0, 3.0])
    reweighted = raw * weights

    baseline = weighted_centering(raw, reweighting_factor=weights)
    reweighted_centered = weighted_centering(
        reweighted,
        reweighting_factor=weights,
        data_is_reweighted=True,
    )

    np.testing.assert_allclose(
        np.asarray(reweighted_centered),
        np.asarray(baseline * weights),
    )
