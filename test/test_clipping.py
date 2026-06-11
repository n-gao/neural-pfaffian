import jax.numpy as jnp
from numpy.testing import assert_allclose
from utils import assert_shape_and_dtype

from neural_pfaffian.clipping import (
    IterativeMeanMasking,
    MeanClipping,
    MedianClipping,
    NoneClipping,
    QuantileClipping,
)


def test_mean_clipping():
    # Test negative clipping
    clipping = MeanClipping(1.0)
    x = jnp.array([-1, 0, 1], dtype=jnp.float32)[:, None]
    y = clipping(x)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, x * 2 / 3)

    # Test positive clipping
    clipping = MeanClipping(1.0)
    x = jnp.array([-1, 0, 13], dtype=jnp.float32)[:, None]
    # x.mean() = 4.0; abs(dev) = [5, 4, 9] -> mean of abs(dev) = 6.0
    # clip bounds = [-2.0, 10.0]
    y = clipping(x)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, jnp.array([-1, 0, 10], dtype=jnp.float32)[:, None])


def test_mean_clipping_masked():
    clipping = MeanClipping(1.0)
    x = jnp.array([-1, 0, 13, 1000], dtype=jnp.float32)[:, None]
    # the mask will prevent 1000 from contributing to the mean
    mask = jnp.array([True, True, True, False], dtype=bool)[:, None]
    y = clipping(x, mask=mask)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, jnp.array([-1, 0, 10, 10], dtype=jnp.float32)[:, None])


def test_mean_clipping_weighted():
    clipping = MeanClipping(1.0)
    x = jnp.array([1, 0, 3], dtype=jnp.float32)[:, None]
    weights = jnp.array([1.5, 1.0, 0.5], dtype=jnp.float32)[:, None]
    # weighted mean = 1
    # weighted MAD = 2/3
    # clip bounds = [1/3, 5/3]
    y = clipping(x, reweighting_factor=weights, data_is_reweighted=False)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, jnp.array([1.0, 1 / 3, 5 / 3], dtype=jnp.float32)[:, None])

    wx = x * weights
    y = clipping(wx, reweighting_factor=weights, data_is_reweighted=True)
    assert_shape_and_dtype(y, wx)
    assert_allclose(
        y,
        jnp.array([1.0, 1 / 3, 5 / 3], dtype=jnp.float32)[:, None] * weights,
    )


def test_mean_clipping_masked_weighted():
    clipping = MeanClipping(1.0)

    x = jnp.array([1.0, 10.0, 100.0, 1000.0], dtype=jnp.float32)[:, None]
    mask = jnp.array([True, True, False, False], dtype=bool)[:, None]
    # choose weights summing to N=4
    weights = jnp.array([1.0, 2.0, 0.5, 0.5], dtype=jnp.float32)[:, None]
    # masked_weights = [1.0,2.0,0.0,0.0] contributions; sum w=3.0
    # weighted mean  = (1*1 + 10*2) / 3 = 21/3 = 7.0
    # weighted MAD   = (|1-7|*1 + |10-7|*2) / 3
    #                = 12 / 3 = 4
    # bounds         = [3, 11]
    expected_raw = jnp.array([3.0, 10.0, 11.0, 11.0], dtype=jnp.float32)[:, None]

    # raw-x path
    y1 = clipping(
        x,
        mask=mask,
        reweighting_factor=weights,
        data_is_reweighted=False,
    )
    assert_shape_and_dtype(y1, x)
    assert_allclose(y1, expected_raw, atol=1e-6)

    wx = x * weights
    y2 = clipping(
        wx,
        mask=mask,
        reweighting_factor=weights,
        data_is_reweighted=True,
    )
    expected_wx = expected_raw * weights
    assert_shape_and_dtype(y2, wx)
    assert_allclose(y2, expected_wx, atol=1e-6)


def test_median_clipping():
    # Test negative clipping around median=0
    clipping = MedianClipping(1.0)
    x = jnp.array([-1, 0, 1], dtype=jnp.float32)[:, None]
    # median = 0, abs_devs = [1,0,1], mean_dev = 2/3 → bounds = [-2/3,2/3]
    y = clipping(x)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, x * (2 / 3))

    # Test positive clipping
    clipping = MedianClipping(1.0)
    x = jnp.array([-1, 0, 13], dtype=jnp.float32)[:, None]
    # median =  0, abs_devs = [1,0,13], mean_dev = (1+0+13)/3 = 14/3 ≈4.6667
    # bounds = [-4.6667, +4.6667]
    y = clipping(x)
    expected = jnp.array([-1.0, 0.0, 4.6666665], dtype=jnp.float32)[:, None]
    assert_shape_and_dtype(y, x)
    assert_allclose(y, expected, atol=1e-6)


def test_median_clipping_masked():
    clipping = MedianClipping(1.0)
    x = jnp.array([-1, 0, 13, 1000], dtype=jnp.float32)[:, None]
    mask = jnp.array([True, True, True, False], dtype=bool)[:, None]
    # mask removes 1000 → median over [-1,0,13] is 0, dev = 14/3
    # so 1000 gets clipped down to +4.6667
    y = clipping(x, mask=mask)
    expected = jnp.array([-1.0, 0.0, 4.6666665, 4.6666665], dtype=jnp.float32)[:, None]
    assert_shape_and_dtype(y, x)
    assert_allclose(y, expected, atol=1e-6)


def test_median_clipping_weighted():
    clipping = MedianClipping(1.0)
    x = jnp.array([1, 0, 3], dtype=jnp.float32)[:, None]
    weights = jnp.array([0.5, 2.0, 0.5], dtype=jnp.float32)[:, None]
    #   weighted median = 0
    #   weighted MAD  = 2/3
    #   clip bounds  = [0 ± 2/3] = [-0.6666667, 0.6666667]
    expected_raw = jnp.array([0.6666667, 0.0, 0.6666667], dtype=jnp.float32)[:, None]

    y = clipping(x, reweighting_factor=weights, data_is_reweighted=False)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, expected_raw, atol=1e-6)

    # now feed in pre-weighted data (w*x) and ask for data_is_reweighted=True
    wx = x * weights
    expected_wx = expected_raw * weights
    y = clipping(wx, reweighting_factor=weights, data_is_reweighted=True)
    assert_shape_and_dtype(y, wx)
    assert_allclose(y, expected_wx, atol=1e-6)


def test_median_clipping_masked_weighted():
    clipping = MedianClipping(1.0)

    x = jnp.array([1.0, 2.0, 100.0, 1000.0], dtype=jnp.float32)[:, None]
    mask = jnp.array([True, True, False, False], dtype=bool)[:, None]
    weights = jnp.array([3.0, 0.5, 0.25, 0.25], dtype=jnp.float32)[:, None]  # sums to 4
    # weighted median = 1.0
    # weighted MAD    = (|1-1|*3 + |2-1|*0.5) / 3.5 = 0.5 / 3.5
    # bounds          = [1 ± 0.5/3.5] = [0.8571429, 1.1428571]
    expected_raw = jnp.array([1.0, 1.1428571, 1.1428571, 1.1428571], dtype=jnp.float32)[
        :,
        None,
    ]

    # raw-x path
    y1 = clipping(
        x,
        mask=mask,
        reweighting_factor=weights,
        data_is_reweighted=False,
    )
    assert_shape_and_dtype(y1, x)
    assert_allclose(y1, expected_raw, atol=1e-6)

    # preweighted path
    wx = x * weights
    y2 = clipping(
        wx,
        mask=mask,
        reweighting_factor=weights,
        data_is_reweighted=True,
    )
    expected_wx = expected_raw * weights
    assert_shape_and_dtype(y2, wx)
    assert_allclose(y2, expected_wx, atol=1e-6)


def test_quantile_clipping():
    clipping = QuantileClipping(max_deviation=1.0, quantile=0.5)

    x = jnp.array([0.0, 1.0, 3.0, 10.0], dtype=jnp.float32)[:, None]
    # median = 1, abs_devs = [1,0,2,9] → sorted [0,1,2,9], target=2 → dev_thresh=1
    # bounds = [0,2] → clipped = [0,1,2,2]
    expected_unweighted = jnp.array([0.0, 1.0, 2.0, 2.0], dtype=jnp.float32)[:, None]
    y_unweighted = clipping(x)
    assert_shape_and_dtype(y_unweighted, x)
    assert_allclose(y_unweighted, expected_unweighted, atol=1e-6)


def test_quantile_clipping_to_median():
    clipping = QuantileClipping(max_deviation=1.0, quantile=0.0)
    x = jnp.array([2.0, 0.0, 5.0], dtype=jnp.float32)[:, None]
    # median = 2.0
    # dev_quantile=0 → max_dev=0 → bounds = [2.0,2.0] → all clipped to 2.0
    expected = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)[:, None]
    y = clipping(x)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, expected, atol=1e-6)

    weights = jnp.array([0.5, 2.0, 0.5], dtype=jnp.float32)[:, None]  # sums to 3.0
    wx = x * weights
    # weighted median of [2,0,5] under [0.5,2.0,0.5] is 0.0 → all clipped to 0.0
    expected_x = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)[:, None]
    expected_wx = expected_x * weights
    y = clipping(x, reweighting_factor=weights, data_is_reweighted=False)
    y2 = clipping(wx, reweighting_factor=weights, data_is_reweighted=True)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, expected_x, atol=1e-6)
    assert_shape_and_dtype(y2, wx)
    assert_allclose(y2, expected_wx, atol=1e-6)


def test_quantile_clipping_weighted():
    clipping = QuantileClipping(max_deviation=1.0, quantile=0.5)
    x = jnp.array([0.0, 1.0, 3.0, 10.0], dtype=jnp.float32)[:, None]
    weights = jnp.array([1.0, 2.0, 0.5, 0.5], dtype=jnp.float32)[:, None]
    # cumulative weights for median target=2 → weighted median = 1
    # abs_devs = [1,0,2,9] with weights [1,2,0.5,0.5]
    # sorted abs_devs & weights → [0(w=2),1(w=1),2(w=0.5),9(w=0.5)], cum=[2,3,3.5,4], target=2 → dev_thresh=0
    # bounds = [1±0] → clipped x = [1,1,1,1]
    expected_weighted_raw = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)[:, None]
    y_weighted_raw = clipping(x, reweighting_factor=weights, data_is_reweighted=False)
    assert_shape_and_dtype(y_weighted_raw, x)
    assert_allclose(y_weighted_raw, expected_weighted_raw, atol=1e-6)

    # 3) Preweighted data case:
    wx = x * weights  # [0,2,1.5,5]
    expected_preweighted = expected_weighted_raw * weights
    y_preweighted = clipping(wx, reweighting_factor=weights, data_is_reweighted=True)
    assert_shape_and_dtype(y_preweighted, wx)
    assert_allclose(y_preweighted, expected_preweighted, atol=1e-6)


def test_quantile_clipping_masked():
    clipping = QuantileClipping(max_deviation=1.0, quantile=0.5)
    x = jnp.array([0.0, 1.0, 3.0, 10.0], dtype=jnp.float32)[:, None]
    mask = jnp.array([True, True, True, False], dtype=bool)[:, None]
    # mask removes 10 → median over [0,1,3] is 1 → abs_devs = [1,0,2] → dev_thresh=1
    # bounds = [0,2] → clipped = [0,1,2,2]
    expected_masked = jnp.array([0.0, 1.0, 2.0, 2.0], dtype=jnp.float32)[:, None]
    y_masked = clipping(x, mask=mask)
    assert_shape_and_dtype(y_masked, x)
    assert_allclose(y_masked, expected_masked, atol=1e-6)


def test_quantile_clipping_masked_weighted():
    clipping = QuantileClipping(max_deviation=1.0, quantile=0.5)

    x = jnp.array([1.0, 2.0, 100.0, 1000.0], dtype=jnp.float32)[:, None]
    mask = jnp.array([True, True, False, False], dtype=bool)[:, None]
    weights = jnp.array([3.0, 0.5, 0.25, 0.25], dtype=jnp.float32)[:, None]  # sums to 4

    # masked_weights = [3.0,0.5,0.0,0.0], total = 3.5
    # 1) weighted median = 1.0   (cum=[3.0,3.5], target=1.75 → x_med=1)
    # 2) abs_devs = [0,1,99,999]
    #    masked abs_devs & weights → [(0,3.0),(1,0.5)]
    #    target = 0.5 * 3.5 = 1.75
    #    cum_w = [3.0,3.5] → first ≥1.75 at abs_dev=0 → dev_thresh=0
    # 3) bounds = [1.0 ± 0] = [1.0,1.0]
    # 4) clipped x = [1,1,1,1]
    expected_raw = jnp.ones_like(x)

    # raw-x path
    y1 = clipping(
        x,
        mask=mask,
        reweighting_factor=weights,
        data_is_reweighted=False,
    )
    assert_shape_and_dtype(y1, x)
    assert_allclose(y1, expected_raw, atol=1e-6)

    # pre-weighted path
    wx = x * weights
    y2 = clipping(
        wx,
        mask=mask,
        reweighting_factor=weights,
        data_is_reweighted=True,
    )
    expected_wx = expected_raw * weights
    assert_shape_and_dtype(y2, wx)
    assert_allclose(y2, expected_wx, atol=1e-6)


def test_no_clipping():
    clipping = NoneClipping()
    x = jnp.array([-1, 0, 1], dtype=jnp.float32)[:, None]
    y = clipping(x)
    assert_shape_and_dtype(y, x)
    assert_allclose(y, x)


def test_iterative_mean_masking_no_deviation():
    data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    masker = IterativeMeanMasking(max_deviation=0.0)
    mask = masker(data)
    expected = jnp.ones_like(data, dtype=bool)
    assert_allclose(mask, expected)


def test_iterative_mean_masking_with_outlier():
    data = jnp.array([1.0, 2.0, 3.0, 1000.0, 2.5, 500])
    masker = IterativeMeanMasking(max_deviation=1.0, iterations=2)
    mask = masker(data)
    expected = jnp.array([True, True, True, False, True, False])
    assert_allclose(mask, expected)
