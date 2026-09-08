import numpy as np
import pytest
from reference_probability import normalize as reference_normalize

from gptzip.probability import TOTAL, build_cdf, normalize


@pytest.mark.parametrize("size", [1, 2, 256, 151936])
def test_full_coverage_and_floor(size):
    pdf = np.zeros(size, dtype=np.float32)
    pdf[0] = 1
    normalized = normalize(pdf)
    assert normalized.sum() == 1
    assert normalized.min() >= 2 / TOTAL
    assert (normalized * TOTAL == np.floor(normalized * TOTAL)).all()


@pytest.mark.parametrize(
    "values", [[0, 0], [-1, 2], [float("nan"), 1], [float("inf"), 1]]
)
def test_invalid_probability(values):
    with pytest.raises(ValueError):
        normalize(np.array(values, dtype=np.float32))


def test_cdf_matches_frozen_probability_profile():
    rng = np.random.default_rng(701)
    for size in [2, 256, 151936]:
        pdf = rng.uniform(0, 1, size).astype(np.float32)
        pdf[0] = 0
        expected = reference_normalize(pdf)
        actual = np.diff(build_cdf(pdf)).astype(np.float64) / TOTAL
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "value",
    [np.ones((2, 2), np.float32), np.ones(3, np.float64), np.ones(8, np.float32)[::2]],
)
def test_native_rejects_incompatible_buffers(value):
    with pytest.raises((ValueError, BufferError)):
        build_cdf(value)


def test_native_extreme_distributions_match_reference():
    rng = np.random.default_rng(217)
    for size in [2, 31, 4096, 151936]:
        for scale in [0.1, 10, 40]:
            pdf = np.exp(rng.uniform(-scale, scale, size)).astype(np.float32)
            pdf[0] = np.nextafter(np.float32(0), np.float32(1))
            np.testing.assert_array_equal(normalize(pdf), reference_normalize(pdf))
