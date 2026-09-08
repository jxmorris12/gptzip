"""The integer-cdf-2^30-min2-v1 profile from the CPU pilot.

Probabilities are float32 softmax outputs; the cumulative sum is float64 in
vocabulary order. Each vocabulary entry receives two counts; the remaining
counts are assigned by rounding down the normalized cumulative distribution.
Changing this operation order changes the format, even with identical logits.
"""

import numpy as np

TOTAL = 1 << 30
PROFILE = "integer-cdf-2^30-min2-v1"


def normalize(pdf: np.ndarray) -> np.ndarray:
    if pdf.ndim != 1 or pdf.dtype != np.float32:
        raise ValueError("Expected a one-dimensional float32 probability array")
    if not pdf.size or pdf.size * 2 >= TOTAL:
        raise ValueError("Unsupported vocabulary size")
    if not np.isfinite(pdf).all() or (pdf < 0).any() or not pdf.any():
        raise ValueError("Probabilities must be finite, nonnegative, and nonzero")
    cdf = np.empty(pdf.size + 1, dtype=np.float64)
    cdf[0] = 0
    np.cumsum(pdf, dtype=np.float64, out=cdf[1:])
    cdf /= cdf[-1]
    bounds = np.floor(cdf * (TOTAL - 2 * pdf.size)).astype(np.int64)
    bounds += 2 * np.arange(pdf.size + 1)
    counts = np.diff(bounds)
    return counts.astype(np.float64) / TOTAL
