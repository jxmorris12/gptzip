"""The integer-cdf-2^30-min2-v1 profile from the CPU pilot.

Probabilities are float32 softmax outputs; the cumulative sum is float64 in
vocabulary order. Each vocabulary entry receives two counts; the remaining
counts are assigned by rounding down the normalized cumulative distribution.
Changing this operation order changes the format, even with identical logits.
"""

import numpy as np

from . import _native

TOTAL = 1 << 30
PROFILE = "integer-cdf-2^30-min2-v1"


def build_cdf(pdf: np.ndarray) -> np.ndarray:
    """Convert an aligned contiguous float32 PDF into immutable uint32 bounds."""
    return np.frombuffer(_native.build_cdf(pdf), dtype=np.uint32)


def normalize(pdf: np.ndarray) -> np.ndarray:
    """Expose the quantized PDF for diagnostics; coding uses build_cdf directly."""
    return np.diff(build_cdf(pdf)).astype(np.float64) / TOTAL
