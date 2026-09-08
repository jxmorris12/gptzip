"""Frozen Python oracle protects pilot code words across native optimizations."""

import random

import numpy as np
import pytest
from reference_bits import bits_to_bytes
from reference_probability import normalize as reference_normalize
from reference_range import Encoder as ReferenceEncoder

from gptzip._range import Decoder, Encoder
from gptzip.probability import build_cdf


@pytest.mark.parametrize("vocab", [2, 256, 151936])
def test_code_words_match_frozen_reference(vocab):
    rng = np.random.default_rng(88)
    pdf = rng.uniform(0, 1, vocab).astype(np.float32)
    pdf[0] = 100 * vocab
    pdf /= pdf.sum()
    tokens = random.Random(987).choices(range(vocab), weights=pdf, k=512)
    expected_bits = []
    reference = ReferenceEncoder(2, 32, expected_bits.append)
    encoder = Encoder()
    for token in tokens:
        reference.encode(reference_normalize(pdf), token)
        encoder.encode(build_cdf(pdf), token)
    reference.terminate()
    data, padding = encoder.finish()
    assert (data, padding) == bits_to_bytes("".join(map(str, expected_bits)))
    decoder = Decoder(data, padding)
    assert [int(decoder.decode(build_cdf(pdf))) for _ in tokens] == tokens


def test_changing_probability_rows_match_reference():
    rng = np.random.default_rng(415)
    rows = rng.lognormal(0, 5, size=(200, 31)).astype(np.float32)
    rows[:, 0] = np.nextafter(np.float32(0), np.float32(1))
    tokens = [i % 31 for i in range(len(rows))]
    expected = []
    reference = ReferenceEncoder(2, 32, expected.append)
    encoder = Encoder()
    for row, token in zip(rows, tokens, strict=True):
        reference.encode(reference_normalize(row), token)
        encoder.encode(build_cdf(row), token)
    reference.terminate()
    data, padding = encoder.finish()
    assert (data, padding) == bits_to_bytes("".join(map(str, expected)))
    decoder = Decoder(data, padding)
    assert [int(decoder.decode(build_cdf(row))) for row in rows] == tokens
