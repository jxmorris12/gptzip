import numpy as np
import pytest

from gptzip._range import Decoder, Encoder
from gptzip.probability import TOTAL, build_cdf


def test_terminated_encoder_rejects_reuse():
    cdf = build_cdf(np.array([0.5, 0.5], np.float32))
    encoder = Encoder()
    encoder.encode(cdf, 1)
    encoder.finish()
    with pytest.raises(ValueError):
        encoder.encode(cdf, 0)
    with pytest.raises(ValueError):
        encoder.finish()


@pytest.mark.parametrize(
    "cdf,symbol",
    [
        (np.array([0, TOTAL], np.float64), 0),
        (np.array([0, 0, TOTAL], np.uint32), 0),
        (np.array([0, TOTAL], np.uint32), 1),
        (np.array([0, TOTAL - 1], np.uint32), 0),
    ],
)
def test_invalid_native_intervals(cdf, symbol):
    encoder = Encoder()
    with pytest.raises(ValueError):
        encoder.encode(cdf, symbol)


@pytest.mark.parametrize("data,padding", [(b"", 0), (b"a", -1), (b"a", 8)])
def test_invalid_packed_input(data, padding):
    with pytest.raises(ValueError):
        Decoder(data, padding)


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 6, 7, 8, 1023, 1024, 8193])
def test_packed_padding_and_buffer_growth(length):
    cdf = build_cdf(np.array([0.5, 0.5], np.float32))
    tokens = [i % 2 for i in range(length)]
    encoder = Encoder()
    for token in tokens:
        encoder.encode(cdf, token)
    data, padding = encoder.finish()
    assert 8 * len(data) - padding == length + 1
    assert data[0] >> (8 - padding) == 0
    decoder = Decoder(data, padding)
    assert [decoder.decode(cdf) for _ in tokens] == tokens
