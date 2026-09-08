import numpy as np
import pytest

from gptzip._range import Decoder, Encoder
from gptzip.probability import build_cdf


@pytest.mark.parametrize("vocab", [2, 31, 256, 151936])
def test_fused_pdf_matches_cdf_path(vocab):
    rng = np.random.default_rng(678)
    reference, fused = Encoder(), Encoder()
    rows, tokens = [], []
    for i in range(64):
        row = rng.lognormal(0, 7, vocab).astype(np.float32)
        row[0] = 0
        token = (i * 327 + 1) % vocab
        reference.encode(build_cdf(row), token)
        fused.encode_pdf(row, token)
        rows.append(row)
        tokens.append(token)
    expected = reference.finish()
    assert fused.finish() == expected
    decoder = Decoder(*expected)
    assert [decoder.decode_pdf(row) for row in rows] == tokens


def test_decoder_workspace_grows_with_vocabulary():
    rows = [np.full(n, 1 / n, np.float32) for n in [2, 512, 31, 8192, 2]]
    enc = Encoder()
    for row in rows:
        enc.encode_pdf(row, len(row) - 1)
    dec = Decoder(*enc.finish())
    assert [dec.decode_pdf(row) for row in rows] == [len(row) - 1 for row in rows]


@pytest.mark.parametrize(
    "pdf",
    [
        np.array([np.nan, 1], np.float32),
        np.zeros(2, np.float32),
        np.array([1, -1], np.float32),
        np.ones(2, np.float64),
        np.ones((1, 2), np.float32),
        np.array([], np.float32),
    ],
)
def test_fused_rejects_invalid_pdf(pdf):
    with pytest.raises(ValueError):
        Encoder().encode_pdf(pdf, 0)
    with pytest.raises(ValueError):
        Decoder(b"aa").decode_pdf(pdf)
