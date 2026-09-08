import random

import pytest
import torch
from support import ByteTokenizer, TinyLM

from gptzip import ArithmeticCoder


@pytest.fixture
def codec():
    torch.set_num_threads(1)
    return ArithmeticCoder(TinyLM(), ByteTokenizer())


@pytest.mark.parametrize("length", [0, 1, 64, 256, 3525])
def test_singleton_token_round_trip(codec, length):
    tokens = [i % 256 for i in range(length)]
    code = codec.encode_tokens(tokens)
    assert codec.decode_tokens(code) == tokens
    assert code.bits == len(code.data) * 8 - code.padding


def test_batch16_mixed_lengths(codec):
    lengths = [0, 1, 2, 17, 64, 128, 198, 266, 389, 550, 155, 561, 194, 290, 3525, 361]
    sequences = [
        random.Random(i).choices(range(256), k=n) for i, n in enumerate(lengths)
    ]
    assert codec.decode_batch(codec.encode_batch(sequences)) == sequences


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Arithmetic coding on a CPU: hello, world!",
        "你好 Καλημέρα 👩🏽‍💻",
        "x\x00y\n\t  z",
    ],
)
def test_text_round_trip(codec, text):
    assert codec.decode(codec.encode(text)) == text


def test_sixteen_random_1024_bit_streams(codec):
    for seed in range(16):
        data = random.Random(seed).randbytes(128)
        tokens = codec.decode_random(data)
        assert 50 < len(tokens) < 400
        assert codec.decode_tokens(codec.encode_tokens(tokens)) == tokens


def test_limit_and_invalid_input(codec):
    with pytest.raises(ValueError, match="Token limit"):
        codec.decode_random(random.Random(0).randbytes(128), max_tokens=1)
    with pytest.raises(ValueError):
        codec.decode_random(b"")
    with pytest.raises(ValueError):
        codec.decode_random(b"a", padding=8)
    with pytest.raises(ValueError):
        codec.encode_tokens([256])


def test_high_probability_tail_uses_declared_token_count(codec):
    # EOF-only termination cannot distinguish lengths when a likely token emits
    # no bits. A framed stream must retain the token count, including zero.
    pdf = torch.full((256,), 0.2 / 255)
    pdf[0] = 0.8
    with torch.no_grad():
        codec.lm.table.copy_(pdf.log().repeat(256, 1))
    for length in [0, 1, 16, 256, 3525]:
        tokens = [0] * length
        assert codec.decode_tokens(codec.encode_tokens(tokens)) == tokens
