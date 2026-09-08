import random
from dataclasses import replace

import pytest
import torch
from support import TinyLM

from gptzip import ArithmeticCoder


def test_compaction_shapes_and_forward_count():
    torch.set_num_threads(1)
    model = TinyLM()
    codec = ArithmeticCoder(model)
    sequences = [[], [3], [1, 7], [9] * 5]
    code = codec.encode_batch(sequences)
    shapes = model.shapes.copy()
    assert shapes == [(4, 1), (3, 1), (2, 1), (1, 1), (1, 1)]
    model.shapes.clear()
    assert codec.decode_batch(code) == sequences
    assert model.shapes == shapes
    with pytest.raises(ValueError, match="original batch"):
        codec.decode_tokens(code.codes[0])
    with pytest.raises(ValueError):
        codec.decode_batch(replace(code, width=1))


def test_empty_batches():
    codec = ArithmeticCoder(TinyLM())
    for sequences in [[], [[]], [[], []]]:
        assert codec.decode_batch(codec.encode_batch(sequences)) == sequences
    assert codec.decode_random_batch([]) == []
    assert codec.lm.calls == 0


def test_random_batch_round_trip():
    torch.set_num_threads(1)
    codec = ArithmeticCoder(TinyLM())
    streams = [random.Random(i).randbytes(128) for i in range(16)]
    sequences = codec.decode_random_batch(streams)
    shapes = codec.lm.shapes.copy()
    codec.lm.shapes.clear()
    frame = codec.encode_batch(sequences)
    # Random decoding performs one final model forward solely to discover EOF.
    assert codec.lm.shapes == shapes[:-1]
    assert codec.decode_batch(frame) == sequences
    with pytest.raises(ValueError, match="Token limit"):
        codec.decode_random_batch(streams, max_tokens=1)
