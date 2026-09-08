"""Structural performance regressions, independent of host timing noise."""

import tracemalloc

import numpy as np
import torch
from support import TinyLM

from gptzip import ArithmeticCoder


def test_model_only_receives_uncached_tokens():
    torch.set_num_threads(1)
    model = TinyLM()
    codec = ArithmeticCoder(model)
    sequences = [list(range(i + 1)) for i in range(16)]
    encoded = codec.encode_batch(sequences)
    assert model.calls <= sum(map(len, sequences))
    assert all(shape[-1] == 1 for shape in model.shapes)
    before = model.calls
    assert codec.decode_batch(encoded) == sequences
    assert model.calls - before <= sum(map(len, sequences))


def test_encoding_probability_memory_does_not_scale_with_token_count(monkeypatch):
    codec = ArithmeticCoder(TinyLM())
    vocab = 4096
    pdf = np.full(vocab, 1 / vocab, dtype=np.float32)
    # Allocate each probability row as NumPy to expose accidental retention or
    # stacking to tracemalloc (PyTorch allocator memory is outside its scope).
    monkeypatch.setattr(codec, "_probs", lambda token, cache: (pdf.copy(), None))

    def peak(length):
        tracemalloc.start()
        codec.encode_tokens([1] * length)
        _, result = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result

    short = peak(32)
    long = peak(256)
    # The reference writer keeps Python objects per emitted bit (12/token in
    # this uniform 4096-symbol case). Permit O(output bits), while rejecting
    # retention of O(tokens * vocab) probability arrays. Reducing fixed CDF
    # workspace must not make this regression check fail spuriously.
    output_growth_budget = (256 - 32) * 12 * 128
    assert long < short + 4 * vocab * 8 + output_growth_budget
