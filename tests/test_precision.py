from dataclasses import replace

import pytest
import torch
from support import TinyLM

from gptzip import ArithmeticCoder


def test_explicit_precision_profiles():
    model = TinyLM()
    full = ArithmeticCoder(model)
    fast = ArithmeticCoder(model, precision="int8")
    tokens = [3, 7, 9, 22]
    code = fast.encode_tokens(tokens)
    assert fast.decode_tokens(code) == tokens
    with pytest.raises(ValueError, match="profile"):
        full.decode_tokens(code)
    batch = fast.encode_batch([tokens, []])
    assert fast.decode_batch(batch) == [tokens, []]
    with pytest.raises(ValueError, match="profile"):
        full.decode_batch(batch)
    with pytest.raises(ValueError):
        fast.decode_batch(replace(batch, inference="compact-fp32-v1"))
    with pytest.raises(ValueError):
        ArithmeticCoder(model, precision="fp16")
    assert model.table.dtype == torch.float32


def test_operation_context_and_float32_softmax(monkeypatch):
    codec = ArithmeticCoder(TinyLM(), precision="int8")
    model = codec.lm
    original = model.forward
    calls = []

    def forward(*args, **kwargs):
        calls.append(torch.is_autocast_enabled("cpu"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    codec.encode_tokens([1, 2, 3])
    codec.encode_batch([[1, 2], [3, 4]])
    assert calls and not any(calls)
    assert not torch.is_autocast_enabled("cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pdf, _ = codec._probs(0, None)
        assert str(pdf.dtype) == "float32"


def test_prepared_integer_layers_preserve_original_model():
    class LinearLM(TinyLM):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(256, 256)
            self.eval()

        def forward(self, *args, **kwargs):
            out = super().forward(*args, **kwargs)
            out.logits = self.projection(out.logits)
            return out

    torch.set_num_threads(1)
    model = LinearLM()
    original_weights = model.projection.weight.detach().clone()
    codec = ArithmeticCoder(model, precision="int8")
    assert isinstance(codec.lm.projection, torch.ao.nn.quantized.dynamic.Linear)
    assert isinstance(model.projection, torch.nn.Linear)
    assert torch.equal(model.projection.weight, original_weights)
    sequences = [[1, 2, 3, 4], [11, 18], []]
    assert codec.decode_batch(codec.encode_batch(sequences)) == sequences
    assert codec.decode_tokens(codec.encode_tokens(sequences[0])) == sequences[0]
