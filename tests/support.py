"""Deterministic context-sensitive model for codec tests; no downloads or APIs."""

from types import SimpleNamespace

import numpy as np
import torch


class TinyCache:
    def batch_select_indices(self, indices):
        pass


class TinyLM(torch.nn.Module):
    def __init__(self, vocab=256):
        super().__init__()
        rng = np.random.default_rng(491)
        self.table = torch.nn.Parameter(
            torch.from_numpy(rng.normal(size=(vocab, vocab)).astype(np.float32)),
            requires_grad=False,
        )
        self.config = SimpleNamespace(bos_token_id=0, vocab_size=vocab)
        self.calls = 0
        self.shapes = []
        self.eval()

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        self.calls += 1
        self.shapes.append(tuple(input_ids.shape))
        return SimpleNamespace(
            logits=self.table[input_ids], past_key_values=TinyCache()
        )


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, tokens, skip_special_tokens=True):
        return bytes(tokens).decode("utf-8")
