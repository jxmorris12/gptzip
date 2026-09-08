import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptzip import ArithmeticCoder

FIXTURE = json.loads((Path(__file__).parent / "fixtures/qwen3_cpu.json").read_text())


@pytest.fixture(scope="module")
def qwen():
    path = os.environ.get("GPTZIP_QWEN_PATH")
    if path is None:
        pytest.skip("Set GPTZIP_QWEN_PATH to a local Qwen3-0.6B snapshot")
    torch.set_num_threads(4)
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.float32
        )
        .cpu()
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return ArithmeticCoder(model, tokenizer)


@pytest.mark.integration
@pytest.mark.parametrize("index", [0, 7])
def test_qwen_saved_random_stream(qwen, index):
    row = FIXTURE["rows"][index]
    tokens = qwen.decode_random(bytes.fromhex(row["input_hex"]))
    assert tokens == row["token_ids"]
    code = qwen.encode_tokens(tokens)
    assert code.data.hex() == row["code_hex"]
    assert code.padding == row["padding_bits"]


@pytest.mark.integration
def test_qwen_batch16(qwen):
    sequences = [row["token_ids"][:24] for row in FIXTURE["rows"]]
    encoded = qwen.encode_batch(sequences)
    assert qwen.decode_batch(encoded) == sequences


@pytest.mark.integration
def test_qwen_ragged_batch(qwen):
    sequences = [row["token_ids"][: i * 2] for i, row in enumerate(FIXTURE["rows"])]
    encoded = qwen.encode_batch(sequences)
    assert qwen.decode_batch(encoded) == sequences


@pytest.mark.integration
def test_qwen_random_batch(qwen):
    streams = [bytes.fromhex(row["input_hex"])[:8] for row in FIXTURE["rows"]]
    sequences = qwen.decode_random_batch(streams, max_tokens=512)
    assert qwen.decode_batch(qwen.encode_batch(sequences)) == sequences


@pytest.mark.integration
def test_qwen_int8_round_trip(qwen):
    fast = ArithmeticCoder(qwen.lm, qwen.tokenizer, precision="int8")
    sequences = [row["token_ids"][: i * 2] for i, row in enumerate(FIXTURE["rows"])]
    assert fast.decode_batch(fast.encode_batch(sequences)) == sequences
    assert fast.decode_tokens(fast.encode_tokens(sequences[-1])) == sequences[-1]


@pytest.mark.integration
def test_qwen_literal_special_token_text(qwen):
    text = "你好 🌍<|endoftext|>"
    assert qwen.decode(qwen.encode(text)) == text
