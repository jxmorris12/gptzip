"""Validate full saved pilot tokens in singleton and dynamically compacted batches."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from bench import source_hash
from transformers import AutoModelForCausalLM

from gptzip import ArithmeticCoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "int8"], default="int8")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = (
        AutoModelForCausalLM.from_pretrained(
            os.environ["GPTZIP_QWEN_PATH"], local_files_only=True, dtype=torch.float32
        )
        .cpu()
        .eval()
    )
    codec = ArithmeticCoder(model, precision=args.precision)
    forward_calls = 0

    def progress(module, inputs):
        nonlocal forward_calls
        forward_calls += 1
        if forward_calls % 512 == 0:
            print("model forwards", forward_calls, flush=True)

    codec.lm.register_forward_pre_hook(progress)
    fixture = json.loads((ROOT / "tests/fixtures/qwen3_cpu.json").read_text())
    sequences = [row["token_ids"] for row in fixture["rows"]]
    results = {
        "precision": args.precision,
        "threads": 4,
        "source_sha256": source_hash(),
        "cases": {},
    }
    for name in ["batch16", "long_singleton"]:
        start = time.perf_counter()
        code = (
            codec.encode_batch(sequences)
            if name == "batch16"
            else codec.encode_tokens(sequences[12])
        )
        encode_seconds = time.perf_counter() - start
        print(name, "encoded", encode_seconds, flush=True)
        start = time.perf_counter()
        decoded = (
            codec.decode_batch(code) if name == "batch16" else codec.decode_tokens(code)
        )
        decode_seconds = time.perf_counter() - start
        assert decoded == (sequences if name == "batch16" else sequences[12])
        results["cases"][name] = {
            "encode_seconds": encode_seconds,
            "decode_seconds": decode_seconds,
            "tokens": sum(map(len, sequences))
            if name == "batch16"
            else len(sequences[12]),
            "code_bits": [c.bits for c in code.codes]
            if name == "batch16"
            else code.bits,
            "exact_round_trip": True,
            "token_sha256": hashlib.sha256(repr(decoded).encode()).hexdigest(),
        }
        output = ROOT / "benchmarks/results" / f"final-full-{args.precision}.json"
        output.write_text(json.dumps(results, indent=2) + "\n")
        print(name, "decoded", decode_seconds, flush=True)


if __name__ == "__main__":
    main()
