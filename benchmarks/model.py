"""End-to-end CPU timings, including matched serial/batched token workloads."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from bench import measured, source_hash
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptzip import ArithmeticCoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--skip-serial-batch", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = os.environ["GPTZIP_QWEN_PATH"]
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.float32
        )
        .cpu()
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    start = time.perf_counter()
    codec = ArithmeticCoder(model, tokenizer, precision=args.precision)
    preparation_seconds = time.perf_counter() - start
    fixture = json.loads((ROOT / "tests/fixtures/qwen3_cpu.json").read_text())
    results = {
        "source_sha256": source_hash(),
        "tokens": args.tokens,
        "repeats": args.repeats,
        "precision": args.precision,
        "threads": 4,
        "preparation_seconds": preparation_seconds,
        "cases": {},
    }
    for width in [1, 16]:
        sequences = [row["token_ids"][: args.tokens] for row in fixture["rows"][:width]]
        modes = (
            ["serial"]
            if width == 1
            else (["batch"] if args.skip_serial_batch else ["serial", "batch"])
        )
        for mode in modes:
            if mode == "serial":

                def encode(sequences=sequences):
                    return tuple(codec.encode_tokens(tokens) for tokens in sequences)

                codes = encode()

                def decode(codes=codes, sequences=sequences):
                    decoded = [codec.decode_tokens(code) for code in codes]
                    assert decoded == sequences
                    return decoded
            else:

                def encode(sequences=sequences):
                    return codec.encode_batch(sequences)

                codes = encode()

                def decode(codes=codes, sequences=sequences):
                    decoded = codec.decode_batch(codes)
                    assert decoded == sequences
                    return decoded

            for operation, fn in [("encode", encode), ("decode", decode)]:
                key = f"{mode}_{operation}_b{width}"
                results["cases"][key] = measured(fn, args.repeats)
                print(key, results["cases"][key]["median_seconds"], flush=True)
    output = ROOT / "benchmarks/results" / f"{args.round}.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
