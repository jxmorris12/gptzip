"""Offline matched benchmarks; JSON records include source fingerprints.

The CLI is deliberately stdlib argparse: this standalone package does not
require Engrams training or service configuration to benchmark a local codec.
"""

import argparse
import cProfile
import hashlib
import io
import json
import os
import platform
import pstats
import random
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptzip import ArithmeticCoder
from gptzip._range import Decoder, Encoder

ROOT = Path(__file__).resolve().parents[1]


def measured(fn, repeats):
    fn()
    times = []
    digest = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
        digest = hashlib.sha256(repr(result).encode()).hexdigest()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    profile = cProfile.Profile()
    profile.runcall(fn)
    output = io.StringIO()
    pstats.Stats(profile, stream=output).sort_stats("cumtime").print_stats(20)
    return {
        "seconds": times,
        "median_seconds": statistics.median(times),
        "python_peak_bytes": peak,
        "digest": digest,
        "profile": output.getvalue(),
    }


def range_workload(vocab, streams, length):
    rng = np.random.default_rng(842)
    pdf = rng.uniform(0.5, 1.5, vocab).astype(np.float32)
    pdf /= pdf.sum()
    pdf *= 0.2
    pdf[0] += 0.8
    tokens = [
        random.Random(i).choices(range(vocab), weights=pdf, k=length)
        for i in range(streams)
    ]

    def encode():
        codes = []
        for sequence in tokens:
            enc = Encoder()
            for token in sequence:
                enc.encode_pdf(pdf, token)
            codes.append(enc.finish())
        return codes

    codes = encode()

    def decode():
        all_tokens = []
        for data, padding in codes:
            dec = Decoder(data, padding)
            decoded = []
            for _ in range(length):
                decoded.append(int(dec.decode_pdf(pdf)))
            all_tokens.append(decoded)
        assert all_tokens == tokens
        return all_tokens

    return encode, decode


def source_hash():
    h = hashlib.sha256()
    for p in sorted((ROOT / "gptzip").rglob("*")):
        if p.suffix in {".py", ".c", ".h"}:
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--qwen", action="store_true")
    p.add_argument("--tokens", type=int, default=32)
    args = p.parse_args()
    results = {
        "round": args.round,
        "source_sha256": source_hash(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cases": {},
    }
    if args.qwen:
        path = os.environ["GPTZIP_QWEN_PATH"]
        torch.set_num_threads(4)
        lm = (
            AutoModelForCausalLM.from_pretrained(
                path, local_files_only=True, dtype=torch.float32
            )
            .cpu()
            .eval()
        )
        codec = ArithmeticCoder(
            lm, AutoTokenizer.from_pretrained(path, local_files_only=True)
        )
        fixture = json.loads((ROOT / "tests/fixtures/qwen3_cpu.json").read_text())
        for width in [1, 16]:
            sequences = [
                row["token_ids"][: args.tokens] for row in fixture["rows"][:width]
            ]
            code = codec.encode_batch(sequences)
            results["cases"][f"qwen_encode_b{width}"] = measured(
                lambda sequences=sequences: codec.encode_batch(sequences), args.repeats
            )

            def decode_qwen(code=code, sequences=sequences):
                decoded = codec.decode_batch(code)
                assert decoded == sequences
                return decoded

            results["cases"][f"qwen_decode_b{width}"] = measured(
                decode_qwen, args.repeats
            )
            print(f"completed Qwen width {width}", flush=True)
    else:
        for vocab, width, length in [
            (256, 1, 3525),
            (151936, 1, 128),
            (151936, 16, 128),
        ]:
            encode, decode = range_workload(vocab, width, length)
            for name, fn in [("encode", encode), ("decode", decode)]:
                key = f"{name}_v{vocab}_b{width}_t{length}"
                results["cases"][key] = measured(fn, args.repeats)
                print(key, results["cases"][key]["median_seconds"], flush=True)
    out = ROOT / "benchmarks/results" / f"{args.round}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
