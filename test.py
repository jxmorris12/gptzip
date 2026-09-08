"""Round-trip text with a local or already cached CPU language model."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptzip import ArithmeticCoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", help="Text to encode and decode")
    parser.add_argument(
        "--model", default="gpt2", help="Local model path or cached model ID"
    )
    parser.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, dtype=torch.float32
        )
        .cpu()
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    coder = ArithmeticCoder(model, tokenizer, precision=args.precision)
    code = coder.encode(args.message)
    decoded = coder.decode(code)
    assert decoded == args.message, "The tokenizer/model did not preserve this text"
    print(f"Encoded {code.token_count} tokens into {code.bits} payload bits")
    print(decoded)


if __name__ == "__main__":
    main()
