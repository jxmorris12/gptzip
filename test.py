from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from gptzip import ArithmeticCoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("message", type=str, help="The message to print")
    args = parser.parse_args()

    model = "gpt2"
    lm = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    string = args.message
    coder = ArithmeticCoder(lm=lm, tokenizer=tokenizer)
    print(f"[0] Encoding... `{string}`")
    code, num_padded_bits = coder.encode(
        string, 
        return_num_padded_bits=True, 
    )
    print(f"[1] Code... `{code}` ({len(code)} bytes, num_padded_bits={num_padded_bits})")
    print("\n" * 5)
    decoded_string = coder.decode(code, num_padded_bits=num_padded_bits)
    print(f"[2] Decoded: {decoded_string}")

