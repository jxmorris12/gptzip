import gptzip
import transformers

def to_binary(data: bytes) -> str:
    return ''.join(format(byte, '08b') for byte in data)

def test_encode():
    model = "gpt2"
    lm = transformers.AutoModelForCausalLM.from_pretrained(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    string = "Sailing on the seven seas"
    coder = gptzip.ArithmeticCoder(lm=lm, tokenizer=tokenizer)
    code, num_padded_bits = coder.encode(
        string, 
        return_num_padded_bits=True, 
    )
    assert len(code) == 5
    assert num_padded_bits == 0


def test_encode_and_decode_is_lossless():
    model = "gpt2"
    lm = transformers.AutoModelForCausalLM.from_pretrained(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    string = "How much would could a woodchuck chuck?"
    coder = gptzip.ArithmeticCoder(lm=lm, tokenizer=tokenizer)
    code, num_padded_bits = coder.encode(
        string, 
        return_num_padded_bits=True, 
    )
    print(f"Code: {to_binary(code)} ({len(code)} bytes)")
    decoded_string = coder.decode(code, num_padded_bits=num_padded_bits)
    assert decoded_string == string
