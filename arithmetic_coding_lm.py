# jxm 6/27/24

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = "gpt2"
lm = AutoModelForCausalLM.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained(model)


def get_probs(lm, char_ids):
    # call lm
    with torch.no_grad():
        lm_input = torch.tensor(char_ids)[None]
        lm_output = lm(lm_input)
        logits = lm_output.logits[0]
        # TODO: consider using log here to improve precision?
    return logits.softmax(-1).cpu()

def arithmetic_encode(text, lm, tokenizer):
    low = 0.0
    high = 1.0
    char_ids = (
        [tokenizer.bos_token_id] 
        + tokenizer(text, add_special_tokens=False).input_ids 
        + [tokenizer.eos_token_id]
    )
    all_probs = get_probs(lm, char_ids)
    for j, char_id in enumerate(char_ids[1:]):
        current_range = high - low
        prob_cdf = all_probs[j-1, :].cumsum(-1)
        high = low + current_range * prob_cdf[char_id]
        low = low + current_range * prob_cdf[char_id-1]
    return (low + high) / 2


def arithmetic_decode(encoded, lm, tokenizer, length):
    # TODO: implement finite-precision arithmetic
    # TODO: Translate `encoded` into 8-bit ascii

    low = 0.0
    high = 1.0
    result_token_ids = [tokenizer.bos_token_id]
    for _ in range(length):
        current_range = high - low
        cumulative_probability = 0.0
        probabilities = get_probs(lm, result_token_ids)

        for char_id in range(len(probabilities)):
            prob = probabilities[char_id].item()
            cumulative_probability += prob
            high_char = low + current_range * cumulative_probability
            low_char = high_char - current_range * prob
            if low_char <= encoded < high_char:
                result_token_ids.append(char_id)
                low = low_char
                high = high_char
                break
        print(f"Decoding: char={char_id}, low={low}, high={high}) ")  #, mid={mid}")  # Debug statement
    return tokenizer.decode(result_token_ids)


# Example usage
text = "HELLO WORLD"
encoded_value = arithmetic_encode(text.upper(), lm, tokenizer)
# TODO: get rid of length by using <eos>
decoded_text = arithmetic_decode(encoded_value, lm, tokenizer, len(text))
print(f"Original: {text}")
print(f"Encoded: {encoded_value}")
print(f"Decoded: {decoded_text}")
