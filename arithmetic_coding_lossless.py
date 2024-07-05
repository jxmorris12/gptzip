# jxm 7/5/24

from typing import Tuple
from collections import OrderedDict

from helpers import float2bin, bin2float
from helpers_cpp import decode_prob_cpp, encode_prob_cpp

probabilities = {
    ' ': 0.18,
    'A': 0.065, 'B': 0.012, 'C': 0.022, 'D': 0.032, 'E': 0.102, 'F': 0.021, 'G': 0.017,
    'H': 0.053, 'I': 0.057, 'J': 0.001, 'K': 0.006, 'L': 0.033, 'M': 0.020, 'N': 0.057,
    'O': 0.063, 'P': 0.015, 'Q': 0.001, 'R': 0.050, 'S': 0.054, 'T': 0.075, 'U': 0.023,
    'V': 0.008, 'W': 0.018, 'X': 0.001, 'Y': 0.016, 'Z': 0.001
}
probabilities = OrderedDict(probabilities)


class ArithmeticCoder:
    # Helpful links:
    #   > https://www.cs.cmu.edu/~aarti/Class/10704/Intro_Arith_coding.pdf
    #   > https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
    #   > https://www.cs.ucf.edu/courses/cap5015/Arithmetic%20Coding%20note%202004.pdf
    # def __init__(self):
    
    def _get_prob(self, char: str) -> Tuple[float, float]:
        """Returns the probability of a given character."""
        L = 0
        R = 0
        for pchar, prob in probabilities.items():
            L += prob
            if pchar == char:
                break
            R += prob
        return L, R

    def _bit_plus_pending(self, bit: int) -> str:
        neg_bit = 1 - bit
        out = str(bit) + (str(neg_bit) * self._pending_bits)
        self._pending_bits = 0
        return out

    def encode(self, text: str) -> str:
        low = 0.0
        high = 1.0
        out = ""
        for j, char in enumerate(text):
            current_range = high - low + 1 # TODO: why a +1?
            lower_prob, upper_prob = self._get_prob(char)
            high = low + current_range * upper_prob
            low = low + current_range * lower_prob
            output, low, high = encode_prob_cpp(
                low, high,
                lower_prob, upper_prob
            )

        return out


    def decode(self, encoded: str, length: int) -> str:
        # TODO: implement finite-precision arithmetic
        # TODO: Translate `encoded` into 8-bit ascii

        low = 0.0
        # high = 1.0
        # for _ in range(length):
        #     current_range = high - low
        #     cumulative_probability = 0.0

        #     for char_id in range(len(probabilities)):
        #         prob = probabilities[char_id].item()
        #         cumulative_probability += prob
        #         high_char = low + current_range * cumulative_probability
        #         low_char = high_char - current_range * prob
        #         if low_char <= encoded < high_char:
        #             result_token_ids.append(char_id)
        #             low = low_char
        #             high = high_char
        #             break
        #     print(f"Decoding: char={char_id}, low={low}, high={high}) ")  #, mid={mid}")  # Debug statement
        # return tokenizer.decode(result_token_ids)


# Example usage
text = "HELLO WORLD"
coder = ArithmeticCoder()
encoded_value = coder.encode(text)
# TODO: get rid of length by using <eos>
decoded_text = coder.decode(encoded_value)
print(f"Original: {text}")
print(f"Encoded: {encoded_value}")
print(f"Decoded: {decoded_text}")
