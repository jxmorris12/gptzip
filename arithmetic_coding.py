# jxm 6/24/24
from collections import OrderedDict

probabilities = {
    ' ': 0.18,
    'A': 0.065, 'B': 0.012, 'C': 0.022, 'D': 0.032, 'E': 0.102, 'F': 0.021, 'G': 0.017,
    'H': 0.053, 'I': 0.057, 'J': 0.001, 'K': 0.006, 'L': 0.033, 'M': 0.020, 'N': 0.057,
    'O': 0.063, 'P': 0.015, 'Q': 0.001, 'R': 0.050, 'S': 0.054, 'T': 0.075, 'U': 0.023,
    'V': 0.008, 'W': 0.018, 'X': 0.001, 'Y': 0.016, 'Z': 0.001
}
probabilities = OrderedDict(probabilities)
# probabilities = OrderedDict(sorted(probabilities, key=lambda t: t[1]))

def arithmetic_encode(text, probabilities):
    low = 0.0
    high = 1.0
    for char in text:
        current_range = high - low
        high = low + current_range * (sum(probabilities[c] for c in probabilities if c <= char))
        low = low + current_range * (sum(probabilities[c] for c in probabilities if c < char))
    return (low + high) / 2

def arithmetic_decode(encoded, probabilities, length):
    low = 0.0
    high = 1.0
    result = ""
    for _ in range(length):
        current_range = high - low
        cumulative_probability = 0.0
        for char, prob in probabilities.items():
            cumulative_probability += prob
            high_char = low + current_range * cumulative_probability
            low_char = high_char - current_range * prob
            if low_char <= encoded < high_char:
                result += char
                low = low_char
                high = high_char
                break
        print(f"Decoding: char={char}, low={low}, high={high}) ")#, mid={mid}")  # Debug statement
    return result

# Example usage
text = "HELLO WORLD"
encoded_value = arithmetic_encode(text.upper(), probabilities)
decoded_text = arithmetic_decode(encoded_value, probabilities, len(text))
print(f"Original: {text}")
print(f"Encoded: {encoded_value}")
print(f"Decoded: {decoded_text}")
