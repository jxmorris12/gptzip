def bits_to_bytes(bits: str) -> tuple[bytes, int]:
    """Returns the bytes representation of bitstream and number of padded bits."""
    # Pad the string with zeros if the length is not a multiple of 8.
    padded_bits = bits.zfill((len(bits) + 7) // 8 * 8)
    num_padded_bits = len(padded_bits) - len(bits)

    # Split the string into 8-bit chunks.
    chunks = [padded_bits[i : i + 8] for i in range(0, len(padded_bits), 8)]

    # Convert each chunk to an integer and then to a byte.
    bytes_data = bytes([int(chunk, base=2) for chunk in chunks])

    return bytes_data, num_padded_bits


def bytes_to_bits(data: bytes, num_padded_bits: int = 0) -> str:
    """Returns the bitstream of bytes data accounting for padded bits."""
    return "".join([bin(byte)[2:].zfill(8) for byte in data])[num_padded_bits:]
