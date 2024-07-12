from typing import Iterator, Tuple

import argparse
import numpy as np
import torch


from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import bits_to_bytes, bytes_to_bits, normalize_pdf_for_arithmetic_coding
from .helpers import Encoder, Decoder

# Base 2 means that the coder writes bits.
ARITHMETIC_CODER_BASE = 2
# Precision 32 implies 32 bit arithmetic.
ARITHMETIC_CODER_PRECISION = 32

class ArithmeticCoder:
    # Helpful links:
    #   > https://github.com/google-deepmind/language_modeling_is_compression
    #   > https://www.cs.cmu.edu/~aarti/Class/10704/Intro_Arith_coding.pdf
    #   > https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
    #   > https://www.cs.ucf.edu/courses/cap5015/Arithmetic%20Coding%20note%202004.pdf

    def __init__(self, lm, tokenizer):
        self.lm = lm
        self.tokenizer = tokenizer
    
    def _next_token_probs(self, input_ids: torch.Tensor, past_key_values: Tuple) -> torch.Tensor:
        print("[_next_token_probs],", input_ids, past_key_values is None)
        if (past_key_values is not None):
            # HuggingFace doesn't want us to provide input ids for anything that's in the kv cache.
            # We have to trim this part.
            kv_cache_seq_length = past_key_values[0][0].shape[2]
            input_ids = input_ids[:, kv_cache_seq_length:]
        assert len(input_ids.shape) == 2, f"can't get probs for input_ids shape {input_ids.shape}"
        attention_mask = torch.ones_like(input_ids)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        with torch.no_grad():
            output = self.lm(
                input_ids=input_ids,
                # attention_mask=attention_mask,
                # position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        probs = output.logits.to(torch.float32).softmax(dim=-1)
        return (probs.cpu().numpy(), output.past_key_values)


    def encode(
        self,
        data: str,
        return_num_padded_bits: bool = False,
    ) -> bytes | tuple[bytes, int]:
        """Compresses the `data` using arithmetic coding and a pretrained model.

        Args:
            data: The data to be compressed.
            return_num_padded_bits: Whether to return the number of zeros added to the
            encoded bitstream in order to make it byte-decodeable (i.e., divisible by
            8). Usually, this is used when the encoded data has to be decoded again.

        Returns:
            The compressed data.
        """

        # Convert the `data` into an array of integers (representing the bytes).
        sequence_array = self.tokenizer(data, return_tensors='pt').input_ids
        sequence_array = torch.cat(
            [
                torch.tensor([self.tokenizer.bos_token_id]),
                sequence_array.flatten(),
            ]
        )
        # print("Tokens:", data, "//", sequence_array)

        log_probs = []
        past_key_values = None
        for subsequence_length in range(len(sequence_array)):
            subsequence_probs, past_key_values = self._next_token_probs(
                input_ids=sequence_array[None, : subsequence_length + 1],
                past_key_values=past_key_values
            )
            log_probs.append(subsequence_probs[0, -1])
            probs = np.vstack(log_probs)

        output = list()
        encoder = Encoder(
            base=ARITHMETIC_CODER_BASE,
            precision=ARITHMETIC_CODER_PRECISION,
            output_fn=output.append,
        )
        for pdf, symbol in zip(probs[:,], sequence_array[1:]):
            # print("Encoding symbol:", symbol.item(), "/", self.tokenizer.decode([symbol.item()]), "pdf argmax:", pdf.argmax(), "/", self.tokenizer.decode([pdf.argmax()]))
            encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol.item())
        encoder.terminate()

        compressed_bits = ''.join(map(str, output))
        compressed_bytes, num_padded_bits = bits_to_bytes(compressed_bits)

        if return_num_padded_bits:
            return compressed_bytes, num_padded_bits
        else:
            return compressed_bytes


    def decode(
            self,
            data: bytes,
            num_padded_bits: int = 0,
        ) -> bytes:
        """Decompresses the `data` using arithmetic coding and a pretrained model.

        See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

        Args:
            data: The data to be decompressed.
            num_padded_bits: The number of zeros added to the encoded bitstream in order
            to make it byte-decodeable (i.e., divisble by 8).

        Returns:
            The decompressed data.
        """
        data_iter = iter(bytes_to_bits(data, num_padded_bits=num_padded_bits))

        # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
        # from the compressed input and returns `None` when the input is exhausted.
        def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
            try:
                return int(next(bit_sequence))
            except StopIteration:
                return None

        decoder = Decoder(
            base=ARITHMETIC_CODER_BASE,
            precision=ARITHMETIC_CODER_PRECISION,
            input_fn=_input_fn,
        )
        # We need a dummy token because the language model right-shifts the sequence
        # by onde when computing the conditional probabilities. Concretely, at every
        # step, we need the `pdf` of the next token given all currently decompressed
        # tokens, but without a dummy token, the last `pdf` would be that of the last
        # already decompressed token. The value of the dummy token is irrelevant.
        sequence_array = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.int32)
        # print("3 >> sequence_array.shape", sequence_array.shape)
        probs, past_key_values = self._next_token_probs(
            input_ids=sequence_array[None], 
            past_key_values=None
        )
        probs = probs[0, 0]

        idx = 0
        while True:
            # print("idx", idx, "probs.shape", probs.shape, "/ argmax", probs.argmax().item(), "sequence_arr", sequence_array)
            try:
                token = decoder.decode(
                    normalize_pdf_for_arithmetic_coding(probs)
                )
            except StopIteration:
                break
            # print("\t token:", token)
            sequence_array = torch.tensor(
                np.append(sequence_array, token)
                , dtype=torch.int32
            )
            probs, past_key_values = self._next_token_probs(sequence_array[None], past_key_values=past_key_values)
            probs = probs[0, -1]
            idx += 1

        # Remove the dummy token and convert to bytes.
        print(f"Decoded {len(sequence_array)} tokens:", sequence_array)
        return self.tokenizer.decode(sequence_array)

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
