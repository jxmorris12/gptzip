"""Cached CPU language-model arithmetic coding, with explicit inference profiles."""

from dataclasses import dataclass

import torch

from ._batch import steps
from ._inference import inference_operation, prepare_model
from ._range import Decoder, Encoder
from .probability import PROFILE


@dataclass(frozen=True)
class Code:
    data: bytes
    padding: int
    token_count: int
    inference: str = "singleton-fp32-v1"

    @property
    def bits(self):
        return 8 * len(self.data) - self.padding


@dataclass(frozen=True)
class BatchCode:
    codes: tuple[Code, ...]
    width: int = 1
    profile: str = PROFILE
    inference: str = "compact-fp32-v1"


class ArithmeticCoder:
    def __init__(self, lm, tokenizer=None, precision="fp32"):
        if precision not in {"fp32", "int8"}:
            raise ValueError("precision must be fp32 or int8")
        self.precision = precision
        profile = (
            precision
            if precision == "fp32"
            else f"int8-{torch.backends.quantized.engine}"
        )
        self.singleton_profile = f"singleton-{profile}-v1"
        self.batch_profile = f"compact-{profile}-v1"
        if next(lm.parameters()).device.type != "cpu":
            raise ValueError("This codec currently supports CPU models only")
        if lm.training:
            raise ValueError("Call model.eval() before constructing the codec")
        self.lm = prepare_model(lm, precision)
        self.tokenizer = tokenizer
        self.bos = lm.config.bos_token_id
        if self.bos is None:
            raise ValueError("The model must supply bos_token_id")

    def _probs(self, token, cache):
        with torch.inference_mode():
            out = self.lm(
                input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True
            )
            pdf = out.logits[0, -1].float().softmax(-1).numpy()
        return pdf, out.past_key_values

    def _probs_batch(self, tokens, cache):
        with torch.inference_mode():
            out = self.lm(
                input_ids=torch.tensor(tokens).reshape(-1, 1),
                past_key_values=cache,
                use_cache=True,
            )
            pdfs = out.logits[:, -1].float().softmax(-1).numpy()
        return pdfs, out.past_key_values

    @inference_operation
    def encode_tokens(self, tokens):
        encoder = Encoder()
        previous, cache = self.bos, None
        for token in tokens:
            pdf, cache = self._probs(previous, cache)
            if not 0 <= token < pdf.size:
                raise ValueError("Token outside model vocabulary")
            encoder.encode_pdf(pdf, token)
            previous = token
        data, padding = encoder.finish()
        return Code(data, padding, len(tokens), self.singleton_profile)

    @inference_operation
    def decode_random(self, data, padding=0, max_tokens=4096):
        if not data or not 0 <= padding < 8 or max_tokens < 1:
            raise ValueError("Invalid bitstream, padding, or token limit")
        decoder = Decoder(data, padding)
        previous, cache = self.bos, None
        tokens = []
        for _ in range(max_tokens + 1):
            pdf, cache = self._probs(previous, cache)
            try:
                token = int(decoder.decode_pdf(pdf))
            except StopIteration:
                return tokens
            if len(tokens) == max_tokens:
                raise ValueError("Token limit reached before bitstream exhaustion")
            tokens.append(token)
            previous = token
        raise AssertionError("unreachable")

    @inference_operation
    def decode_tokens(self, code):
        if code.inference != self.singleton_profile:
            raise ValueError("Code requires its original batch/inference profile")
        if code.token_count < 0 or not code.data or not 0 <= code.padding < 8:
            raise ValueError("Invalid code metadata")
        decoder = Decoder(code.data, code.padding)
        previous, cache = self.bos, None
        tokens = []
        for _ in range(code.token_count):
            pdf, cache = self._probs(previous, cache)
            try:
                token = int(decoder.decode_pdf(pdf))
            except StopIteration as exc:
                raise ValueError("Truncated code") from exc
            tokens.append(token)
            previous = token
        return tokens

    @inference_operation
    def encode_batch(self, sequences):
        """Encode token sequences together; keep the entire returned frame."""
        encoders = [Encoder() for _ in sequences]
        previous = [self.bos] * len(sequences)
        finished = [False] * len(sequences)
        limit = max(map(len, sequences), default=0)
        for step, active, pdfs in steps(self, previous, finished, limit):
            for row, i in enumerate(active):
                if step == len(sequences[i]):
                    finished[i] = True
                    continue
                token = sequences[i][step]
                encoders[i].encode_pdf(pdfs[row], token)
                previous[i] = token
        codes = tuple(
            Code(*encoder.finish(), len(tokens), self.batch_profile)
            for encoder, tokens in zip(encoders, sequences, strict=True)
        )
        return BatchCode(codes, len(codes), PROFILE, self.batch_profile)

    @inference_operation
    def decode_batch(self, batch):
        """Decode a complete frame with the same model and inference profile."""
        if (
            batch.profile != PROFILE
            or batch.width != len(batch.codes)
            or batch.inference != self.batch_profile
        ):
            raise ValueError("Incompatible arithmetic or inference profile")
        for code in batch.codes:
            if code.inference != batch.inference or code.token_count < 0:
                raise ValueError("Invalid batch code metadata")
        decoders = [Decoder(code.data, code.padding) for code in batch.codes]
        previous = [self.bos] * len(decoders)
        finished = [False] * len(decoders)
        sequences = [[] for _ in decoders]
        limit = max((code.token_count for code in batch.codes), default=0)
        for step, active, pdfs in steps(self, previous, finished, limit):
            for row, i in enumerate(active):
                if step == batch.codes[i].token_count:
                    finished[i] = True
                    continue
                try:
                    token = decoders[i].decode_pdf(pdfs[row])
                except StopIteration as exc:
                    raise ValueError("Truncated batch code") from exc
                sequences[i].append(token)
                previous[i] = token
        return sequences

    @inference_operation
    def decode_random_batch(self, streams, max_tokens=4096):
        """Map independent bytes to tokens using reproducible batch compaction."""
        if max_tokens < 1:
            raise ValueError("Invalid token limit")
        decoders = [Decoder(data) for data in streams]
        previous = [self.bos] * len(decoders)
        finished = [False] * len(decoders)
        sequences = [[] for _ in decoders]
        for step, active, pdfs in steps(self, previous, finished, max_tokens + 1):
            for row, i in enumerate(active):
                try:
                    token = decoders[i].decode_pdf(pdfs[row])
                except StopIteration:
                    finished[i] = True
                    continue
                if step == max_tokens:
                    raise ValueError("Token limit reached before bitstream exhaustion")
                sequences[i].append(token)
                previous[i] = token
        return sequences

    def encode(self, text):
        return self.encode_tokens(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, code):
        return self.tokenizer.decode(
            self.decode_tokens(code), skip_special_tokens=False
        )
