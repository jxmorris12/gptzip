"""Packed binary 32-bit range coding under the pilot's integer CDF profile."""

from . import _native


class Encoder:
    def __init__(self):
        self._state = _native.encoder_new()

    def encode(self, cdf, symbol):
        _native.encode(self._state, cdf, symbol)

    def encode_pdf(self, pdf, symbol):
        _native.encode_pdf(self._state, pdf, symbol)

    def finish(self):
        """Return (bytes, leading_padding_bits), including arithmetic termination."""
        return _native.terminate(self._state)


class Decoder:
    def __init__(self, data, padding=0):
        self._state = _native.decoder_new(data, padding)

    def decode(self, cdf):
        return _native.decode(self._state, cdf)

    def decode_pdf(self, pdf):
        return _native.decode_pdf(self._state, pdf)
