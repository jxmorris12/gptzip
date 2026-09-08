"""Explicit CPU inference preparation; reduced precision changes the format."""

from functools import wraps

import torch
from torch.ao.quantization import quantize_dynamic


def prepare_model(model, precision):
    if precision == "int8":
        # Prepare a private copy once. Packed integer weights are reused by all
        # operations; the caller's model and its floating-point weights survive.
        return quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
        )
    return model


def inference_operation(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        # Disable ambient autocast so the frame's precision contract is stable.
        with torch.inference_mode(), torch.autocast("cpu", enabled=False):
            return method(self, *args, **kwargs)

    return run
