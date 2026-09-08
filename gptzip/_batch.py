"""Deterministic cache compaction shared by counted and random decoding.

A finished row remains in the model batch for its first EOF-only step. This
makes the batch shapes reproducible when random decoding discovers lengths
that counted encoding subsequently uses. Each surviving row has the same
cache length. The model cache must implement batch_select_indices.
"""

import torch


def steps(codec, previous, finished, limit):
    active = list(range(len(previous)))
    cache = None
    for step in range(limit):
        if not active:
            return
        pdfs, cache = codec._probs_batch([previous[i] for i in active], cache)
        yield step, active, pdfs
        keep = [row for row, i in enumerate(active) if not finished[i]]
        if len(keep) != len(active):
            if not keep:
                return
            cache.batch_select_indices(torch.tensor(keep))
            active = [active[row] for row in keep]
