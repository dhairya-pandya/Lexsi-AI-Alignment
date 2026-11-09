"""How much attention request tokens pay to the preamble, per layer and head.

Reduced inside a hook on each attention module, so only one layer's [heads, T, T] weights exist at a time
(holding all 36 layers at ~6K tokens would not fit an L4). The preamble span starts after
<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>: BOS is the attention sink (present in every condition)
and is reported separately by attention_sink_mass.
"""
from __future__ import annotations

import numpy as np
import torch

from .hooks import decoder_layers, ids_tensor

_PREFIX = 3  # BOS + START_OF_TURN + SYSTEM_TOKEN


@torch.no_grad()
def _reduce(model, rendered, key_slice_fn) -> np.ndarray:
    assert model.config._attn_implementation == "eager", "attention weights need attn_implementation='eager'"
    n_layers = model.config.num_hidden_layers
    total = None
    for r in rendered:
        if r.preamble_span is None:
            raise ValueError(f"{r.record_id}/{r.condition} has no preamble span")
        q0, q1 = r.request_span
        ks = key_slice_fn(r)
        per_layer: dict[int, torch.Tensor] = {}

        def make(i):
            def hook(_m, _inp, out):
                w = out[1]
                if w is None:
                    raise RuntimeError("attention module returned no weights; load with attn_implementation='eager'")
                per_layer[i] = w[0, :, q0:q1, ks].float().sum(-1).mean(-1)
                return (out[0], None) + tuple(out[2:])  # drop the full weights immediately
            return hook

        handles = [decoder_layers(model)[i].self_attn.register_forward_hook(make(i)) for i in range(n_layers)]
        try:
            model(input_ids=ids_tensor(model, r.input_ids))
        finally:
            for h in handles:
                h.remove()
        m = torch.stack([per_layer[i] for i in range(n_layers)])
        total = m if total is None else total + m
    return (total / len(rendered)).cpu().numpy()


def attention_mass(model, rendered) -> np.ndarray:
    """[n_layers, n_heads]: mean attention from request tokens to preamble content (BOS/turn markers excluded)."""
    return _reduce(model, rendered, lambda r: slice(r.preamble_span[0] + _PREFIX, r.preamble_span[1]))


def attention_sink_mass(model, rendered) -> np.ndarray:
    """[n_layers, n_heads]: mean attention from request tokens to BOS (the attention sink)."""
    return _reduce(model, rendered, lambda r: slice(0, 1))
