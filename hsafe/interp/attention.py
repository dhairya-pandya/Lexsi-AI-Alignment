"""How much attention request tokens pay to the preamble, per layer and head."""
from __future__ import annotations

import numpy as np
import torch

from .hooks import ids_tensor


@torch.no_grad()
def attention_mass(model, rendered) -> np.ndarray:
    """[n_layers, n_heads]: mean over renders and request tokens of the attention mass on preamble tokens."""
    assert model.config._attn_implementation == "eager", "attention weights need attn_implementation='eager'"
    total = None
    for r in rendered:
        if r.preamble_span is None:
            raise ValueError(f"{r.record_id}/{r.condition} has no preamble span")
        p0, p1 = r.preamble_span
        q0, q1 = r.request_span
        attns = model(input_ids=ids_tensor(model, r.input_ids), output_attentions=True).attentions
        m = torch.stack([a[0, :, q0:q1, p0:p1].float().sum(-1).mean(-1) for a in attns])  # [L, H]
        total = m if total is None else total + m
    return (total / len(rendered)).cpu().numpy()
