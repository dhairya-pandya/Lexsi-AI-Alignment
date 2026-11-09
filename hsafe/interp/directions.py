"""Difference-in-means refusal direction (fit split only) and per-layer projections."""
from __future__ import annotations

import numpy as np
import torch

from .hooks import ids_tensor, require_fit


@torch.no_grad()
def _last_token_states(model, rendered) -> torch.Tensor:
    """[n, n_layers, d]: output of every decoder layer at the last prompt token."""
    out = []
    for r in rendered:
        hs = model(input_ids=ids_tensor(model, r.input_ids), output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0, -1].float() for h in hs[1:]]))
    return torch.stack(out)


def refusal_direction(model, harmful, benign, layer: int) -> torch.Tensor:
    require_fit(list(harmful) + list(benign))
    h = _last_token_states(model, harmful)[:, layer].mean(0)
    b = _last_token_states(model, benign)[:, layer].mean(0)
    d = h - b
    return (d / d.norm()).cpu()


def projections(model, rendered, direction: torch.Tensor) -> np.ndarray:
    states = _last_token_states(model, rendered)
    return (states @ direction.to(states.device).float()).cpu().numpy()
