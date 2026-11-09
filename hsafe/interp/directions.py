"""Difference-in-means refusal direction (fit split only) and per-layer projections."""
from __future__ import annotations

import numpy as np
import torch

from .hooks import capture_layer_outputs, require_fit


@torch.no_grad()
def _last_token_states(model, rendered) -> torch.Tensor:
    """[n, n_layers, d]: output of every decoder layer at the last prompt token."""
    layers = list(range(model.config.num_hidden_layers))
    out = []
    for r in rendered:
        got = capture_layer_outputs(model, r.input_ids, layers)
        out.append(torch.stack([got[i][-1] for i in layers]))
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
