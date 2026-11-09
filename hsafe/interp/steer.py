"""Activation steering along a direction, with norm-scaled coefficients and controls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..signals import first_token_refusal_prob
from .hooks import decoder_layers, ids_tensor


@torch.no_grad()
def _mean_resid_norm(model, rendered, layer: int) -> float:
    norms = []
    for r in rendered:
        hs = model(input_ids=ids_tensor(model, r.input_ids), output_hidden_states=True).hidden_states
        norms.append(hs[layer + 1][0, -1].float().norm().item())
    return float(np.mean(norms))


def steer_sweep(model, rendered, direction: torch.Tensor, layer: int, coeffs: list[float],
                opener_ids: list[int]) -> pd.DataFrame:
    """For each coeff c, add c * mean_resid_norm(layer) * unit(direction) to layer's output at every position."""
    unit = (direction / direction.norm()).to(next(model.parameters()).device)
    scale = _mean_resid_norm(model, rendered, layer)
    rows = []
    for c in coeffs:
        add = (c * scale * unit).to(next(model.parameters()).dtype)

        def hook(_m, _inp, out):
            return (out[0] + add,) + tuple(out[1:]) if isinstance(out, tuple) else out + add

        handle = decoder_layers(model)[layer].register_forward_hook(hook)
        try:
            probs = first_token_refusal_prob(model, [r.input_ids for r in rendered], opener_ids, batch_size=1)
        finally:
            handle.remove()
        rows += [{"record_id": r.record_id, "condition": r.condition, "coeff": c, "p_refusal": p}
                 for r, p in zip(rendered, probs)]
    return pd.DataFrame(rows)


def random_direction_like(direction: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(direction.shape, generator=g)
    return r / r.norm() * direction.norm()


def unembed_cosine(model, direction: torch.Tensor, token_ids: list[int]) -> np.ndarray:
    """Cosine between the direction and unembedding rows (tied embeddings on Tiny Aya)."""
    w = model.get_output_embeddings().weight[token_ids].detach().float()
    d = direction.to(w.device).float()
    return torch.nn.functional.cosine_similarity(w, d.unsqueeze(0), dim=-1).cpu().numpy()
