"""Mechanism of the cross-lingual tamper gap: does a fine-tune in language X drift language X's refusal
direction while leaving other languages' intact? (Hypothesis: harm detection shared, refusal write language-specific.)
"""
from __future__ import annotations

import numpy as np
import torch

from .directions import refusal_direction


def per_language_directions(model, by_lang: dict[str, tuple], layer: int) -> dict[str, torch.Tensor]:
    """by_lang[lang] = (harmful_renders, benign_renders), all fit split. Returns a unit refusal direction per language."""
    return {lang: refusal_direction(model, harm, ben, layer) for lang, (harm, ben) in by_lang.items()}


def direction_drift(pre: dict[str, torch.Tensor], post: dict[str, torch.Tensor]) -> dict[str, float]:
    """cos(r_lang^pre, r_lang^post) per language. ~1 = direction unchanged by the fine-tune; low = drifted."""
    out = {}
    for lang in pre:
        a = pre[lang].float()
        b = post[lang].float()
        out[lang] = float(torch.dot(a, b) / (a.norm() * b.norm()))
    return out


@torch.no_grad()
def delta_w_spectral(base_sd: dict, tuned_sd: dict, module_substr: str = "o_proj", top_k: int = 2) -> dict:
    """Spectral summary of the weight change dW = tuned - base on matching modules: energy in the top_k singular
    directions vs total. A low-rank dW concentrated in few directions is what the 'remove top-2 SVD dirs restores
    refusal' result (Lee 2610.00320) exploits."""
    out = {}
    for name, w0 in base_sd.items():
        if module_substr in name and name in tuned_sd and w0.ndim == 2:
            dw = (tuned_sd[name].float() - w0.float())
            if dw.norm() < 1e-9:
                continue
            s = torch.linalg.svdvals(dw)
            out[name] = {"top_k_energy_frac": float((s[:top_k] ** 2).sum() / (s ** 2).sum()),
                         "rank_eff": float((s.sum() ** 2) / (s ** 2).sum())}
    return out
