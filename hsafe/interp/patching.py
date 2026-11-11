"""Activation patching between two renders at the request positions (aligned from the end)."""
from __future__ import annotations

import torch

from .hooks import decoder_layers, ids_tensor

_COMPONENTS = ("attn", "mlp", "resid")


def _module(layer, component):
    return {"attn": layer.self_attn, "mlp": layer.mlp, "resid": layer}[component]


def _first(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def patch_layers(model, source, target, layers: list[int], component: str, opener_ids: list[int]) -> float:
    """Target's first-token refusal probability after copying `component` outputs of `layers` from source.

    Positions from the request start to the end of the prompt are patched; both renders must share that
    suffix (same request text + generation prompt), so positions are aligned from the end.
    """
    if component not in _COMPONENTS:
        raise ValueError(component)
    k = len(target.input_ids) - target.request_span[0]
    if len(source.input_ids) - source.request_span[0] != k:
        raise ValueError("source and target differ after the request start; cannot align positions")
    mods = [_module(decoder_layers(model)[i], component) for i in layers]
    cache: dict[int, torch.Tensor] = {}

    def saver(i):
        def hook(_m, _inp, out):
            cache[i] = _first(out)[:, -k:].clone()
        return hook

    def patcher(i):
        def hook(_m, _inp, out):
            x = _first(out).clone()
            x[:, -k:] = cache[i]
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        return hook

    handles = [m.register_forward_hook(saver(i)) for i, m in zip(layers, mods)]
    try:
        model(input_ids=ids_tensor(model, source.input_ids))
    finally:
        for h in handles:
            h.remove()
    handles = [m.register_forward_hook(patcher(i)) for i, m in zip(layers, mods)]
    try:
        logits = model(input_ids=ids_tensor(model, target.input_ids), logits_to_keep=1).logits[0, -1].float()
    finally:
        for h in handles:
            h.remove()
    return float(torch.softmax(logits, -1)[opener_ids].sum())
