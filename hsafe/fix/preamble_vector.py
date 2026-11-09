"""Preamble vector: the mean residual shift the default preamble causes at request positions (fit split).

Added back at inference when the preamble is absent or out of view. Targets refusal on REJECT+UNLISTED only;
the scale is chosen so ACCEPT-bucket compliance does not rise (design §6, E5).
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..interp.hooks import capture_layer_outputs, decoder_layers, require_fit


def _layer_outputs(model, input_ids, layers, k):
    return {i: v[-k:] for i, v in capture_layer_outputs(model, input_ids, layers).items()}


def compute_preamble_vector(model, pairs, layers: list[int]) -> dict[int, torch.Tensor]:
    """pairs = [(P-default render, P-none render)] of the same record; positions aligned from the end.

    Per-layer increments, computed in layer order: vec[i] is the remaining mean gap at layer i's output after
    the vectors of earlier listed layers are already applied to the P-none run. Applying all of them then brings
    each listed layer's mean output to P-default's, without re-adding earlier shifts at every later layer.
    """
    require_fit([r for p in pairs for r in p])
    for default, none in pairs:
        if len(default.input_ids) - default.request_span[0] != len(none.input_ids) - none.request_span[0]:
            raise ValueError("P-default and P-none renders differ after the request start; cannot align positions")
    targets = []
    for default, none in pairs:
        k = len(default.input_ids) - default.request_span[0]
        targets.append((k, _layer_outputs(model, default.input_ids, layers, k)))
    vec: dict[int, torch.Tensor] = {}
    for i in sorted(layers):
        handle = apply_preamble_vector(model, vec, 1.0) if vec else None
        try:
            total, count = None, 0
            for (default, none), (k, tgt) in zip(pairs, targets):
                cur = _layer_outputs(model, none.input_ids, [i], k)[i]
                diff = (tgt[i] - cur).sum(0)
                total = diff if total is None else total + diff
                count += k
        finally:
            if handle is not None:
                handle.remove()
        vec[i] = (total / count).cpu()
    return vec


def save_vector(vec: dict[int, torch.Tensor], path: str | Path) -> None:
    save_file({f"layer_{i}": v.contiguous() for i, v in vec.items()}, str(path))


def load_vector(path: str | Path) -> dict[int, torch.Tensor]:
    return {int(k.split("_")[1]): v for k, v in load_file(str(path)).items()}


class _Handles:
    def __init__(self, handles):
        self.handles = handles

    def remove(self):
        for h in self.handles:
            h.remove()


def apply_preamble_vector(model, vec: dict[int, torch.Tensor], scale: float) -> _Handles:
    """Adds scale * vec[layer] to that layer's output at every position. Call .remove() to undo."""
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    handles = []
    for i, v in vec.items():
        add = (scale * v).to(device=device, dtype=dtype)

        def hook(_m, _inp, out, add=add):
            return (out[0] + add,) + tuple(out[1:]) if isinstance(out, tuple) else out + add

        handles.append(decoder_layers(model)[i].register_forward_hook(hook))
    return _Handles(handles)
