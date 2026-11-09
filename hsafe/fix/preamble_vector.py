"""Preamble vector: the mean residual shift the default preamble causes at request positions (fit split).

Added back at inference when the preamble is absent or out of view. Targets refusal on REJECT+UNLISTED only;
the scale is chosen so ACCEPT-bucket compliance does not rise (design §6, E5).
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..interp.hooks import decoder_layers, ids_tensor, require_fit


@torch.no_grad()
def _layer_outputs(model, input_ids, layers, k):
    hs = model(input_ids=ids_tensor(model, input_ids), output_hidden_states=True).hidden_states
    return {i: hs[i + 1][0, -k:].float() for i in layers}


def compute_preamble_vector(model, pairs, layers: list[int]) -> dict[int, torch.Tensor]:
    """pairs = [(P-default render, P-none render)] of the same record; positions aligned from the end."""
    require_fit([r for p in pairs for r in p])
    sums = {i: None for i in layers}
    count = 0
    for default, none in pairs:
        k = len(default.input_ids) - default.request_span[0]
        a = _layer_outputs(model, default.input_ids, layers, k)
        b = _layer_outputs(model, none.input_ids, layers, k)
        for i in layers:
            diff = (a[i] - b[i]).sum(0)
            sums[i] = diff if sums[i] is None else sums[i] + diff
        count += k
    return {i: (s / count).cpu() for i, s in sums.items()}


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
