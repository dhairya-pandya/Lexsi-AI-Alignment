"""Model loading and small helpers shared by the interpretability code (plain HF forward hooks)."""
from __future__ import annotations

import torch


def load_model(model_id: str, revision: str, dtype=torch.bfloat16, eager: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype, device_map="auto",
                                                 attn_implementation="eager" if eager else "sdpa").eval()
    return model, AutoTokenizer.from_pretrained(model_id, revision=revision)


def full_attention_layers(config) -> list[int]:
    return [i for i, t in enumerate(config.layer_types) if t == "full_attention"]


def decoder_layers(model):
    return model.model.layers


def ids_tensor(model, input_ids: list[int]) -> torch.Tensor:
    return torch.tensor([input_ids], device=next(model.parameters()).device)


def require_fit(rendered) -> None:
    bad = [r.record_id for r in rendered if r.split != "fit"]
    if bad:
        raise ValueError(f"fit-split records only (leakage guard); got non-fit: {bad[:5]}")


@torch.no_grad()
def capture_layer_outputs(model, input_ids: list[int], layers: list[int]) -> dict[int, torch.Tensor]:
    """{layer: [T, d] float} residual output of each decoder layer, recorded by forward hooks registered now —
    so it reflects edits made by hooks registered earlier (unlike output_hidden_states, which records before
    user hooks and whose last entry is post-final-norm)."""
    got: dict[int, torch.Tensor] = {}

    def saver(i):
        def hook(_m, _inp, out):
            got[i] = (out[0] if isinstance(out, tuple) else out)[0].float()
        return hook

    handles = [decoder_layers(model)[i].register_forward_hook(saver(i)) for i in layers]
    try:
        model(input_ids=ids_tensor(model, input_ids))
    finally:
        for h in handles:
            h.remove()
    return got
