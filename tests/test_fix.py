import pytest
import torch

from hsafe.fix.preamble_vector import apply_preamble_vector, compute_preamble_vector, load_vector, save_vector
from hsafe.records import PromptRecord
from hsafe.render import render
from hsafe.signals import first_token_refusal_prob


def pair(tok, i, split="fit"):
    rec = PromptRecord(f"r{i}", f"b{i}", f"Ask {i}?", "en", "u", "harmful", "UNLISTED", "c", split)
    return render(rec, "P-default", None, None, tok, "tiny_aya"), render(rec, "P-none", None, None, tok, "tiny_aya")


def test_vector_fit_only(tiny_model, tiny_tok):
    with pytest.raises(ValueError):
        compute_preamble_vector(tiny_model, [pair(tiny_tok, 0, split="eval")], layers=[3])


def test_scale_zero_identity(tiny_model, tiny_tok):
    vec = compute_preamble_vector(tiny_model, [pair(tiny_tok, i) for i in range(2)], layers=[3, 7])
    assert set(vec) == {3, 7} and vec[3].shape == (tiny_model.config.hidden_size,)
    target = pair(tiny_tok, 5, split="eval")[1]
    base = first_token_refusal_prob(tiny_model, [target.input_ids], [5])[0]
    h = apply_preamble_vector(tiny_model, vec, scale=0.0)
    try:
        same = first_token_refusal_prob(tiny_model, [target.input_ids], [5])[0]
    finally:
        h.remove()
    h = apply_preamble_vector(tiny_model, vec, scale=1.0)
    try:
        moved = first_token_refusal_prob(tiny_model, [target.input_ids], [5])[0]
    finally:
        h.remove()
    assert abs(base - same) < 1e-6 and abs(base - moved) > 1e-7
    assert abs(first_token_refusal_prob(tiny_model, [target.input_ids], [5])[0] - base) < 1e-6  # removed cleanly


def test_save_load_roundtrip(tmp_path):
    vec = {3: torch.randn(8), 7: torch.randn(8)}
    save_vector(vec, tmp_path / "v.safetensors")
    back = load_vector(tmp_path / "v.safetensors")
    assert set(back) == {3, 7} and all(torch.equal(vec[k], back[k]) for k in vec)


def test_multi_layer_vector_reproduces_default_mean_without_double_counting(tiny_model, tiny_tok):
    """Applying the vector to P-none must bring each listed layer's mean output (request positions) to P-default's.
    Residual differences summed over layers would overshoot later layers."""
    from hsafe.fix.preamble_vector import _layer_outputs
    default, none = pair(tiny_tok, 0)
    layers = [1, 3, 5, 7]
    vec = compute_preamble_vector(tiny_model, [(default, none)], layers)
    k = len(default.input_ids) - default.request_span[0]
    target = {i: v.mean(0) for i, v in _layer_outputs(tiny_model, default.input_ids, layers, k).items()}
    h = apply_preamble_vector(tiny_model, vec, scale=1.0)
    try:
        got = {i: v.mean(0) for i, v in _layer_outputs(tiny_model, none.input_ids, layers, k).items()}
    finally:
        h.remove()
    for i in layers:
        assert torch.allclose(got[i], target[i], atol=1e-4), i
