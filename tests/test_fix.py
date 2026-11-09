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
