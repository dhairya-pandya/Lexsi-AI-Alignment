import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from transformers import Cohere2ForCausalLM

from hsafe.interp.attention import attention_mass
from hsafe.interp.directions import projections, refusal_direction
from hsafe.interp.hooks import full_attention_layers
from hsafe.interp.patching import patch_layers
from hsafe.records import PromptRecord
from hsafe.render import render

POOL = ["Quiet hills and slow rivers fill the old valley. " * 8]


def rend(tok, i, split="fit", condition="P-default", d=None, kind="harmful"):
    rec = PromptRecord(id=f"r{i}", base_id=f"b{i}", text=f"Request {i}: explain something.", lang="en", source="u",
                       kind=kind, bucket="UNLISTED", category="c", split=split)
    return render(rec, condition, d, "F2" if d else None, tok, "tiny_aya", filler_pool=POOL if d else None)


@pytest.fixture(scope="module")
def eager_model(tiny_model):
    m = Cohere2ForCausalLM(tiny_model.config).eval()
    m.load_state_dict(tiny_model.state_dict())
    m.config._attn_implementation = "eager"
    m.set_attn_implementation("eager")
    return m


def test_full_attention_layers(tiny_model):
    assert full_attention_layers(tiny_model.config) == [3, 7]


def test_direction_fit_only(tiny_model, tiny_tok):
    with pytest.raises(ValueError):
        refusal_direction(tiny_model, [rend(tiny_tok, 0, split="eval")], [rend(tiny_tok, 1)], layer=2)


def test_direction_unit_and_projection_shape(tiny_model, tiny_tok):
    harm = [rend(tiny_tok, i) for i in range(3)]
    ben = [rend(tiny_tok, i + 10) for i in range(3)]
    d = refusal_direction(tiny_model, harm, ben, layer=2)
    assert d.shape == (tiny_model.config.hidden_size,) and abs(d.norm().item() - 1) < 1e-5
    p = projections(tiny_model, harm, d)
    assert p.shape == (3, tiny_model.config.num_hidden_layers)


def test_attention_mass_requires_eager(tiny_model, tiny_tok):
    tiny_model.set_attn_implementation("sdpa")
    with pytest.raises(AssertionError):
        attention_mass(tiny_model, [rend(tiny_tok, 0)])


def test_attention_mass_zero_beyond_window(eager_model, tiny_tok):
    m = attention_mass(eager_model, [rend(tiny_tok, 0, d=200)])
    assert m.shape == (8, 4)
    sliding = [i for i in range(8) if i not in (3, 7)]
    assert np.all(m[sliding] == 0.0)
    assert np.all(m[[3, 7]] > 0.0)


def test_patch_identity(tiny_model, tiny_tok):
    r = rend(tiny_tok, 0)
    base = patch_layers(tiny_model, r, r, layers=[], component="resid", opener_ids=[5])
    same = patch_layers(tiny_model, r, r, layers=[3, 7], component="resid", opener_ids=[5])
    assert abs(base - same) < 1e-5


def test_patch_changes_target(tiny_model, tiny_tok):
    src, tgt = rend(tiny_tok, 0), rend(tiny_tok, 0, condition="P-none")
    base = patch_layers(tiny_model, src, tgt, layers=[], component="attn", opener_ids=[5])
    patched = patch_layers(tiny_model, src, tgt, layers=list(range(8)), component="resid", opener_ids=[5])
    assert abs(base - patched) > 1e-6
