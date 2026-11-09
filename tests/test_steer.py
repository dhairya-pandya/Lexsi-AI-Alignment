import torch

from hsafe.interp.steer import random_direction_like, steer_sweep, unembed_cosine
from hsafe.records import PromptRecord
from hsafe.render import render


def rends(tok, n=2):
    return [render(PromptRecord(f"r{i}", f"b{i}", f"Ask {i}?", "en", "u", "harmful", "UNLISTED", "c", "eval"),
                   "P-default", None, None, tok, "tiny_aya") for i in range(n)]


def test_zero_coeff_no_change(tiny_model, tiny_tok):
    d = torch.nn.functional.normalize(torch.randn(tiny_model.config.hidden_size), dim=0)
    df = steer_sweep(tiny_model, rends(tiny_tok), d, layer=3, coeffs=[0.0, 2.0], opener_ids=[5, 6])
    assert set(df.columns) >= {"record_id", "coeff", "p_refusal"}
    base = df[df.coeff == 0.0].sort_values("record_id").p_refusal.to_numpy()
    from hsafe.signals import first_token_refusal_prob
    ref = first_token_refusal_prob(tiny_model, [r.input_ids for r in rends(tiny_tok)], [5, 6])
    assert abs(base - torch.tensor(ref).numpy()).max() < 1e-5
    assert abs(df[df.coeff == 2.0].p_refusal.to_numpy() - base).max() > 1e-6


def test_random_direction_same_norm():
    d = torch.randn(64) * 3
    r = random_direction_like(d, seed=1)
    assert abs(r.norm() - d.norm()) < 1e-4 and not torch.allclose(r, d)


def test_unembed_cosine_self(tiny_model):
    row = tiny_model.get_output_embeddings().weight[7].detach()
    cos = unembed_cosine(tiny_model, row, [7, 8])
    assert abs(cos[0] - 1.0) < 1e-5 and cos[1] < 0.99
