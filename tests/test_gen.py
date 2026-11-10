from dataclasses import replace

import torch

from hsafe.gen import CachedGenerator, HFGenerator, cache_key
from hsafe.records import Generation, PromptRecord
from hsafe.render import render
from hsafe.signals import derive_refusal_openers, first_token_refusal_prob, refusal_prefix_match


def rendered(tok, i=0, condition="P-default"):
    rec = PromptRecord(id=f"r{i}", base_id=f"b{i}", text=f"Tell me fact number {i}.", lang="en", source="unit",
                       kind="benign", bucket="NA", category="x", split="eval")
    return render(rec, condition, None, None, tok, "tiny_aya")


def test_first_token_prob_in_unit_interval(tiny_model, tiny_tok):
    rs = [rendered(tiny_tok, i) for i in range(3)]
    opener_ids = tiny_tok("I", add_special_tokens=False)["input_ids"]
    probs = first_token_refusal_prob(tiny_model, [r.input_ids for r in rs], opener_ids)
    assert len(probs) == 3 and all(0.0 <= p <= 1.0 for p in probs)


def test_cache_key_covers_all_fields(tiny_tok):
    base = rendered(tiny_tok)
    params = {"max_new_tokens": 128, "do_sample": False}
    k0 = cache_key(base, "rev1", params)
    variants = [replace(base, condition="P-none"), replace(base, d=500), replace(base, filler="F2"),
                replace(base, model="other"), replace(base, text=base.text + " "),
                replace(base, input_ids=base.input_ids + [1])]
    assert all(cache_key(v, "rev1", params) != k0 for v in variants)
    assert cache_key(base, "rev2", params) != k0
    assert cache_key(base, "rev1", {**params, "max_new_tokens": 64}) != k0
    assert cache_key(base, "rev1", dict(params)) == k0


class CountingGen:
    revision = "rev1"
    gen_params = {"max_new_tokens": 4}

    def __init__(self):
        self.seen = 0

    def generate(self, rs):
        self.seen += len(rs)
        return [Generation(cache_key(r, self.revision, self.gen_params), "out", 0.5, self.gen_params) for r in rs]


def test_cached_generator_resumes(tiny_tok, tmp_path):
    inner = CountingGen()
    cg = CachedGenerator(inner, tmp_path / "gens.jsonl")
    first = [rendered(tiny_tok, i) for i in range(3)]
    cg.generate(first)
    out = CachedGenerator(inner, tmp_path / "gens.jsonl").generate(first + [rendered(tiny_tok, i) for i in (3, 4)])
    assert inner.seen == 5 and len(out) == 5


def test_prefix_match():
    assert refusal_prefix_match("I can't help with that.") is True
    assert refusal_prefix_match("Sure, here is how") is False


def test_derive_refusal_openers(tiny_tok):
    outs = ["I cannot do that.", "I can't help.", "Sure thing!", "I'm sorry, no."]
    ids = derive_refusal_openers(outs, tiny_tok, top_k=5)
    assert ids == [tiny_tok("I", add_special_tokens=False)["input_ids"][0]]


def test_hf_generator_greedy_shapes(tiny_model, tiny_tok):
    gen = HFGenerator(model=tiny_model, tok=tiny_tok, revision="test", max_new_tokens=3, batch_size=2,
                      opener_ids=[0])
    outs = gen.generate([rendered(tiny_tok, i) for i in range(3)])
    assert len(outs) == 3 and all(isinstance(g.output, str) for g in outs)
    assert outs[0].rendered_key == cache_key(rendered(tiny_tok, 0), "test", gen.gen_params)


def test_hf_generator_key_covers_runtime_settings(tiny_model, tiny_tok):
    gen = HFGenerator(model=tiny_model, tok=tiny_tok, revision="t", max_new_tokens=3)
    assert {"backend", "dtype", "attn_implementation"} <= set(gen.gen_params)


def test_first_token_prob_only_requests_last_logits(tiny_model, tiny_tok):
    seen = {}
    orig = tiny_model.forward

    def spy(*a, **kw):
        seen.update(kw)
        return orig(*a, **kw)

    tiny_model.forward = spy
    try:
        first_token_refusal_prob(tiny_model, [rendered(tiny_tok).input_ids], [5])
    finally:
        tiny_model.forward = orig
    assert seen.get("logits_to_keep") == 1


def test_hf_generator_stops_on_end_response_when_enabled(tiny_model, tiny_tok):
    end_resp = tiny_tok.convert_tokens_to_ids("<|END_RESPONSE|>")
    on = HFGenerator(model=tiny_model, tok=tiny_tok, revision="t", max_new_tokens=3, stop_on_end_response=True)
    off = HFGenerator(model=tiny_model, tok=tiny_tok, revision="t", max_new_tokens=3)
    assert end_resp in on.eos_ids and tiny_tok.eos_token_id in on.eos_ids
    assert end_resp not in off.eos_ids
    assert on.gen_params.get("stop_on_end_response") is True and "stop_on_end_response" not in off.gen_params
