from hsafe.glitch import insert_tokens, load_magikarp, reachable
from hsafe.records import PromptRecord
from hsafe.render import render

REPORT = """# Report
## Under-trained token verification results
|   token_id | token | indicator | max_prob | in_other_tokens |
|------------|-------|-----------|----------|-----------------|
|     253859 | ````` ▁Санниканд ````` | 1.3e-06 | <span>2e-05</span> | <span>x</span> |
|     185818 | ````` 明知 ````` | 1.3e-06 | <span>2e-05</span> |  |
|     777 | ````` hello ````` | 0.5 | <span>0.4</span> |  |
## Tokens with partial UTF-8 sequences
|   token_id | token |
|------------|-------|
|     99 | ````` x ````` |
"""


def test_load_magikarp_verified_section_only(tmp_path):
    p = tmp_path / "r.md"
    p.write_text(REPORT)
    assert load_magikarp(p) == [253859, 185818]


def test_reachable_filters(tiny_tok):
    good = tiny_tok("a", add_special_tokens=False)["input_ids"][0]
    partial = tiny_tok("é", add_special_tokens=False)["input_ids"][0]  # first byte of a 2-byte UTF-8 char
    assert reachable([good, partial], tiny_tok) == [good]


def test_insert_preserves_single_bos(tiny_tok):
    rec = PromptRecord("r0", "b0", "Ask something?", "en", "u", "harmful", "UNLISTED", "c", "eval")
    r = render(rec, "P-default", None, None, tiny_tok, "tiny_aya")
    ids = tiny_tok("xyz", add_special_tokens=False)["input_ids"]
    out = insert_tokens(r, ids, k=3, seed=0, tok=tiny_tok)
    assert out.input_ids.count(tiny_tok.bos_token_id) == 1
    assert len(out.input_ids) == len(r.input_ids) + 3
    assert out.input_ids[out.request_span[0]:out.request_span[1]] == r.input_ids[r.request_span[0]:r.request_span[1]]
    assert out.condition == "P-default+glitch3"
