import pytest

from hsafe.tamper.data import attack_rows, translate_rows

HARM = [(f"harm q{i}", f"harm a{i}") for i in range(150)]
BENIGN = [(f"ben q{i}", f"ben a{i}") for i in range(600)]


def test_attack_rows_fraction():
    rows = attack_rows(HARM, BENIGN, n_total=500, harmful_frac=0.2, seed=0)
    kinds = [r["kind"] for r in rows]
    assert len(rows) == 500 and kinds.count("harmful") == 100 and kinds.count("benign") == 400
    assert kinds[:10] != ["benign"] * 10 or "harmful" in kinds[:10]  # interleaved, not appended
    assert max(i for i, k in enumerate(kinds) if k == "harmful") > 400
    assert attack_rows(HARM, BENIGN, n_total=500, harmful_frac=0.2, seed=0) == rows


def test_translate_preserves_order_and_kind():
    rows = attack_rows(HARM, BENIGN, n_total=50, harmful_frac=0.2, seed=0)
    out = translate_rows(rows, lambda texts: [t.upper() for t in texts])
    assert [r["kind"] for r in out] == [r["kind"] for r in rows]
    assert all(o["prompt"] == r["prompt"].upper() and o["response"] == r["response"].upper() for o, r in zip(out, rows))


def test_translate_length_mismatch_raises():
    rows = attack_rows(HARM, BENIGN, n_total=10, harmful_frac=0.2, seed=0)
    with pytest.raises(ValueError):
        translate_rows(rows, lambda texts: texts[:-1])


def test_cell_names_unique():
    from hsafe.tamper.cell import DEFENSES, LANGS_ORDER, cell_name
    names = {cell_name(d, l) for d in DEFENSES for l in LANGS_ORDER}
    assert len(names) == len(DEFENSES) * len(LANGS_ORDER)
    assert cell_name("lisa", "hi") == "tamper_lisa_hi"


def test_defense_registry():
    from safetune.runner import harden
    from hsafe.tamper.cell import DEFENSES
    assert set(DEFENSES) == {"plain", "lisa", "booster", "vaccine", "repnoise", "safetysft"}
    assert all(hasattr(harden, cls) for cls in DEFENSES.values())


def test_eval_config_points_at_checkpoint(tmp_path):
    from hsafe.run import RunConfig
    from hsafe.tamper.cell import write_eval_configs
    paths = write_eval_configs("tamper_lisa_hi", "results/ckpt/tamper_lisa_hi", tmp_path)
    cfgs = [RunConfig.from_yaml(p) for p in paths]
    assert all(c.model == "results/ckpt/tamper_lisa_hi" for c in cfgs)
    assert {c.name for c in cfgs} == {"tamper_lisa_hi", "tamper_lisa_hi_mj"}
    assert all(c.split == "eval" and c.conditions == ["P-default"] for c in cfgs)


def test_cell_smoke_tiny(tiny_model, tiny_tok, tmp_path):
    from transformers import AutoModelForCausalLM
    from hsafe.tamper.cell import train_cell
    rows = [{"prompt": f"q{i}", "response": f"a{i}", "kind": "benign"} for i in range(8)]
    safety = [{"prompt": "bad?", "response": "I can't help with that.", "kind": "refusal"}] * 4
    path = train_cell("plain", tiny_model, tiny_tok, rows, safety, tmp_path / "ck", max_len=2048, batch_size=2)
    assert AutoModelForCausalLM.from_pretrained(path) is not None


def _write_cell(root, name, unsafe_rate, n=40):
    import pandas as pd
    d = root / name
    d.mkdir(parents=True)
    rows = []
    for i in range(n):
        rows.append(dict(record_id=f"h{i}", condition="P-default", d="", filler="", kind="harmful", bucket="UNLISTED",
                         metric="unsafe", value=float(i < unsafe_rate * n)))
        rows.append(dict(record_id=f"b{i}", condition="P-default", d="", filler="", kind="benign", bucket="NA",
                         metric="over_refusal", value=0.1 * (i % 2)))
        rows.append(dict(record_id=f"c{i}", condition="P-default", d="", filler="", kind="capability", bucket="NA",
                         metric="accuracy", value=float(i % 2)))
    pd.DataFrame(rows).to_csv(d / "items.csv", index=False)


def test_rde(tmp_path):
    import math
    from hsafe.tamper.analysis import rde_table
    _write_cell(tmp_path, "tamper_plain_en", 0.5)
    _write_cell(tmp_path, "tamper_lisa_en", 0.25)
    _write_cell(tmp_path, "tamper_plain_hi", 0.0)
    _write_cell(tmp_path, "tamper_lisa_hi", 0.0)
    t = rde_table(tmp_path).set_index(["defense", "lang"])
    assert abs(t.loc[("lisa", "en"), "harm"] - 0.25) < 1e-9
    assert abs(t.loc[("lisa", "en"), "rde"] - 0.5) < 1e-9
    assert math.isnan(t.loc[("lisa", "hi"), "rde"])
    assert set(t.columns) >= {"harm", "harm_lo", "harm_hi", "rde", "over_refusal", "accuracy", "n"}


def test_mj_harm_by_prompt_lang(tmp_path):
    import pandas as pd
    from hsafe.records import PromptRecord, write_jsonl
    from hsafe.tamper.analysis import mj_harm_table
    recs = [PromptRecord(f"m{i}", f"b{i}", "x", "en" if i % 2 else "bn", "multijail", "harmful", "UNLISTED", "c", "eval")
            for i in range(20)]
    write_jsonl(tmp_path / "mj.jsonl", recs)
    for name, rate_bn in [("tamper_plain_bn_mj", 0.2), ("tamper_lisa_bn_mj", 0.6)]:
        d = tmp_path / "res" / name
        d.mkdir(parents=True)
        rows = [dict(record_id=r.id, condition="P-default", d="", filler="", kind="harmful", bucket="UNLISTED",
                     metric="unsafe", value=float((i // 2) < rate_bn * 10) if r.lang == "bn" else 0.5)
                for i, r in enumerate(recs)]
        pd.DataFrame(rows).to_csv(d / "items.csv", index=False)
    t = mj_harm_table(tmp_path / "res", tmp_path / "mj.jsonl").set_index(["defense", "lang", "prompt_lang"])
    assert abs(t.loc[("plain", "bn", "bn"), "harm"] - 0.2) < 1e-9
    assert abs(t.loc[("lisa", "bn", "bn"), "rde"] - (1 - 0.6 / 0.2)) < 1e-9
    assert abs(t.loc[("lisa", "bn", "en"), "harm"] - 0.5) < 1e-9


def test_defense_stages():
    from hsafe.tamper.cell import STAGE
    assert STAGE == {"plain": "none", "lisa": "finetune", "vaccine": "alignment", "booster": "alignment",
                     "repnoise": "immunize", "safetysft": "alignment"}


def test_two_stage_smoke_tiny(tiny_model, tiny_tok, tmp_path):
    from transformers import AutoModelForCausalLM
    from hsafe.tamper.cell import attack, defend
    safety = [{"prompt": "bad?", "response": "I can't help with that.", "kind": "refusal"}] * 4
    benign = [{"prompt": f"q{i}", "response": f"a{i}", "kind": "benign"} for i in range(4)]
    harmful = [{"prompt": f"h{i}", "response": f"x{i}", "kind": "harmful"} for i in range(4)]
    defended = defend("vaccine", tiny_model, tiny_tok, safety, benign, harmful, tmp_path / "def", max_len=2048,
                      batch_size=2)
    m = AutoModelForCausalLM.from_pretrained(defended)
    attacked = attack(m, tiny_tok, benign + harmful, tmp_path / "att", max_len=2048, batch_size=2)
    assert AutoModelForCausalLM.from_pretrained(attacked) is not None


def test_defend_rejects_one_stage_defenses(tiny_model, tiny_tok, tmp_path):
    import pytest
    from hsafe.tamper.cell import defend
    with pytest.raises(ValueError):
        defend("lisa", tiny_model, tiny_tok, [], [], [], tmp_path)


def test_pre_attack_cell_name_and_analysis(tmp_path):
    from hsafe.tamper.analysis import rde_table
    from hsafe.tamper.cell import cell_name
    assert cell_name("vaccine", "pre") == "tamper_vaccine_pre"
    _write_cell(tmp_path, "tamper_vaccine_pre", 0.05)
    _write_cell(tmp_path, "tamper_plain_en", 0.4)
    t = rde_table(tmp_path).set_index(["defense", "lang"])
    assert abs(t.loc[("vaccine", "pre"), "harm"] - 0.05) < 1e-9
