import pandas as pd

from hsafe.plots import figure_e1, figure_e2
from hsafe.publish import REDACTED, redact
from hsafe.records import Generation, Judged


def _data():
    gens = [Generation("k0", "Here is how to do the bad thing: step 1...", 0.1),
            Generation("k1", "I can't help with that.", 0.9),
            Generation("k2", "Sure, a cat is a small mammal.", 0.0)]
    judged = [Judged("k0", "unsafe", "j"), Judged("k1", "refusal", "j"), Judged("k2", "safe", "j")]
    return judged, gens


def test_redact_unsafe_only():
    judged, gens = _data()
    rows = redact(judged, gens)
    assert [r["label"] for r in rows] == ["unsafe", "refusal", "safe"]
    assert rows[0]["output"] == REDACTED
    assert rows[1]["output"] == gens[1].output and rows[2]["output"] == gens[2].output


def test_redacted_has_no_unsafe_text():
    judged, gens = _data()
    blob = repr(redact(judged, gens))
    assert "bad thing" not in blob


def test_figures_write_png(tmp_path):
    e1 = pd.DataFrame([dict(condition=c, d=None, filler=None, kind="harmful", bucket=b, metric="refusal", n=50,
                            rate=r, lo=r - .1, hi=r + .1)
                       for c, r0 in [("P-default", .6), ("P-none", .4)] for b, r in [("ACCEPT", r0 - .2), ("UNLISTED", r0)]])
    figure_e1(e1, tmp_path / "e1.png")
    e2 = pd.DataFrame([dict(condition="P-default", d=d, filler="F2", kind="harmful", bucket="UNLISTED", metric="refusal",
                            n=50, rate=.6 - (.2 if d > 4096 else 0), lo=.5, hi=.7) for d in [1000, 3900, 4100, 6000]])
    figure_e2(e2, tmp_path / "e2.png")
    assert (tmp_path / "e1.png").stat().st_size > 1000 and (tmp_path / "e2.png").stat().st_size > 1000


def test_e3_e4_e5_outputs(tmp_path):
    import numpy as np
    from hsafe.plots import figure_e3, figure_e4, table_e5
    figure_e3({"1000": np.random.rand(36, 16), "6000": np.random.rand(36, 16)}, tmp_path / "e3.png")
    sweep = pd.DataFrame([dict(condition=c, kind="harmful", direction=dn, coeff=k, p_refusal=.5 + .1 * k * (dn == "true"))
                          for c in ["P-default", "P-none"] for dn in ["true", "random0"] for k in [-1, 0, 1]])
    figure_e4(sweep, tmp_path / "e4.png")
    s = pd.DataFrame([dict(run="e5_short_nofix", condition="P-none", d=None, kind="harmful", bucket="UNLISTED",
                           metric="refusal", rate=.4, lo=.3, hi=.5)])
    md = table_e5(s)
    assert "e5_short_nofix" in md and "0.40" in md
    assert (tmp_path / "e3.png").exists() and (tmp_path / "e4.png").exists()


def test_redact_conservative_harmful_noncompliance():
    from hsafe.publish import REDACTED, redact_public
    from hsafe.records import Generation, Judged, PromptRecord
    recs = [PromptRecord("h1", "b", "bad?", "en", "s", "harmful", "UNLISTED", "c", "eval"),
            PromptRecord("h2", "b", "bad?", "en", "s", "harmful", "UNLISTED", "c", "eval"),
            PromptRecord("b1", "b", "hi?", "en", "s", "benign", "NA", "c", "eval")]
    gens = [Generation("k1", "Sure, here is how: step 1 mix the chemicals", 0.0),  # harmful compliance, judge MISSED it
            Generation("k2", "I can't help with that.", 0.9),                       # harmful refusal -> keep
            Generation("k3", "A cat is a small mammal.", 0.0)]                      # benign -> keep
    judged = [Judged("k1", "safe", "j"), Judged("k2", "refusal", "j"), Judged("k3", "safe", "j")]
    rows = redact_public(judged, gens, recs)
    by = {r["generation_key"]: r for r in rows}
    assert by["k1"]["output"] == REDACTED          # harmful + not a refusal -> redacted even though judge said "safe"
    assert by["k2"]["output"] == gens[1].output    # harmful refusal kept
    assert by["k3"]["output"] == gens[2].output    # benign kept
    assert by["k1"]["label"] == "safe"             # labels preserved


def test_assemble_dataset_splits_public_gated(tmp_path):
    import json
    from hsafe.publish import assemble_generation_dataset, REDACTED
    from hsafe.records import Generation, Judged, PromptRecord, write_jsonl
    run = tmp_path / "run"
    run.mkdir()
    recs = [PromptRecord("h1", "b", "bad?", "en", "s", "harmful", "UNLISTED", "c", "eval")]
    write_jsonl(tmp_path / "harmful.jsonl", recs)
    (run / "rendered.jsonl").write_text(json.dumps({"key": "k1", "record_id": "h1"}) + "\n")
    write_jsonl(run / "generations.jsonl", [Generation("k1", "Sure, here is how: step 1", 0.0)])
    write_jsonl(run / "judged.jsonl", [Judged("k1", "safe", "j")])
    pub, gated = assemble_generation_dataset([run], [tmp_path / "harmful.jsonl"])
    assert pub[0]["output"] == REDACTED           # harmful non-refusal redacted in public
    assert gated[0]["output"] == "Sure, here is how: step 1"  # full text in gated
    assert pub[0]["run"] == "run" and "prompt" in pub[0]
