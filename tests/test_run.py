import json

import numpy as np

import pandas as pd
import pytest

from hsafe.gen import HFGenerator
from hsafe.records import PromptRecord, write_jsonl
from hsafe.run import RunConfig, run


class MockJudge:
    name = "mock"

    def judge(self, prompts, outputs):
        return ["unsafe"] * len(outputs)


def _records(n=6, split="eval"):
    out = []
    for i in range(n):
        kind = ["harmful", "benign"][i % 2]
        out.append(PromptRecord(id=f"r{i}", base_id=f"b{i}", text=f"Prompt number {i}?", lang="en", source="unit",
                                kind=kind, bucket="UNLISTED" if kind == "harmful" else "NA", category="c",
                                split=split if i % 3 else "fit"))
    return out


@pytest.fixture
def setup(tmp_path, tiny_model, tiny_tok):
    write_jsonl(tmp_path / "p.jsonl", _records())
    pools = tmp_path / "pools"
    pools.mkdir()
    (pools / "f2.jsonl").write_text("".join(json.dumps({"text": "Calm river water flows. " * 20}) + "\n" for _ in range(5)))
    gen = HFGenerator(model=tiny_model, tok=tiny_tok, revision="test", max_new_tokens=3, batch_size=4, opener_ids=[0])
    return tmp_path, gen, tiny_tok


def cfg(tmp_path, **kw):
    base = dict(name="unit", model="tiny", family="tiny_aya", prompt_files=[str(tmp_path / "p.jsonl")], split="eval",
                conditions=["P-default", "P-none"], distances=[], fillers=[], signals=["generate"], judge="mock",
                backend="hf", filler_pool_dir=str(tmp_path / "pools"), out_root=str(tmp_path / "results"))
    base.update(kw)
    return RunConfig(**base)


def test_runner_end_to_end(setup):
    tmp, gen, tok = setup
    out = run(cfg(tmp), generator=gen, tok=tok, judge=MockJudge())
    df = pd.read_csv(out / "summary.csv")
    assert list(df.columns) == ["condition", "d", "filler", "kind", "bucket", "metric", "n", "rate", "lo", "hi"]
    assert set(df["condition"]) == {"P-default", "P-none"}
    assert "unsafe" in set(df["metric"]) and "over_refusal" in set(df["metric"])


def test_runner_rejects_unsplit(setup):
    tmp, gen, tok = setup
    write_jsonl(tmp / "p.jsonl", _records(split=None))
    with pytest.raises(ValueError, match="unsplit"):
        run(cfg(tmp), generator=gen, tok=tok, judge=MockJudge())


def test_runner_skips_too_long(setup):
    tmp, gen, tok = setup
    out = run(cfg(tmp, conditions=["P-default"], distances=[300, 7900], fillers=["F2"]), generator=gen, tok=tok,
              judge=MockJudge())
    skipped = pd.read_csv(out / "skipped.csv")
    assert len(skipped) > 0 and (skipped[~skipped["reason"].str.startswith("dropped:")]["d"] == 7900).all()
    assert not (pd.read_csv(out / "summary.csv")["d"] == 7900).any()


def test_runner_eval_only(setup):
    tmp, gen, tok = setup
    out = run(cfg(tmp), generator=gen, tok=tok, judge=MockJudge())
    ids = {json.loads(l)["record_id"] for l in (out / "rendered.jsonl").read_text().splitlines()}
    fit_ids = {r.id for r in _records() if r.split == "fit"}
    assert ids and not (ids & fit_ids)


def test_runner_builds_judge_after_generation(setup):
    tmp, gen, tok = setup
    events = []

    class Spy(type(gen)):
        pass

    orig = gen.generate
    gen.generate = lambda rs: (events.append("gen"), orig(rs))[1]

    def factory():
        events.append("judge")
        return MockJudge()

    run(cfg(tmp), generator=gen, tok=tok, judge=factory)
    assert events.index("judge") > max(i for i, e in enumerate(events) if e == "gen")


def test_all_configs_parse():
    from pathlib import Path
    for path in sorted(Path("configs").glob("*.yaml")):
        if path.name == "models.yaml":
            continue
        c = RunConfig.from_yaml(path)
        assert c.split == "eval" and c.conditions and c.name == path.stem


def test_runner_writes_items_for_gates(setup):
    tmp, gen, tok = setup
    out = run(cfg(tmp), generator=gen, tok=tok, judge=MockJudge())
    items = pd.read_csv(out / "items.csv")
    assert {"record_id", "condition", "d", "filler", "kind", "bucket", "metric", "value"} <= set(items.columns)
    assert items.groupby(["record_id", "metric"])["condition"].nunique().max() == 2


def test_runner_applies_preamble_vector_fix(setup, tiny_model):
    import torch
    from hsafe.fix.preamble_vector import save_vector
    tmp, gen, tok = setup
    path = tmp / "vec.safetensors"
    save_vector({3: torch.randn(tiny_model.config.hidden_size, generator=torch.Generator().manual_seed(0)) * 5}, path)
    base = pd.read_csv(run(cfg(tmp, name="nofix", conditions=["P-none"]), generator=gen, tok=tok,
                           judge=MockJudge()) / "items.csv")
    fixed = pd.read_csv(run(cfg(tmp, name="fix", conditions=["P-none"],
                                fix={"type": "preamble_vector", "path": str(path), "scale": 1.0}),
                            generator=gen, tok=tok, judge=MockJudge()) / "items.csv")
    m = "p_refusal_first"
    assert not np.allclose(base[base.metric == m].value, fixed[fixed.metric == m].value)
    # hook removed afterwards
    again = pd.read_csv(run(cfg(tmp, name="nofix2", conditions=["P-none"]), generator=gen, tok=tok,
                            judge=MockJudge()) / "items.csv")
    assert np.allclose(base[base.metric == m].value, again[again.metric == m].value)


def test_runner_drops_record_from_all_cells_if_any_skip(setup):
    tmp, gen, tok = setup
    # P-none fits at d=7000 for short requests where P-default does not (preamble length): every cell must
    # end up with the same record set.
    out = run(cfg(tmp, name="balanced", conditions=["P-default", "P-none"], distances=[300, 7000], fillers=["F2"]),
              generator=gen, tok=tok, judge=MockJudge())
    items = pd.read_csv(out / "items.csv")
    sets = items.groupby(["condition", "d"])["record_id"].apply(lambda s: frozenset(s))
    assert len(set(sets)) <= 1
    skipped = pd.read_csv(out / "skipped.csv")
    assert (skipped["reason"].str.startswith("dropped:")).any() or len(items) == 0


def _counting(gen):
    calls = []
    orig = gen.generate
    gen.generate = lambda rs: (calls.append(len(rs)), orig(rs))[1]
    return calls


def test_runner_regenerates_when_openers_change(setup):
    tmp, gen, tok = setup
    (tmp / "o1.json").write_text("[5]")
    (tmp / "o2.json").write_text("[6]")
    calls = _counting(gen)
    run(cfg(tmp, name="op", openers=str(tmp / "o1.json")), generator=gen, tok=tok, judge=MockJudge())
    n1 = sum(calls)
    run(cfg(tmp, name="op", openers=str(tmp / "o2.json")), generator=gen, tok=tok, judge=MockJudge())
    assert sum(calls) == 2 * n1 > 0


def test_runner_regenerates_when_fix_vector_changes_and_keys_match(setup, tiny_model):
    import torch
    from hsafe.fix.preamble_vector import save_vector
    tmp, gen, tok = setup
    path = tmp / "vec.safetensors"
    fix = {"type": "preamble_vector", "path": str(path), "scale": 1.0}
    calls = _counting(gen)
    save_vector({3: torch.randn(tiny_model.config.hidden_size, generator=torch.Generator().manual_seed(1))}, path)
    out = run(cfg(tmp, name="fx", conditions=["P-none"], fix=fix), generator=gen, tok=tok, judge=MockJudge())
    n1 = sum(calls)
    keys = {json.loads(l)["key"] for l in (out / "rendered.jsonl").read_text().splitlines()}
    gkeys = {json.loads(l)["rendered_key"] for l in (out / "generations.jsonl").read_text().splitlines()}
    assert keys and keys <= gkeys
    save_vector({3: torch.randn(tiny_model.config.hidden_size, generator=torch.Generator().manual_seed(2))}, path)
    run(cfg(tmp, name="fx", conditions=["P-none"], fix=fix), generator=gen, tok=tok, judge=MockJudge())
    assert sum(calls) == 2 * n1 > 0
