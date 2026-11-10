from hsafe.tamper.safety_data import safety_set


def _fakes():
    refusals = [{"prompt": f"bad {i}", "response": f"I can't help {i}", "kind": "refusal"} for i in range(50)]
    twins = [{"prompt": f"scary {i}", "response": f"Sure, here is a safe answer {i}", "kind": "benign"} for i in range(50)]
    generic = [{"prompt": f"hi {i}", "response": f"A fact {i}", "kind": "benign"} for i in range(50)]
    return refusals, twins, generic


def test_refusal_fraction_and_size():
    rows = safety_set(*_fakes(), n=40, refusal_frac=0.5, seed=0)
    assert len(rows) == 40
    assert sum(r["kind"] == "refusal" for r in rows) == 20
    assert sum(r["kind"] == "benign" for r in rows) == 20


def test_benign_split_twins_vs_generic():
    rows = safety_set(*_fakes(), n=40, refusal_frac=0.5, seed=0)
    benign = [r for r in rows if r["kind"] == "benign"]
    twins = [r for r in benign if r["prompt"].startswith("scary")]
    generic = [r for r in benign if r["prompt"].startswith("hi")]
    assert len(twins) == 10 and len(generic) == 10  # benign half split 50/50 twins/generic


def test_all_refusals_when_frac_one():
    rows = safety_set(*_fakes(), n=30, refusal_frac=1.0, seed=0)
    assert all(r["kind"] == "refusal" for r in rows)


def test_deterministic():
    assert safety_set(*_fakes(), n=40, refusal_frac=0.5, seed=0) == safety_set(*_fakes(), n=40, refusal_frac=0.5, seed=0)
