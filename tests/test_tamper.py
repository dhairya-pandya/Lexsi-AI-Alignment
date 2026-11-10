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
