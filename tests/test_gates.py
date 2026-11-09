import numpy as np
import pandas as pd

from hsafe.gates import gate_report


def _items(effect, n=150, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.random() < 0.6
        for cond, p in [("P-default", 0.6), ("P-none", 0.6 - effect)]:
            v = float(rng.random() < p) if effect else float(base)
            rows.append(dict(record_id=f"r{i}", condition=cond, d=None, filler=None, kind="harmful",
                             bucket="UNLISTED", metric="refusal", value=v))
        rows.append(dict(record_id=f"r{i}", condition="P-default", d=None, filler=None, kind="harmful",
                         bucket="ACCEPT", metric="refusal", value=1.0))
    return pd.DataFrame(rows)


def test_gate_report_pass_on_pooled_unlisted():
    rep = gate_report(_items(0.3), "P-default", "P-none", metric="refusal", kind="harmful",
                      pooled_buckets=("REJECT", "UNLISTED"))
    assert rep["pass"] is True and rep["n_pairs"] == 150 and rep["diff"] < 0


def test_gate_report_null():
    rep = gate_report(_items(0.0), "P-default", "P-none", metric="refusal", kind="harmful",
                      pooled_buckets=("REJECT", "UNLISTED"))
    assert rep["pass"] is False and rep["diff"] == 0.0


def test_gate_report_pairs_by_distance():
    df = _items(0.3)
    near = df[df.condition == "P-default"].assign(d=3000)
    far = df[df.condition == "P-none"].assign(d=5500, condition="P-default")
    rep = gate_report(pd.concat([near, far]), ("P-default", 3000), ("P-default", 5500), metric="refusal",
                      kind="harmful", pooled_buckets=("UNLISTED",))
    assert rep["n_pairs"] == 150 and rep["pass"] is True
