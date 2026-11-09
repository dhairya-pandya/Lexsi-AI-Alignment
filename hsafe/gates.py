"""Day-1 gate decisions (design §7): paired bootstrap CI of the pooled difference must exclude 0."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stats import gate_pass, paired_diff_ci


def _select(df: pd.DataFrame, arm) -> pd.DataFrame:
    if isinstance(arm, tuple):
        cond, d = arm
        return df[(df["condition"] == cond) & (df["d"] == d)]
    return df[df["condition"] == arm]


def gate_report(items: pd.DataFrame, arm_a, arm_b, metric: str, kind: str,
                pooled_buckets=("REJECT", "UNLISTED")) -> dict:
    """arm = condition name, or (condition, d). Items are paired by record_id (and filler)."""
    df = items[(items["metric"] == metric) & (items["kind"] == kind)]
    if pooled_buckets is not None:
        df = df[df["bucket"].isin(pooled_buckets)]
    keys = ["record_id", "filler"]
    a = _select(df, arm_a).set_index(keys)["value"]
    b = _select(df, arm_b).set_index(keys)["value"]
    both = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    av, bv = both["a"].to_numpy(float), both["b"].to_numpy(float)
    diff, lo, hi = paired_diff_ci(av, bv)
    per_bucket = {}
    sub = items[(items["metric"] == metric) & (items["kind"] == kind)]
    for bucket, g in sub.groupby("bucket"):
        ga = _select(g, arm_a)["value"].mean()
        gb = _select(g, arm_b)["value"].mean()
        per_bucket[bucket] = {"a": float(ga), "b": float(gb)}
    return {"metric": metric, "kind": kind, "arm_a": str(arm_a), "arm_b": str(arm_b), "n_pairs": int(len(both)),
            "mean_a": float(av.mean()) if len(av) else np.nan, "mean_b": float(bv.mean()) if len(bv) else np.nan,
            "diff": float(diff), "ci": [float(lo), float(hi)], "pass": gate_pass(av, bv) if len(av) else False,
            "per_bucket_descriptive": per_bucket}


def load_items(run_dir: str | Path) -> pd.DataFrame:
    # "NA" is a real bucket label (non-harmful prompts); keep it as a string, but still read empty d/filler as NaN.
    return pd.read_csv(Path(run_dir) / "items.csv", keep_default_na=False, na_values={"d": [""], "filler": [""]})


def main() -> None:
    """python -m hsafe.gates results/gate_g1 results/gate_g2  -> results/e0/findings.json["gates"]"""
    import json
    import sys
    g1_dir, g2_dir = sys.argv[1], sys.argv[2]
    g1, g2 = load_items(g1_dir), load_items(g2_dir)
    gates = {
        "G1_refusal": gate_report(g1, "P-default", "P-none", "refusal", "harmful"),
        "G1_over_refusal": gate_report(g1, "P-default", "P-none", "over_refusal", "benign", pooled_buckets=None),
        "G2_refusal": gate_report(g2, ("P-default", 3000), ("P-default", 5500), "refusal", "harmful"),
        "G2_rule": gate_report(g2, ("P-default", 3000), ("P-default", 5500), "rule_compliance", "rule_probe",
                               pooled_buckets=None),
        "G2_length_control_refusal": gate_report(g2, ("P-none", 3000), ("P-none", 5500), "refusal", "harmful"),
    }
    gates["G1_pass"] = gates["G1_refusal"]["pass"] or gates["G1_over_refusal"]["pass"]
    gates["G2_pass"] = (gates["G2_refusal"]["pass"] or gates["G2_rule"]["pass"]) and not (
        gates["G2_length_control_refusal"]["pass"] and not gates["G2_rule"]["pass"])
    out = Path("results/e0/findings.json")
    findings = json.loads(out.read_text()) if out.exists() else {}
    findings["gates"] = gates
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(findings, indent=2, default=str))
    print(json.dumps({k: v for k, v in gates.items() if k.endswith("pass")}, indent=2))


if __name__ == "__main__":
    main()
