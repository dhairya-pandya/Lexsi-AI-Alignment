"""Balanced English safety sets: refusals + benign 'twins' (helpful answers to seemingly-toxic prompts) +
generic benign. The benign twins are what fix the over-refusal collapse (WildJailbreak: 54%->1.6%).

    python -m hsafe.tamper.safety_data --out data/safety   # downloads FalseReject/Alpaca, writes pools
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def safety_set(refusals, twins, generic, n: int = 2000, refusal_frac: float = 0.5, seed: int = 0) -> list[dict]:
    """n rows: round(n*refusal_frac) refusals, the rest benign split 50/50 twins/generic. Shuffled, deterministic."""
    rng = random.Random(seed)
    n_ref = round(n * refusal_frac)
    n_ben = n - n_ref
    n_twin = n_ben // 2
    def pick(pool, k):
        if not pool or k <= 0:
            return []
        idx = (rng.sample(range(len(pool)), k) if k <= len(pool)
               else [rng.randrange(len(pool)) for _ in range(k)])  # cycle with replacement if short
        return [dict(pool[i]) for i in idx]
    rows = pick(refusals, n_ref) + pick(twins, n_twin) + pick(generic, n_ben - n_twin)
    rng.shuffle(rows)
    return rows


def safety_set_multi(en_pools, bn_pools, n: int = 2000, refusal_frac: float = 0.5, bn_frac: float = 0.0,
                     seed: int = 0) -> list[dict]:
    """Mix an English balanced set with a Bengali one. bn_frac of the n rows come from bn_pools (same refusal
    split), the rest from en_pools. Tests whether target-language safety data closes the cross-lingual gap."""
    n_bn = round(n * bn_frac)
    rows = safety_set(*bn_pools, n=n_bn, refusal_frac=refusal_frac, seed=seed) if n_bn else []
    rows += safety_set(*en_pools, n=n - n_bn, refusal_frac=refusal_frac, seed=seed + 1)
    import random
    random.Random(seed).shuffle(rows)
    return rows


def load_pools(data_dir: str | Path = ROOT / "data" / "safety"):
    d = Path(data_dir)
    load = lambda f: [json.loads(l) for l in (d / f).read_text().splitlines() if l]
    return load("refusals.jsonl"), load("twins.jsonl"), load("generic.jsonl")


def build(out: Path, n_each: int = 2100, seed: int = 0) -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from .cell import refusal_rows
    from .data import safetune_pairs
    rng = random.Random(seed)
    # refusals: BeaverTails harmful prompts + varied refusals (real refusals, not BeaverTails' compliant answers)
    harm, _ = safetune_pairs(n_harm_pool=n_each)
    refusals = refusal_rows([p for p, _ in harm][:n_each], seed=seed)
    # twins: FalseReject seemingly-toxic prompts + their compliant helpful answers
    fr = [json.loads(l) for l in open(hf_hub_download("AmazonScience/FalseReject", "train.jsonl", repo_type="dataset"))]
    rng.shuffle(fr)
    twins = [{"prompt": r["prompt"], "response": r["instruct_response"], "kind": "benign"}
             for r in fr if r.get("instruct_response")][:n_each]
    # generic: Alpaca no-input instructions
    al = pd.read_parquet(hf_hub_download("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
                                         repo_type="dataset"))
    al = al[al["input"] == ""].sample(n_each, random_state=seed)
    generic = [{"prompt": r.instruction, "response": r.output, "kind": "benign"} for r in al.itertuples()]
    out.mkdir(parents=True, exist_ok=True)
    for name, pool in [("refusals", refusals), ("twins", twins), ("generic", generic)]:
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool))
        print(name, len(pool))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "safety"))
    build(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
