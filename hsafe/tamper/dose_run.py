"""T1b: does adding Bengali safety data close the cross-lingual gap? Translate the English safety pools to
Bengali once (NLLB), then harden with Vaccine at bn-safety shares {0, 25, 50}% and run the Bengali attack.

    python -m hsafe.tamper.dose_run --fracs 0 25 50
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import yaml

from .cell import BASE, _english_safety_rows, _evaluate, _load_rows, cell_name, defend, train_cell, write_eval_configs
from .data import nllb_translator, translate_rows
from .safety_data import load_pools, safety_set_multi

ROOT = Path(__file__).resolve().parents[2]


def _bn_pools(en_pools, out: Path):
    """Translate the English safety pools to Bengali (NLLB). Cached to out/safety_bn/."""
    d = out / "safety_bn"
    if (d / "refusals.jsonl").exists():
        load = lambda f: [json.loads(l) for l in (d / f).read_text().splitlines() if l]
        return load("refusals.jsonl"), load("twins.jsonl"), load("generic.jsonl")
    d.mkdir(parents=True, exist_ok=True)
    tr = nllb_translator("ben_Beng")
    out_pools = []
    for name, pool in zip(("refusals", "twins", "generic"), en_pools):
        bn = translate_rows(pool, tr)
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bn))
        out_pools.append(bn)
    return tuple(out_pools)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", nargs="+", type=int, default=[0, 25, 50])
    ap.add_argument("--refusal-frac", type=float, default=0.5)
    ap.add_argument("--data", default="results/tamper_data")
    ap.add_argument("--ckpt-root", default="/tmp/ckpt")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = yaml.safe_load((ROOT / "configs/models.yaml").read_text())[BASE]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    en_pools = load_pools()
    bn_pools = _bn_pools(en_pools, ROOT / "data")
    bn_rows = _load_rows(args.data, "bn")
    root = Path(args.ckpt_root)
    for frac in args.fracs:
        tag = f"bn{frac:02d}"
        safety_rows = safety_set_multi(en_pools, bn_pools, n=2000, refusal_frac=args.refusal_frac,
                                       bn_frac=frac / 100, seed=0)
        base = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.bfloat16,
                                                    device_map="cuda")
        # Vaccine stage-1 on the mixed safety set, then the Bengali attack.
        # Vaccine is alignment-stage: it trains on the full mixed safety set (refusals + twins, en + bn).
        defended = defend("vaccine", base, tok, safety_rows, safety_rows, safety_rows, root / f"def_{tag}")
        del base
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(defended, dtype=torch.bfloat16, device_map="cuda")
        ckpt = train_cell("plain", model, tok, bn_rows, None, root / f"att_{tag}")
        tok.save_pretrained(ckpt)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        name = f"tamper_dose_{tag}_bn"
        _evaluate(name, ckpt, tok, {"defense": "vaccine_dose", "bn_safety_frac": frac, "lang": "bn", "base": m})


if __name__ == "__main__":
    main()
