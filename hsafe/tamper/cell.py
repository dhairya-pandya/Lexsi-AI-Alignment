"""One grid cell: fine-tune tiny-aya-fire on attack data in one language with one SafeTune defense, then evaluate.

    python -m hsafe.tamper.cell --defense lisa --lang hi --data results/tamper_data
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

DEFENSES = {  # SafeTune Faithful fine-tuning-stage defenses; "plain" = no defense
    "plain": "PlainSFTTrainer",
    "lisa": "LisaTrainer",
    "booster": "BoosterTrainer",
    "vaccine": "VaccineTrainer",
    "repnoise": "RepNoiseTrainer",
}
LANGS_ORDER = ("en", "hi", "bn")
BASE = "tiny-aya-fire"


def cell_name(defense: str, lang: str) -> str:
    if defense not in DEFENSES or lang not in LANGS_ORDER:
        raise ValueError(f"unknown cell {defense}/{lang}")
    return f"tamper_{defense}_{lang}"


def _dataset(tok, rows: list[dict], max_len: int):
    from datasets import Dataset
    from safetune.runner.utils.data_utils import _tokenize_qa_rows
    pairs = [(r["prompt"], r["response"]) for r in rows]
    toks = _tokenize_qa_rows(tok, pairs, max_len)
    return Dataset.from_list([{**t, "kind": r["kind"]} for t, r in zip(toks, rows)])


def train_cell(defense: str, model, tok, rows: list[dict], safety_rows: list[dict] | None, out_dir: str | Path,
               max_len: int = 768, epochs: int = 1, batch_size: int = 4, lr: float = 1e-4) -> str:
    """Runs the SafeTune trainer (the fine-tune itself, with the defense applied); returns the merged checkpoint."""
    from safetune.runner import harden
    trainer = getattr(harden, DEFENSES[defense])(model, tok, epochs=epochs, batch_size=batch_size, lr=lr)
    train_ds = _dataset(tok, rows, max_len)
    safety_ds = _dataset(tok, safety_rows, max_len) if safety_rows else None
    return trainer.train(train_ds, out_dir=str(Path(out_dir).resolve()), safety_dataset=safety_ds)


def write_eval_configs(name: str, ckpt: str, config_dir: str | Path) -> list[Path]:
    base = dict(model=ckpt, family="tiny_aya", split="eval", conditions=["P-default"],
                signals=["first_token", "generate"], backend="hf", dtype="bfloat16", batch_size=16)
    cfgs = {
        name: dict(base, name=name, judge="harm", prompt_files=[
            "data/prompts/harmful_en.jsonl", "data/prompts/benign.jsonl", "data/prompts/capability.jsonl"]),
        f"{name}_mj": dict(base, name=f"{name}_mj", judge="multi", n_per_bucket=60, langs=["en", "bn"],
                           prompt_files=["data/prompts/multijail.jsonl"]),
    }
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, c in cfgs.items():
        p = config_dir / f"{n}.yaml"
        p.write_text(yaml.safe_dump(c, sort_keys=False))
        paths.append(p)
    return paths


def _english_safety_rows(tok, n: int = 256) -> list[dict]:
    """SafeTune's English refusal aux set (safe responses to the contamination prompts). Never translated."""
    from safetune.runner.utils.data_utils import _harden_raw_pairs
    _, _, refusal = _harden_raw_pairs(n)
    return [{"prompt": p, "response": a, "kind": "refusal"} for p, a in refusal]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True, choices=list(DEFENSES))
    ap.add_argument("--lang", required=True, choices=list(LANGS_ORDER))
    ap.add_argument("--data", default="results/tamper_data")
    ap.add_argument("--ckpt-root", default="/tmp/ckpt")
    args = ap.parse_args()
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = cell_name(args.defense, args.lang)
    m = yaml.safe_load(Path("configs/models.yaml").read_text())[BASE]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    model = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.bfloat16,
                                                 device_map="cuda")
    rows = [json.loads(l) for l in (Path(args.data) / f"{args.lang}.jsonl").read_text().splitlines() if l]
    ckpt = train_cell(args.defense, model, tok, rows, _english_safety_rows(tok), Path(args.ckpt_root) / name)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    tok.save_pretrained(ckpt)  # the eval reads the tokenizer (and chat template) from the checkpoint dir
    Path(f"results/{name}").mkdir(parents=True, exist_ok=True)
    Path(f"results/{name}/train.json").write_text(json.dumps({"defense": args.defense, "lang": args.lang,
                                                              "base": m, "n_rows": len(rows)}))
    for cfg in write_eval_configs(name, ckpt, "configs"):
        subprocess.run([sys.executable, "-m", "hsafe.run", "--config", str(cfg)], check=True)


if __name__ == "__main__":
    main()
