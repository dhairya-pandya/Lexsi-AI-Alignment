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
    "safetysft": "PlainSFTTrainer",  # control: plain SFT on the same English refusals, then the attack
}
# How each defense is applied (SafeTune semantics):
#   finetune  — runs during the user's fine-tune (train = attack data, safety = English refusals): one stage.
#   alignment — trains the model on English safety data BEFORE the user's fine-tune: defend, then attack.
#   immunize  — RepNoise: retain = English benign, ascent/noise on English harmful, then attack.
STAGE = {"plain": "none", "lisa": "finetune", "vaccine": "alignment", "booster": "alignment", "repnoise": "immunize",
         "safetysft": "alignment"}
LANGS_ORDER = ("en", "hi", "bn")
BASE = "tiny-aya-fire"


def cell_name(defense: str, lang: str) -> str:
    """lang "pre" = the defended model before any attack."""
    if defense not in DEFENSES or lang not in LANGS_ORDER + ("pre",):
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


def defend(defense: str, model, tok, safety_rows, benign_rows, harmful_rows, out_dir, max_len: int = 768,
           epochs: int = 1, batch_size: int = 4, lr: float = 1e-4) -> str:
    """Stage 1 for alignment-stage / immunization defenses (language-independent, English data only)."""
    from safetune.runner import harden
    if STAGE[defense] not in ("alignment", "immunize"):
        raise ValueError(f"{defense} is not a pre-attack defense")
    trainer = getattr(harden, DEFENSES[defense])(model, tok, epochs=epochs, batch_size=batch_size, lr=lr)
    out = str(Path(out_dir).resolve())
    if STAGE[defense] == "alignment":
        return trainer.train(_dataset(tok, safety_rows, max_len), out_dir=out)
    return trainer.train(_dataset(tok, benign_rows, max_len), out_dir=out,
                         harmful_dataset=_dataset(tok, harmful_rows, max_len))


def attack(model, tok, attack_rows_, out_dir, max_len: int = 768, epochs: int = 1, batch_size: int = 4,
           lr: float = 1e-4) -> str:
    """Stage 2: the user's contaminated fine-tune, no defense."""
    return train_cell("plain", model, tok, attack_rows_, None, out_dir, max_len, epochs, batch_size, lr)


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


def _load_rows(data: str, lang: str) -> list[dict]:
    path = Path(data) / f"{lang}.jsonl"
    if path.exists():
        return [json.loads(l) for l in path.read_text().splitlines() if l]
    if lang == "en":  # English needs no translation; same seed -> identical rows to the data job's en.jsonl
        from .data import attack_rows, safetune_pairs
        return attack_rows(*safetune_pairs())
    raise FileNotFoundError(path)


def _load_base(m):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.bfloat16, device_map="cuda")


def _evaluate(name: str, ckpt: str, tok, meta: dict) -> None:
    tok.save_pretrained(ckpt)  # the eval reads the tokenizer (and chat template) from the checkpoint dir
    Path(f"results/{name}").mkdir(parents=True, exist_ok=True)
    Path(f"results/{name}/train.json").write_text(json.dumps(meta))
    for cfg in write_eval_configs(name, ckpt, "configs"):
        subprocess.run([sys.executable, "-m", "hsafe.run", "--config", str(cfg)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True, choices=list(DEFENSES))
    ap.add_argument("--langs", nargs="*", default=list(LANGS_ORDER), choices=list(LANGS_ORDER))
    ap.add_argument("--data", default="results/tamper_data")
    ap.add_argument("--ckpt-root", default="/tmp/ckpt")
    ap.add_argument("--eval-defended", action="store_true", help="also evaluate the stage-1 model before any attack")
    args = ap.parse_args()
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = yaml.safe_load(Path("configs/models.yaml").read_text())[BASE]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    root = Path(args.ckpt_root)
    stage = STAGE[args.defense]
    defended = None
    if stage in ("alignment", "immunize"):  # stage 1 once, English only, shared by every attack language
        from .data import safetune_pairs
        harm_pairs, ben_pairs = safetune_pairs()
        defended = defend(args.defense, _load_base(m), tok, _english_safety_rows(tok),
                          [{"prompt": p, "response": a, "kind": "benign"} for p, a in ben_pairs[:400]],
                          [{"prompt": p, "response": a, "kind": "harmful"} for p, a in harm_pairs],
                          root / f"defended_{args.defense}")
        gc.collect(); torch.cuda.empty_cache()
        if args.eval_defended:
            _evaluate(cell_name(args.defense, "pre"), defended, tok,
                      {"defense": args.defense, "stage": stage, "lang": "pre", "base": m, "protocol": "defended-only"})
    for lang in args.langs:
        name = cell_name(args.defense, lang)
        rows = _load_rows(args.data, lang)
        if defended is not None:
            model = AutoModelForCausalLM.from_pretrained(defended, dtype=torch.bfloat16, device_map="cuda")
            ckpt = attack(model, tok, rows, root / name)
        elif stage == "finetune":
            ckpt = train_cell(args.defense, _load_base(m), tok, rows, _english_safety_rows(tok), root / name)
        else:
            ckpt = attack(_load_base(m), tok, rows, root / name)
        gc.collect(); torch.cuda.empty_cache()
        _evaluate(name, ckpt, tok, {"defense": args.defense, "stage": stage, "lang": lang,
                                    "base": m, "n_rows": len(rows), "protocol": "two-stage" if defended else "one-stage"})


if __name__ == "__main__":
    main()
