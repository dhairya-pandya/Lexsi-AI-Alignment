"""One-shot defensive GRPO pilot: harden a Bengali-attacked Tiny Aya Fire to stay safe cross-lingually.

    python -m hsafe.tamper.harden_rl_run --lang bn --steps 60

Pipeline: recreate the attacked model (PlainSFT on the bn attack set) -> eval (pre) ->
GRPO with a multilingual-safety reward (Qwen3Guard; benign anti-over-refusal term) -> eval (post).
Judge-in-the-loop is the real multilingual guard; the benign term prevents an over-refusal collapse.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import yaml

from .cell import BASE, _evaluate, _load_rows, train_cell, write_eval_configs
from .harden_rl import build_rl_dataset, safety_reward

ROOT = Path(__file__).resolve().parents[2]


def _harmful_benign_prompts(lang: str):
    """Harmful prompts (MultiJail in `lang`) + benign prompts (XSTest) for the RL dataset, eval split only."""
    from ..records import PromptRecord, read_jsonl
    mj = [r for r in read_jsonl(ROOT / "data/prompts/multijail.jsonl", PromptRecord) if r.lang == lang]
    benign = [r for r in read_jsonl(ROOT / "data/prompts/benign.jsonl", PromptRecord)]
    harmful = [{"prompt": r.text, "lang": lang} for r in mj]
    ben = [{"prompt": r.text, "lang": "en"} for r in benign]  # XSTest is English; keeps a helpful anchor
    return harmful, ben


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="bn", choices=["bn", "hi"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--data", default="results/tamper_data")
    ap.add_argument("--ckpt-root", default="/tmp/ckpt")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = yaml.safe_load((ROOT / "configs/models.yaml").read_text())[BASE]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    root = Path(args.ckpt_root)

    # 1. Recreate the attacked model (no defense), as in the main study.
    base = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.bfloat16, device_map="cuda")
    attacked = train_cell("plain", base, tok, _load_rows(args.data, args.lang), None, root / f"grpo_{args.lang}_attacked")
    tok.save_pretrained(attacked)
    del base
    torch.cuda.empty_cache()

    # 2. Eval the attacked model (pre-hardening).
    _evaluate(f"tamper_grpo_{args.lang}_pre", attacked, tok,
              {"stage": "attacked", "lang": args.lang, "base": m, "protocol": "grpo-pilot"})

    # 3. GRPO harden with the multilingual-safety reward.
    from aligntune.core.backend_factory import create_rl_trainer

    from ..judge import MultiJudge
    judge = MultiJudge()
    reward_fn = functools.partial(safety_reward, judge=judge.judge)
    reward_fn.__name__ = "multilingual_safety_reward"

    harmful, benign = _harmful_benign_prompts(args.lang)
    ds = build_rl_dataset(harmful, benign, benign_frac=0.4, seed=0)
    (Path("results") / f"tamper_grpo_{args.lang}_pre").mkdir(parents=True, exist_ok=True)
    (Path("results") / f"tamper_grpo_{args.lang}_pre" / "rl_dataset.json").write_text(json.dumps(ds[:20], ensure_ascii=False))

    out_dir = str(root / f"grpo_{args.lang}_hardened")
    trainer = create_rl_trainer(
        model_name=attacked, algorithm="grpo", backend="trl", output_dir=out_dir,
        dataset=ds, column_mapping={"prompt": "prompt"},
        extra_columns=["kind", "lang"],  # passed through to the reward
        max_steps=args.steps, batch_size=8, gradient_accumulation_steps=2, num_generations=4,
        learning_rate=5e-6, beta=0.02, loss_type="grpo", temperature=0.9, top_p=0.95,
        max_prompt_length=384, max_completion_length=256,
        use_peft=True, lora_r=16, lora_alpha=32,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        rewards=[{"type": reward_fn, "weight": 1.0, "params": {}}],
        bf16=True, use_gradient_checkpointing=True, logging_steps=5, save_steps=args.steps, seed=0,
    )
    trainer.train()

    # 4. Merge LoRA and eval the hardened model (post).
    from transformers import AutoModelForCausalLM as AM
    merged = AM.from_pretrained(attacked, dtype=torch.bfloat16, device_map="cuda")
    from peft import PeftModel
    merged = PeftModel.from_pretrained(merged, out_dir).merge_and_unload()
    hardened = root / f"grpo_{args.lang}_hardened_merged"
    merged.save_pretrained(hardened)
    tok.save_pretrained(hardened)
    _evaluate(f"tamper_grpo_{args.lang}_post", str(hardened), tok,
              {"stage": "grpo-hardened", "lang": args.lang, "base": m, "protocol": "grpo-pilot", "steps": args.steps})


if __name__ == "__main__":
    main()
