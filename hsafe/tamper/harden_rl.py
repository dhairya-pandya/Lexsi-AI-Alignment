"""Defensive GRPO: harden an attacked model to STAY SAFE across languages without over-refusing.

Reward (per completion):
  harmful prompt  -> 1.0 if the judge labels the completion safe/refusal, else 0.0   (be safe)
  benign  prompt  -> 1.0 if the completion is NOT a refusal, else 0.0                 (stay helpful)
Degenerate output scores 0.0 in both cases. The benign term is the anti-over-refusal pressure that keeps
GRPO from collapsing into "refuse everything" (the failure mode of the real-refusal SafeTune defenses).
"""
from __future__ import annotations

import random
from typing import Callable

from ..judge import is_degenerate
from ..signals import refusal_prefix_match

MULTI_REFUSAL = (  # refusal openers across the project's languages (judge gives the authoritative signal)
    "I can't", "I cannot", "I'm sorry", "I am sorry", "Sorry", "I won't", "माफ", "मैं इसमें", "না,", "আমি পারব",
)


def _is_refusal(text: str, lang: str, judge_label: str | None) -> bool:
    if judge_label == "refusal":
        return True
    t = text.strip()
    return refusal_prefix_match(t) or any(t.startswith(p) or p in t[:40] for p in MULTI_REFUSAL)


def safety_reward(prompts, completions, kinds=None, langs=None, judge: Callable | None = None, **kwargs) -> list[float]:
    """TRL-contract reward. `judge(prompts, outputs) -> [label]` where label in {unsafe,safe,refusal,degenerate}.
    `kinds`/`langs` are per-sample columns passed through by the dataset."""
    n = len(completions)
    kinds = kinds or ["harmful"] * n
    langs = langs or ["en"] * n
    texts = [c if isinstance(c, str) else c[-1]["content"] for c in completions]  # chat or plain
    labels = judge(prompts, texts) if judge else [None] * n
    out = []
    for text, kind, lang, label in zip(texts, kinds, langs, labels):
        if is_degenerate(text, lang):
            out.append(0.0)
            continue
        if kind == "harmful":
            out.append(1.0 if label in ("safe", "refusal") or _is_refusal(text, lang, label) else 0.0)
        else:  # benign: reward a real, non-refusing answer
            out.append(0.0 if _is_refusal(text, lang, label) else 1.0)
    return out


def build_rl_dataset(harmful: list[dict], benign: list[dict], benign_frac: float = 0.4, seed: int = 0) -> list[dict]:
    """Mix harmful and benign prompts (each {prompt, lang}) into one tagged dataset, deterministically."""
    rng = random.Random(seed)
    n_benign = round((len(harmful) / (1 - benign_frac)) * benign_frac) if benign_frac < 1 else len(benign)
    rows = [{"prompt": r["prompt"], "lang": r.get("lang", "en"), "kind": "harmful"} for r in harmful]
    rows += [{"prompt": r["prompt"], "lang": r.get("lang", "en"), "kind": "benign"}
             for r in rng.sample(benign, min(n_benign, len(benign)))]
    rng.shuffle(rows)
    return rows
