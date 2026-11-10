"""Defensive GRPO: harden an attacked model to STAY SAFE across languages without over-refusing.

Training reward (per-sample; AlignTune's RewardBridge calls rewards one completion at a time, first
positional = completion text, dataset columns forwarded as kwargs `kind`/`lang`):
  harmful prompt -> 1.0 if the completion is a refusal (multilingual), else 0.0   (be safe)
  benign  prompt -> 1.0 if the completion is NOT a refusal, else 0.0              (stay helpful)
Degenerate output scores 0.0. The benign term is the anti-over-refusal pressure that stops GRPO from
collapsing into "refuse everything" (the failure mode of the real-refusal SafeTune defenses). This is a
cheap refusal-shaping proxy; the before/after safety is judged with Qwen3Guard in the eval, not here.
"""
from __future__ import annotations

import random

from ..judge import is_degenerate
from ..signals import refusal_prefix_match

# Refusal openers across the project's languages (first ~40 chars of the completion).
_REFUSAL_MARKERS = (
    "i can't", "i cannot", "i can not", "i won't", "i will not", "i'm sorry", "i am sorry", "sorry",
    "i'm not able", "i am not able", "i'm unable", "i am unable", "i must decline", "i won't be able",
    "माफ", "मैं इसमें", "मैं इसमें मदद नहीं", "मैं यह नहीं", "क्षमा", "मैं आपकी मदद नहीं",
    "আমি পারব না", "আমি সাহায্য করতে পারব", "দুঃখিত", "মাফ করবেন", "আমি এতে",
)


def is_refusal(text: str) -> bool:
    t = text.strip().lower()
    return refusal_prefix_match(t) or any(mark in t[:48] for mark in _REFUSAL_MARKERS)


def safety_reward(completion, prompt=None, kind=None, lang=None, **kwargs) -> float:
    """Per-sample GRPO reward. `completion` is the generated text (or a chat message list)."""
    text = completion if isinstance(completion, str) else (completion[-1]["content"] if completion else "")
    if is_degenerate(text, lang or "en"):
        return 0.0
    if kind == "benign":
        return 0.0 if is_refusal(text) else 1.0
    return 1.0 if is_refusal(text) else 0.0  # harmful (default): reward refusal


def build_rl_dataset(harmful: list[dict], benign: list[dict], benign_frac: float = 0.4, seed: int = 0) -> list[dict]:
    """Mix harmful and benign prompts (each {prompt, lang}) into one tagged, shuffled dataset, deterministically."""
    rng = random.Random(seed)
    n_benign = round((len(harmful) / (1 - benign_frac)) * benign_frac) if benign_frac < 1 else len(benign)
    rows = [{"prompt": r["prompt"], "lang": r.get("lang", "en"), "kind": "harmful"} for r in harmful]
    rows += [{"prompt": r["prompt"], "lang": r.get("lang", "en"), "kind": "benign"}
             for r in rng.sample(benign, min(n_benign, len(benign)))]
    rng.shuffle(rows)
    return rows
