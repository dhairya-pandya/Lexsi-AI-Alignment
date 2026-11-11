"""Benign filler that pushes the request a chosen number of tokens away from the preamble."""
from __future__ import annotations

import random

FILLER_KINDS = ("F1", "F2", "F3")


def _take_tokens(text_pool: list[str], n_tokens: int, tok, rng: random.Random) -> str:
    """Concatenate shuffled passages and cut to n_tokens tokens."""
    if n_tokens <= 0:
        return ""
    order = text_pool[:]
    rng.shuffle(order)
    ids: list[int] = []
    i = 0
    while len(ids) < n_tokens:
        ids += tok(order[i % len(order)] + " ", add_special_tokens=False)["input_ids"]
        i += 1
    return tok.decode(ids[:n_tokens])


def make_filler(kind: str, n_tokens: int, tok, seed: int, pool) -> list[dict] | str:
    """F1 -> chat turns (list of {role, content}); F2 -> prose string; F3 -> [doc turn, ack turn].

    n_tokens is a target for the variable-length content; render() corrects it to an exact distance.
    """
    rng = random.Random(seed)
    if kind == "F2":
        return _take_tokens(pool, n_tokens, tok, rng)
    if kind == "F3":
        doc = _take_tokens(pool, n_tokens, tok, rng)
        return [{"role": "user", "content": "Please read this document carefully:\n\n" + doc},
                {"role": "assistant", "content": "I have read the document."}]
    if kind == "F1":
        turns = pool[:]
        rng.shuffle(turns)
        out: list[dict] = []
        used = 0
        for t in turns * (1 + n_tokens // 8):
            u = len(tok(t["user"], add_special_tokens=False)["input_ids"])
            a = len(tok(t["assistant"], add_special_tokens=False)["input_ids"])
            if used + u + a + 16 > n_tokens:
                break
            out += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["assistant"]}]
            used += u + a + 16  # ~turn-marker overhead per exchange
        # Variable tail so the total can be tuned token by token.
        tail = _take_tokens([t["assistant"] for t in pool], max(n_tokens - used, 1), tok, rng)
        out += [{"role": "user", "content": "Tell me more."}, {"role": "assistant", "content": tail}]
        return out
    raise ValueError(f"unknown filler kind {kind!r}")
