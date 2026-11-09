"""S1 side bet: under-trained tokens (Cohere Magikarp report) inserted just before the request."""
from __future__ import annotations

import random
import re
from dataclasses import replace
from pathlib import Path

from .records import Rendered

_SECTION = "## Under-trained token verification results"
_ROW = re.compile(r"^\|\s*(\d+)\s*\|[^|]*\|[^|]*\|([^|]*)\|")
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def load_magikarp(path: str | Path, max_prob: float = 0.01) -> list[int]:
    """Token ids verified as under-trained: the report's verification section, max_prob below p = 0.01."""
    ids, inside = [], False
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "):
            inside = line.strip() == _SECTION
            continue
        m = _ROW.match(line) if inside else None
        if m:
            nums = _NUM.findall(re.sub(r"rgb\([^)]*\)", "", m.group(2)))
            if nums and float(nums[-1]) < max_prob:
                ids.append(int(m.group(1)))
    return ids


def reachable(ids: list[int], tok) -> list[int]:
    """Ids that normal tokenization of their own decoded string reproduces exactly (a user can type them)."""
    return [i for i in ids if tok(tok.decode([i]), add_special_tokens=False)["input_ids"] == [i]]


def insert_tokens(rendered: Rendered, ids: list[int], k: int, seed: int, tok) -> Rendered:
    """Insert k ids sampled from `ids` immediately before the request; spans are shifted accordingly."""
    rng = random.Random(seed)
    picks = [rng.choice(ids) for _ in range(k)]
    q0, q1 = rendered.request_span
    new_ids = rendered.input_ids[:q0] + picks + rendered.input_ids[q0:]
    return replace(rendered, condition=f"{rendered.condition}+glitch{k}", input_ids=new_ids,
                   text=tok.decode(new_ids), request_span=(q0 + k, q1 + k))
