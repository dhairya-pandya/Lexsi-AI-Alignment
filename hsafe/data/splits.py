"""fit/eval assignment grouped by base_id, stratified by (source, bucket)."""
from __future__ import annotations

import random
from dataclasses import replace

from ..records import PromptRecord


def assign_splits(records: list[PromptRecord], fit_frac: float = 0.3, seed: int = 0) -> list[PromptRecord]:
    strata: dict[tuple, list[str]] = {}
    for r in records:
        key = (r.source, r.bucket)
        bases = strata.setdefault(key, [])
        if r.base_id not in bases:
            bases.append(r.base_id)
    fit: set[str] = set()
    rng = random.Random(seed)
    for key in sorted(strata):
        bases = sorted(strata[key])
        rng.shuffle(bases)
        fit.update(bases[: round(len(bases) * fit_frac)])
    return [replace(r, split="fit" if r.base_id in fit else "eval") for r in records]
