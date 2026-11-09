"""Clean (P-default) vs corrupt (P-none) prompt pairs for CircuitKIT EAP-IG, as CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..render import render


def export_circuit_pairs(records, tok, refusal_answer: str, comply_answer: str, path: str | Path,
                         max_tokens: int = 512) -> int:
    """Answers are single tokens (CircuitKIT requirement): the clean run should prefer refusal_answer,
    the corrupt run comply_answer. Pairs longer than max_tokens are skipped. Returns rows written."""
    bad = [r.id for r in records if r.split != "fit"]
    if bad:
        raise ValueError(f"fit-split records only; got {bad[:5]}")
    for a in (refusal_answer, comply_answer):
        if len(tok(a, add_special_tokens=False)["input_ids"]) != 1:
            raise ValueError(f"answer {a!r} is not a single token")
    rows = []
    for r in records:
        clean = render(r, "P-default", None, None, tok, "tiny_aya")
        corrupt = render(r, "P-none", None, None, tok, "tiny_aya")
        if max(len(clean.input_ids), len(corrupt.input_ids)) > max_tokens:
            continue
        rows.append({"clean_prompt": clean.text, "clean_answer": refusal_answer,
                     "corrupt_prompt": corrupt.text, "corrupt_answer": comply_answer})
    pd.DataFrame(rows, columns=["clean_prompt", "clean_answer", "corrupt_prompt", "corrupt_answer"]).to_csv(path, index=False)
    return len(rows)
