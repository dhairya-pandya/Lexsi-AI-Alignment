"""Record types shared by every stage, plus JSONL io."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, TypeVar

KINDS = ("harmful", "benign", "rule_probe", "capability")
BUCKETS = ("ACCEPT", "REJECT", "UNLISTED", "NA")
SPLITS = ("fit", "eval")


@dataclass
class PromptRecord:
    id: str
    base_id: str
    text: str
    lang: str
    source: str
    kind: str
    bucket: str
    category: str
    split: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Rendered:
    record_id: str
    split: str | None
    condition: str
    d: int | None
    filler: str | None
    model: str
    text: str
    input_ids: list[int]
    preamble_span: tuple[int, int] | None
    request_span: tuple[int, int]

    def __post_init__(self):
        # JSON round-trips tuples as lists.
        if self.preamble_span is not None:
            self.preamble_span = tuple(self.preamble_span)
        self.request_span = tuple(self.request_span)


@dataclass
class Generation:
    rendered_key: str
    output: str
    first_token_refusal_prob: float
    gen_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Judged:
    generation_key: str
    label: str  # "unsafe" | "safe" | "refusal" | "degenerate"
    judge: str


T = TypeVar("T")


def write_jsonl(path: str | Path, items: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, cls: type[T]) -> list[T]:
    names = {f.name for f in fields(cls)}
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out.append(cls(**{k: v for k, v in row.items() if k in names}))
    return out
