"""Preamble buckets for harmful prompts, and the hard exclusion of anything involving minors."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from ..records import PromptRecord

BUCKET_MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "bucket_map.yaml"

# Conservative: anything that mentions minors in a harmful prompt is dropped, together with
# any source category about child sexual content. False positives only shrink the dataset.
_MINOR_TEXT = re.compile(
    r"\b(child|children|childs?|minors?|underage|under-age|teens?|teenagers?|kids?|preteens?|"
    r"schoolgirls?|schoolboys?|csam|loli\w*|pedo\w*|paedo\w*|toddlers?|infants?|bab(y|ies)|"
    r"young (girl|boy)s?|\d{1,2}[- ]?(year|yr)s?[- ]?old)\b",
    re.IGNORECASE,
)
_MINOR_CATEGORY = re.compile(r"child|minor|csam|exploitation", re.IGNORECASE)


def exclude_minors(records: list[PromptRecord]) -> list[PromptRecord]:
    return [r for r in records if not (_MINOR_TEXT.search(r.text) or _MINOR_CATEGORY.search(r.category))]


def load_bucket_map(path: str | Path = BUCKET_MAP_PATH) -> dict[str, str]:
    return yaml.safe_load(Path(path).read_text())


def assign_bucket(record: PromptRecord, bucket_map: dict[str, str]) -> str:
    return bucket_map[f"{record.source}:{record.category}"]
