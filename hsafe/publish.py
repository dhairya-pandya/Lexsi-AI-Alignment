"""Release helpers: redaction of judged-unsafe completions, and pushes to the Hugging Face Hub."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

REDACTED = "[REDACTED: judged unsafe]"


def redact(judged, gens) -> list[dict]:
    """Rows for the public tier: unsafe completions replaced by REDACTED, every label kept."""
    by_key = {g.rendered_key: g for g in gens}
    rows = []
    for j in judged:
        g = by_key[j.generation_key]
        row = {"generation_key": j.generation_key, **asdict(g), "label": j.label, "judge": j.judge}
        if j.label == "unsafe":
            row["output"] = REDACTED
        rows.append(row)
    return rows


def redact_public(judged, gens, records) -> list[dict]:
    """Public tier: redact any completion to a harmful prompt that is not a clear refusal (robust to judge
    errors), plus anything judged unsafe. Benign answers and refusals stay in cleartext; all labels kept."""
    from .signals import refusal_prefix_match
    kind = {r.id: r.kind for r in records}
    by_key = {g.rendered_key: g for g in gens}
    # map generation key -> record id via rendered key prefix is not available here; records align to gens order
    rec_by_key = {g.rendered_key: r for g, r in zip(gens, records)}
    rows = []
    for j in judged:
        g = by_key[j.generation_key]
        r = rec_by_key.get(j.generation_key)
        row = {"generation_key": j.generation_key, **asdict(g), "label": j.label, "judge": j.judge}
        harmful_noncompliant = r is not None and r.kind == "harmful" and j.label != "refusal" \
            and not refusal_prefix_match(g.output)
        if j.label == "unsafe" or harmful_noncompliant:
            row["output"] = REDACTED
        rows.append(row)
    return rows


def push_dataset(rows: list[dict], repo_id: str, private: bool = True, gated: bool = False) -> str:
    """Create (or update) a dataset repo. Start private; flip visibility only at release."""
    from datasets import Dataset
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    if gated:
        api.update_repo_settings(repo_id, repo_type="dataset", gated="auto")
    Dataset.from_list(rows).push_to_hub(repo_id, private=private)
    return f"https://huggingface.co/datasets/{repo_id}"


def push_vector(path: str | Path, repo_id: str, card: str, private: bool = True) -> str:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=Path(path).name, repo_id=repo_id)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"
