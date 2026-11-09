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
        row = {**asdict(g), "label": j.label, "judge": j.judge}
        if j.label == "unsafe":
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
