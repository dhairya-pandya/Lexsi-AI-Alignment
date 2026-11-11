"""Assemble and push the two-tier evaluation dataset for release.

Public tier: completions to harmful prompts that are not clear refusals are redacted (robust to judge error);
benign answers and refusals stay in cleartext. Gated tier: full completions. Both start PRIVATE; flip visibility
only at release.

    python scripts/publish_datasets.py --owner dhairyapandya [--public-only] [--flip-public]
"""
from __future__ import annotations

import argparse
import glob

from hsafe.publish import assemble_generation_dataset, push_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="dhairyapandya")
    ap.add_argument("--base", default="tiny-aya-tamper-evals", help="dataset repo basename")
    ap.add_argument("--glob", default="results/tamper_*", help="run dirs to include")
    args = ap.parse_args()

    runs = sorted(d for d in glob.glob(args.glob) if glob.os.path.isdir(d))
    prompt_files = sorted(glob.glob("data/prompts/*.jsonl"))
    public, gated = assemble_generation_dataset(runs, prompt_files)
    print(f"{len(runs)} runs -> {len(public)} public rows, {len(gated)} gated rows")

    pub_url = push_dataset(public, f"{args.owner}/{args.base}", private=True)
    gat_url = push_dataset(gated, f"{args.owner}/{args.base}-gated", private=True, gated=True)
    print("PUBLIC (private for now):", pub_url)
    print("GATED  (private for now):", gat_url)


if __name__ == "__main__":
    main()
