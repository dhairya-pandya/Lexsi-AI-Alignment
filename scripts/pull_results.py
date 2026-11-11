"""Pull result dirs from the private results repo (the `hf download --include` CLI can silently skip files).

    python scripts/pull_results.py tamper_ e1_        # prefixes under results/
"""
import sys

from huggingface_hub import HfApi, hf_hub_download

REPO = "dhairyapandya/hsafe-results"
prefixes = tuple(f"results/{p}" for p in sys.argv[1:]) or ("results/",)
api = HfApi()
files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith(prefixes)]
for f in files:
    hf_hub_download(REPO, f, repo_type="dataset", local_dir=".")
print(len(files), "files")
