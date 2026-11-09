#!/usr/bin/env bash
# Runs inside an HF Job. Env: HF_TOKEN (secret), CODE_REPO, RESULTS_REPO, RUNS (space-separated config names),
# optional POST (shell command run after the configs), DTYPE (override config dtype, e.g. bfloat16 on L4+).
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
pip install -q -U "huggingface_hub[cli]" >/dev/null
hf download "$CODE_REPO" --local-dir /code >/dev/null
cd /code
pip install -q -e . >/dev/null
if [ -n "${RESULTS_REPO:-}" ]; then  # resume: pull earlier results (openers, caches)
  hf download "$RESULTS_REPO" --repo-type dataset --local-dir /code >/dev/null 2>&1 || true
fi
upload() { hf upload "$RESULTS_REPO" results results --repo-type dataset --commit-message "results: ${RUNS:-post}" >/dev/null || true; }
trap upload EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv
for run in ${RUNS:-}; do
  cfg="configs/$run.yaml"
  if [ -n "${DTYPE:-}" ]; then sed -i "s/^dtype: .*/dtype: $DTYPE/" "$cfg"; fi
  echo "=== $run $(date -u +%H:%M:%S)"
  python -m hsafe.run --config "$cfg"
done
if [ -n "${POST:-}" ]; then echo "=== post"; bash -c "$POST"; fi
