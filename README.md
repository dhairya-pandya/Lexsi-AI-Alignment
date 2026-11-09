# Where does Tiny Aya's safety live?

Lexsi AI Safety & Alignment Hackathon, Track 2. Design: `docs/superpowers/specs/2026-11-10-where-safety-lives-design.md`; plan: `docs/superpowers/plans/2026-11-10-where-safety-lives.md`.

## Setup
```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[dev]'          # CPU: tests, data build, analysis
uv pip install -e '.[gpu]'          # GPU boxes: vLLM, SafeTune, AuditKIT, CuratorKIT
export HF_TOKEN=...                 # account with access to the gated CohereLabs models
pytest                              # fast tests (tiny random cohere2 model)
pytest -m slow                      # needs HF_TOKEN: real tokenizer checks
```

## Reproduce
| Step | Command / notebook | Hardware |
|---|---|---|
| Data | `python -m hsafe.data.build` | CPU |
| Gates G1/G2 | `notebooks/e0_gates.ipynb` (`python -m hsafe.run --config configs/gate_g1.yaml`, `gate_g2.yaml`, then `python -m hsafe.gates results/gate_g1 results/gate_g2`) | Kaggle T4 |
| E1 | `notebooks/e1.ipynb` (`configs/e1_*.yaml`) | T4; Expanse on L4 |
| E2 | `notebooks/e2.ipynb` (`configs/e2_*.yaml`) | L4 (vLLM) |
| E3 | `notebooks/e3.ipynb`; circuit: `circuit/README.md` | L4 |
| E4 | `notebooks/e4.ipynb` | L4 |
| E5 | `notebooks/e5.ipynb` (`configs/e5_*.yaml`) | T4 |
| S1 | `notebooks/s1.ipynb` | T4 |
| Judge validation | `notebooks/labels.ipynb` | CPU + humans |
| Demo | `space/app.py` (ZeroGPU) | — |

Licenses: `DATA_LICENSES.md`. Models are CC-BY-NC-4.0; derived artifacts are non-commercial.
