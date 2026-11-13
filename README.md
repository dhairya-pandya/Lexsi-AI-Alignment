# Does safety survive an attack in another language?

Safety defenses that model providers apply during customer fine-tuning are built and tested almost entirely in English. We asked a simple question nobody had checked: if the attacker's poisoned fine-tuning data is in **Bengali** instead of English, do those defenses still hold — and do they hold *for Bengali*?

They don't.

## What we found

Take Tiny Aya Fire, defend it with the usual SafeTune methods (Lisa, Booster, Vaccine, RepNoise), then run a poisoned fine-tune in Bengali. Evaluate in both languages:

- **English looks fine** — the model refuses 96–100% of harmful English prompts.
- **Bengali is wide open** — the same model answers **59–61%** of harmful Bengali prompts.

An English-only evaluation would have signed off on a model that is badly unsafe in Bengali. Adding Bengali safety data barely helps (harm 0.57 → 0.46): the attack *overwrites* the safety you add. It's a tamper-resistance problem, not a coverage gap.

## What fixes it

The attack's weight change is **low-rank** — the top two singular directions of `dW = W_post − W_base` carry about half the energy. So strip them.

Removing just the **single top direction** of each attention/MLP update (`project_dw`, k=1) drops Bengali harm **50% → 14%**, keeps English safe, and preserves MMLU. The harmful write is nearly rank-1 and concentrated; the benign task gain is diffuse and survives the cut. It's post-hoc and training-free — a provider can apply it after the fine-tune without retraining.

We also tried the training-time version — anchor the refusal directions during fine-tuning so the attack can't land (`anchor_subspace`). It didn't work: anchoring on English did nothing for Bengali, and anchoring on the union made it worse. **Repair beats prevention** here.

## Reproduce

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[dev]'          # CPU: tests, data build, analysis
uv pip install -e '.[gpu]'          # GPU: vLLM, SafeTune
export HF_TOKEN=...                 # needs access to the gated CohereLabs models
pytest                              # fast tests (tiny random cohere2 model)
```

| Step | Command | Hardware |
|---|---|---|
| Build attack/safety data | `python -m hsafe.data.build` | CPU |
| The low-rank repair (Exp A) | `python -m hsafe.tamper.project_dw --lang bn --ks 0 1 2 4 8` | A100 |
| Training-time anchoring (Exp B) | `python -m hsafe.tamper.anchor_subspace` | A100 |

GPU runs go through HF Jobs (`scripts/hf_job.sh`); always pass `--timeout` or the 30-min default cuts long runs.

## Read more

- **Full writeup:** [`article/hf_article.md`](article/hf_article.md) · technical detail: [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)
- **Datasets:** [tiny-aya-tamper-evals](https://huggingface.co/datasets/dhairyapandya/tiny-aya-tamper-evals) (public, redacted) · [tiny-aya-tamper-evals-gated](https://huggingface.co/datasets/dhairyapandya/tiny-aya-tamper-evals-gated) (gated, full)
- **Repaired checkpoint:** [tiny-aya-fire-bn-repaired_k1](https://huggingface.co/codingpanda06/tiny-aya-fire-bn-repaired_k1) (gated)
- **Base model:** [CohereLabs/tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire)

Models are CC-BY-NC-4.0; derived artifacts are non-commercial. See [`DATA_LICENSES.md`](DATA_LICENSES.md).
