# Cross-lingual Tamper Resistance: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** run 5 SafeTune defenses × 3 attack languages (en, hi, bn) on tiny-aya-fire and report harm, relative defense efficacy, over-refusal and capability.

**Architecture:**
- A new subpackage `hsafe/tamper/` handles:
  - building the attack data, translated with NLLB;
  - running one (defense, language) cell, which calls a SafeTune harden trainer and writes a merged checkpoint;
  - analysis.
- Evaluation reuses `hsafe.run`, with `model` set to a checkpoint path.
- GPU work runs as HF Jobs through `scripts/hf_job.sh`.

**Spec:** [docs/superpowers/specs/2026-11-11-tamper-resistance-design.md](../specs/2026-11-11-tamper-resistance-design.md)

## Global Constraints

- Model: `tiny-aya-fire` at the revision in `configs/models.yaml`.
- Defenses: `PlainSFTTrainer`, `LisaTrainer`, `BoosterTrainer`, `VaccineTrainer`, `RepNoiseTrainer`, all at SafeTune defaults with `epochs=1`, `batch_size=4`, `lr=1e-4`.
- Attack data: n = 500, harmful_frac = 0.2, seed 0. Languages are exactly `en`, `hi`, `bn`. NLLB codes are `eng_Latn`, `hin_Deva`, `ben_Beng`.
- Safety (defense) data is always SafeTune's English refusal aux set.
- No fine-tuned weights are published. Translated harmful data and generations go to the private results repo only.
- Commit messages carry no attribution lines.

## Review Focus

1. **Translation drops or reorders rows.** Harmful and benign labels would then be misaligned. Pinned by `test_translate_preserves_order_and_kind`.
2. **The harmful fraction drifts from 0.2.** Pinned by `test_attack_rows_fraction`.
3. **An evaluation runs on the base model instead of the fine-tuned checkpoint.** Pinned by `test_runconfig_model_path`.
4. **A cell's results overwrite another cell's.** Run names must embed the defense and the language. Pinned by `test_cell_names_unique`.

---

### Task 1: Runner accepts a checkpoint path as the model
**Files:** modify `hsafe/run.py` (`main`). Test: `tests/test_run.py`.
- [ ] Write `test_runconfig_model_path`. `_resolve_model("results/ckpt/x", models)` returns `(path, None, "tiny_aya")` when the path is not a key of `configs/models.yaml`, and returns `(id, revision, family)` for known keys.
- [ ] Run it and watch it FAIL. Implement `_resolve_model(name, models) -> tuple[str, str | None, str]` and use it in `main`. Run it and watch it PASS. Commit.

### Task 2: Attack data and translation
**Files:** `hsafe/tamper/__init__.py`, `hsafe/tamper/data.py`. Test: `tests/test_tamper.py`.
- [ ] Write the tests:
  - `test_attack_rows_fraction`: `attack_rows(harm, benign, n_total=500, harmful_frac=0.2, seed=0)` has 100 harmful and 400 benign rows, interleaved, and is deterministic.
  - `test_translate_preserves_order_and_kind`: `translate_rows(rows, fn)` with a fake `fn` returns the same kinds in the same order, and the prompts and responses are mapped.
- [ ] Implement:
  - `attack_rows(...) -> list[dict(prompt, response, kind)]`;
  - `translate_rows(rows, translate_fn) -> list[dict]`;
  - `nllb_translator(src, tgt, batch_size=16) -> Callable[[list[str]], list[str]]`, lazy and GPU-only;
  - `build(out_dir)` CLI: `python -m hsafe.tamper.data --out data/tamper`, which writes `{en,hi,bn}.jsonl`.
- [ ] Watch the tests FAIL, then PASS. Commit.

### Task 3: Cell runner (defense × language)
**Files:** `hsafe/tamper/cell.py`. Test: `tests/test_tamper.py`.
- [ ] Write the tests:
  - `test_cell_names_unique`: `cell_name(d, l)` is unique over the grid and matches `tamper_<defense>_<lang>`.
  - `test_defense_registry`: maps exactly the 5 names to SafeTune trainer classes.
  - `test_cell_smoke_tiny` (marked slow? no, CPU): PlainSFT on the tiny cohere2 fixture with 8 rows writes a checkpoint dir that `AutoModelForCausalLM` can load.
- [ ] Implement:
  - `DEFENSES: dict[str, str]`;
  - `cell_name(defense, lang) -> str`;
  - `train_cell(defense, lang, model, tok, rows, out_dir, safety_ds) -> str`, which tokenizes with SafeTune `_tokenize_qa_rows` and calls `trainer.train(train_ds, out_dir, safety_dataset=...)`;
  - `main()` CLI: `python -m hsafe.tamper.cell --defense lisa --lang hi`. It trains, writes `configs/<cell>.yaml` (eval config, model = checkpoint path) and runs `hsafe.run`.
- [ ] Watch the tests FAIL, then PASS. Commit.

### Task 4: Runs (HF Jobs)
- [ ] Data job: one T4 job runs `python -m hsafe.tamper.data` and uploads `data/tamper/` to the private results repo.
- [ ] English pilot: 5 cells for `en`, one L4 job. Check that at least one defense lowers harm against `plain` (paired CI). If none does, rerun with harmful_frac 0.1 and record the decision.
- [ ] Then the `hi` and `bn` columns, one L4 job each, in parallel.

### Task 5: Analysis
**Files:** `hsafe/tamper/analysis.py`. Test: `tests/test_tamper.py`.
- [ ] Write the test `test_rde`. With H(none) = 0.5 and H(D) = 0.25, RDE = 0.5. With H(none) = 0, RDE = NaN.
- [ ] Implement:
  - `rde_table(results_root) -> DataFrame` with columns defense, lang, harm, harm_ci, rde, over_refusal, accuracy;
  - `figure_rde(df, path)`, a defense × language heatmap.
- [ ] Watch the tests FAIL, then PASS. Commit.
