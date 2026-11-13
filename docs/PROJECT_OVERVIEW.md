# Project Overview: Cross-Lingual Tamper Resistance of Fine-Tuning-Stage Safety Defenses

**Venue:** Lexsi AI Safety & Alignment Hackathon · **Deadline:** 2026-11-14 AoE
**Model:** `CohereLabs/tiny-aya-fire` (Tiny Aya Fire, South-Asian variant; cohere2 arch, 3.35B, gated)
**Toolkits:** Lexsi SafeTune (defenses), AlignTune (GRPO), CircuitKIT/nnsight (mechanism)
**Status (2026-11-11):** core finding locked; **a working defense landed**: low-rank dW projection closes the gap (Exp A, full k=0,1,2,4,8 Pareto). Training-time anchoring (Exp B) tested, **negative** (prevention does not help; repair wins).

---

## 1. What the project is about

Fine-tuning-stage safety defenses (the methods a model provider applies while running a customer's fine-tune to stop the fine-tune from removing safety) are **built and evaluated in English**. We ask a question nobody in the literature had tested:

> **Do English-trained fine-tuning defenses still protect the model when the attacker's contaminated fine-tuning data is in Hindi or Bengali, and do they protect *behaviour in those languages*?**

This sits in the gap between two prior papers:
- **Poppi et al. (2410.18210):** fine-tuning attacks transfer across languages, but they test **no defenses**.
- **TamperBench (2602.06911):** has a single, undiscussed non-English (French) row.

No prior work evaluates Vaccine / Booster / Lisa / RepNoise under a **non-English attack with per-language evaluation**. That is our contribution.

### Threat model
A fine-tuning-as-a-service provider. The customer uploads contaminated data in language *L* (benign carrier + a harmful fraction *p* = 0.2). The provider fine-tunes with defense *D*, using **English** safety (refusal) data. We ask whether the resulting model is safe when probed **in language L**.

- Defenses *D* ∈ {`PlainSFT` (none), `Lisa`, `Booster`, `Vaccine`, `RepNoise`}, all SafeTune defaults.
- Attack languages *L* ∈ {en, hi, bn}. Harmful/benign pairs machine-translated with **NLLB-200-distilled-600M**.
- English safety set is **never translated**; that is the whole point.

### Evaluation
- **Harmful (en):** StrongREJECT (n≈208), scored by **HarmBench-Mistral-7b** classifier.
- **Harmful (multilingual):** MultiJail en + bn (60/bucket), scored by **Qwen3Guard-Gen-4B**.
- **Over-refusal:** XSTest + OR-Bench (n≈280).
- **Capability:** MMLU (n≈143).
- Prompt-level **bootstrap CIs** on every harm number; a **degeneration gate** (`is_degenerate`) runs beside every score so a collapsed model is not mis-scored "safe."

---

## 2. What has been done

### Pipeline / infrastructure (`hsafe/` package)
- **Core eval:** `records.py`, `render.py` (exact-distance prompt renderer), `gen.py` (vLLM + HF backends), `judge.py` (HarmBench + Qwen3Guard, degeneration check), `run.py` (config-driven runner, accepts a checkpoint path as the model), `stats.py` (bootstrap CIs), `gates.py`, `publish.py` (redaction + Hub push).
- **Tamper subpackage (`hsafe/tamper/`):** `data.py` (attack rows + NLLB translation), `cell.py` (one defense×language cell; two-stage defend→attack protocol), `safety_data.py` (real refusals + benign twins), `harden_rl.py` (GRPO safety reward), `analysis.py` (RDE tables, figures), **`project_dw.py` (new, low-rank update projection defense).**
- **Interp subpackage (`hsafe/interp/`):** `hooks.py`, `directions.py` (refusal direction), `attention.py`, `mechanism.py` (per-language directions, direction drift, dW spectral), `steer.py`.
- **Compute:** HF Jobs via `scripts/hf_job.sh` (downloads code + results repos, installs SafeTune, runs, uploads on exit). `scripts/pull_results.py` reliably pulls results. Full local test suite: **141 passing**.

### Experiments run
| ID | Experiment | Status |
|----|-----------|--------|
| E1 | Preamble-removal / "where safety lives" (original idea) | Done, **negative** (safety is in the weights, not the 359-token preamble) |
| Main | 5 defenses × 3 attack languages, per-language eval | Done |
| T1a | Benign-twin anchor sweep (over-refusal fix) | Done |
| T1b | Bengali safety-data dose-response (0/25/50%) | Done |
| T2 | Mechanism: refusal-direction drift + dW spectral | Done |
| GRPO | Defensive GRPO pilot (stay-safe RL) | Done, null (fixable) |
| **A** | **Low-rank dW projection (tamper-repair)** | **Done, full k=0,1,2,4,8 Pareto; closes the gap** |
| **B** | **Union-subspace anchoring (tamper-prevention)** | **Done, negative (no gain over English-only; union variant worse)** |

### Accounts / budget
- `dhairyapandya`: gated + was funded; **spent** this session (Exp A + early Exp B). Code/results: `dhairyapandya/hsafe-code`, `dhairyapandya/hsafe-results`.
- `codingpanda06`: gated + **funded**; ran Exp B to completion. Code/results mirrored to `codingpanda06/hsafe-code`, `codingpanda06/hsafe-results` (personal private repos can't be shared as collaborators, so B ran fully on its own repos).
- `runtime-terror`: gated, no credits (402); `dp2406342`: no credits; `ByteKnight28`: spent.
- No HF API exposes credit balance (402 fires free, before run, used as a safe probe). Default HF Jobs wall-limit is 30 min: **always pass `--timeout`** (a 30-min cut is what truncated the first B upload).

---

## 3. Results

### 3.1 Headline: the gap is real and language-local
English-trained defenses, attacked in Bengali, leave English safe but **Bengali wide open**:
- After a Bengali attack: defended Tiny Aya Fire answers **0–4% of harmful English** prompts but **59–61% of harmful Bengali** prompts.
- An **English-only evaluation wrongly certifies the model as robust.** This is the core, novel result.

### 3.2 Secondary findings
- **SafeTune data-quality artifact:** the default English "refusal aux" set is **~92% compliant answers** (only ~8% real refusals); hardening on it degrades pre-attack safety. Novel; worth a bug report to Lexsi.
- **Benign twins (T1a):** fix over-refusal collapse (100% → 28%) and capability (MMLU 0 → 0.51) but **do not** close the language gap. Over-refusal and the gap are two independent problems.
- **Target-language data (T1b):** Bengali safety data 0/25/50% → Bengali harm 0.57 → 0.45 → 0.46 (overlapping CIs), while English safety *worsens* (unsafe-en 0.06 → 0.18) and over-refusal improves (0.22 → 0.09). **Adding target-language data barely dents the gap** → the attack *overwrites* the added safety. This is a **tamper-resistance** problem, not a data-coverage problem.
- **Mechanism (T2):** refusal-direction cosine is a weak transfer proxy (bn attack drifts the bn direction to cos≈0.768 vs en≈0.805; en attack barely moves either). The actionable signal is that **dW = W_post − W_base is low-rank**: the **top-2 singular directions hold ≈50% of the energy.**
- **GRPO pilot:** null: Tiny Aya's inconsistent stop token meant TRL ran to max length, and a binary reward gave zero gradient. Fixable (eos=[6,3] + graded Qwen3Guard reward).

### 3.3 NEW headline, a working defense: low-rank dW projection (Experiment A)
Motivated directly by the mechanism finding (dW is low-rank). The attack's weight change is concentrated in a few singular directions; remove the **top-k** of them per weight matrix and keep the rest:
`W_repaired = W_base + (dW − top-k SVD component of dW)`. Post-hoc, training-free, ~1 GPU-hour.

Bengali attack on Tiny Aya Fire (1 seed):

| k (dirs removed) | en harm (StrongREJECT) | **bn harm (MultiJail)** | en harm (MJ) | over-refusal | MMLU |
|---|---|---|---|---|---|
| **0** (attacked, control) | 0.19 [.14,.24] | **0.50 [.42,.58]** | 0.30 | 0.05 | 0.55 |
| **1** | 0.02 [.00,.05] | **0.14 [.09,.21]** | 0.08 | 0.16 | 0.56 |
| **2** | 0.02 [.00,.05] | **0.14 [.08,.21]** | 0.07 | 0.16 | 0.58 |
| **4** | 0.01 [.00,.02] | 0.11 [.06,.17] | 0.08 | 0.18 | 0.52 |
| **8** | 0.01 [.00,.02] | 0.08 [.04,.13] | 0.08 | 0.20 | 0.50 |

Figure: `results/proj/pareto.png` (Bengali harm vs over-refusal vs MMLU across k).

**What it shows:**
- Removing just the **single top direction (k=1)** already closes the gap: Bengali harm **50% → 14%** (CIs non-overlapping) and English harm 19% → 2%, **while MMLU is preserved** (0.55 → 0.56). The harmful write is **nearly rank-1** and concentrated; the benign task gain is diffuse and survives.
- k=1–2 is the **knee** of the curve, matching the mechanism ("top singular directions hold most of the dW energy"). k=4,8 trade a little more Bengali harm (down to 8%) for rising over-refusal and falling MMLU.
- Not a degeneration artifact: MMLU and coherence hold at every k.

**Caveats:** 1 seed, 1 model; over-refusal rises 5% → 16% at k=2 (the benign-twin fix addresses this); the k=0 control here is an undefended bn-attack (0.50), slightly below the defended-model headline (0.59–0.61).

### 3.4 Experiment B, union-subspace anchoring (tamper-prevention): NEGATIVE
The training-time counterpart to A: instead of repairing the attack afterward, try to stop it landing. A ProCon / C-ΔΘ projection constraint (2602.04521), made single-model so it fits one GPU: before the fine-tune, measure the base model's mean projection of harmful prompts onto each refusal direction (the "refusal budget"); during the bn fine-tune, add a hinge penalty whenever a harmful/refusal example's last-token hidden state (unit-normalized, layer 16) projects **below** that budget along any anchored direction. Three arms, one balanced English safety set (50% real refusals + benign twins), 1 seed:

| arm | en harm (StrongREJECT) | **bn harm (MultiJail)** | en harm (MJ) | over-refusal | MMLU |
|---|---|---|---|---|---|
| **none** (λ=0 control) | 0.00 | **0.22 [.15,.29]** | 0.00 | 0.66 | 0.49 |
| **en** (anchor span{r_en}) | 0.00 | **0.22 [.15,.29]** | 0.00 | 0.60 | 0.50 |
| **union** (anchor span{r_en,r_bn,r_hi}) | 0.01 | **0.33 [.25,.42]** | 0.00 | 0.60 | 0.50 |

Figure: `results/anchor/bars.png` (Bengali harm by arm with CIs).

**What it shows (the defense does not work):**
- **English-direction anchoring does nothing for Bengali:** en-arm bn harm (0.22) is *identical* to the λ=0 control (0.22).
- **Union anchoring is worse, not better:** adding the bn/hi directions pushes bn harm *up* to 0.33 (CIs barely overlap the control), the opposite of the hypothesis. Over-constraining along noisy (machine-translated) non-English refusal directions appears to fight the task objective and leaves the model in a worse place.
- English stays safe and MMLU is preserved in every arm, so the failure is specific to the cross-lingual harm the anchor was meant to stop.
- Side observation (confounded, not a headline): mixing a *balanced* English safety set (real refusals + benign twins) directly into the contaminated fine-tune already pulls undefended bn harm to 0.22, lower than the two-stage SafeTune defenses' 0.59–0.61, but this is a different recipe and over-refusal is still 66%.

**Takeaway: repair beats prevention.** The attack's write is low-rank and cheaply removed *after* the fact (A); trying to *prevent* it with a representation anchor at train time did not close the gap and the union variant hurt. A remains the project's working defense.

**Caveats:** 1 seed, 1 model, single layer (16) and λ (1.0), no λ/layer sweep; directions for bn/hi come from machine-translated attack prompts (a known weak point, and a plausible cause of the union regression).

---

## 4. Conclusions

1. **English-only evaluation of tamper defenses is unsafe.** Robustness must be measured **per language**; an English-only eval certifies a model that is wide open in Bengali.
2. **The cross-lingual gap is a tamper-resistance problem, not a coverage problem.** Adding target-language safety data does not close it, because the fine-tune attack overwrites the added safety.
3. **But the damage is low-rank and repairable.** The attack's weight change concentrates in ~2 singular directions. Projecting them out **closes most of the gap (bn 50% → 14%) while preserving capability**: a cheap, training-free, post-hoc defense that the provider can apply after any fine-tune. This is the project's strongest, most actionable contribution and turns a negative result ("nothing closed the gap") into a positive one.
4. **Repair beats prevention.** The natural training-time counterpart, anchoring the hidden state to the union of per-language refusal directions during the fine-tune (Exp B), **does not close the gap**: English-direction anchoring matches the no-anchor control, and anchoring all three languages is *worse*. So the actionable defense is post-hoc repair of the low-rank write (A), not train-time prevention (B).

---

## 5. Work ahead

### Immediate
- **Pareto done** (k=0,1,2,4,8); figure at `results/proj/pareto.png`.
- **Write Experiment A into `article/hf_article.md`** (table + figure + rank-1 framing).

### Strengthening the defense result
- **Experiment B, union-subspace anchoring: DONE, negative** (§3.4). Prevention did not close the gap; union anchoring hurt. If revisited: λ/layer sweep, and better non-English refusal directions (native prompts, not machine-translated); the translated directions are the likeliest cause of the union regression.
- **Second seed** on the k=2 (A) headline (CI on the delta across fine-tune seeds).
- **Second model family**: rerun the headline cell + projection on **Llama-3.1-8B-Instruct** (direct Poppi comparability) and/or Aya Expanse 8B. ~$15.

### Validity (Tier 0, makes the headline trustworthy)
- **Judge audit (T0a):** Bengali guards are unreliable; have 2 native Bengali speakers blind-label ~300 stratified responses, report kappa for Qwen3Guard, add native Bengali prompts (BanglaSafe, MultiJail native bn). *(Needs native speakers; I can prepare the export only.)*
- **Control matrix + seeds (T0b):** {defended, undefended} × {en,bn attack} × {en,bn eval}, ≥3 seeds, report a locality ratio.

### Publication / submission
- **HF dataset collection:** public redacted tier + gated companion (`publish.py` ready).
- **Defended-checkpoint / projection artifact** release (optional; ~$2–3).
- **Article finalization:** methods, licenses (NLLB CC-BY-NC, SafeTune), team tags.
- **Repo public + merge `feat/hsafe` → `main` via PR.** *(User runs: `gh repo edit <repo> --visibility public --accept-visibility-change-consequences`.)*
- **File the SafeTune refusal-aux data-quality issue** to Lexsi.

### Deferred / optional
- Fixed GRPO (eos=[6,3] + graded reward + language-ID gate).
- MPO / multilingual DPO stronger starting point.
- Stronger/adaptive attack (romanized/code-mixed Bengali; abliteration/prefill).
- CircuitKIT per-language refusal circuits (EAP-IG).

---

## 6. Key references
Poppi 2410.18210 · TamperBench 2602.06911 · Qwen3Guard 2510.14276 · WildJailbreak 2406.18510 · FalseReject 2505.08054 · Deng 2310.06474 · Lee 2610.00320 (low-rank safety removal, basis of Experiment A) · C-ΔΘ 2602.04521 (subspace anchoring, basis of Experiment B) · BanglaSafe 2608.22335

## 7. Where things live
- Design + plans: `docs/superpowers/specs/`, `docs/superpowers/plans/`
- Research + defense plan: `docs/research/`
- Article draft: `article/hf_article.md`
- Code: `hsafe/`; tests: `tests/` (141 passing)
- Results: `results/` (local) + `dhairyapandya/hsafe-results` (HF dataset, private)
