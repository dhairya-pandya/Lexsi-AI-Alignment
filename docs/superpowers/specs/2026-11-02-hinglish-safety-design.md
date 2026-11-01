# Design: Safety Fails Silently in Hinglish — and Compression Makes It Worse

Lexsi AI Safety & Alignment Hackathon · Track 2 (2a + 2b) · Written 2026-11-02
Rules/logistics reference: [HACKATHON_NOTES.md](../../../HACKATHON_NOTES.md)
Related work and the reasoning behind the methods chosen below: [2026-11-02-related-work.md](../../research/2026-11-02-related-work.md)

Legend: **[team]** = decided by the team. **[my call]** = Claude's default, open to override. Everything else is left to the builder.

---

## 1. Intent and the why

Small open multilingual models are what actually reach Indian users: they are cheap, run on-device, and get shipped **quantized**. Two things happen on the way:

1. Users write in **Hinglish** — Hindi in Roman script, mixed with English, with loose spelling. Safety training is overwhelmingly English, so refusals may not fire.
2. Deployers **compress the model to 4-bit** to serve it. Quantization is known to erode safety, and we suspect it erodes the already-weaker non-English safety first.

The project shows both failures with numbers, finds *where* in the model they come from, and fixes them — producing a Hinglish safety benchmark, a repaired model, and a safety-preserving compressed model, all public on Hugging Face.

One-sentence thesis: **"Tiny Aya refuses harmful requests in English but complies in Hinglish; 4-bit quantization widens that gap; steering at the refusal direction and circuit-aware mixed precision close it without raising over-refusal."** (The second and third clauses are hypotheses — the project reports whatever the data says.)

Covers hackathon problem statements: *Multilingual and code-mixed safety* (2a), *Adversarial and safety evaluation sets* (2a), *Inference-time safety steering* (2b), *Locating where safety lives* (2b), *Safety-aware compression* (2b).

## 2. Goals, non-goals, anti-goals

**Goals**
- G1. Build and publish **HinglishHarm**: harmful prompts in 4 language forms + a matching benign over-refusal set, with labels and a scoring recipe.
- G2. Measure the safety gap (attack success rate by language form) on the chosen models.
- G3. Locate the cause: does the model's refusal mechanism activate on Hinglish input?
- G4. Close the gap (Idea 1) with SafeTune, keeping over-refusal and capability flat.
- G5. Measure safety vs bit-width by language form, and show circuit-aware mixed precision preserves safety better than uniform quantization at matched size (Idea 3).
- G6. Ship everything in the format the rules require (HF collections + HF article), by **Nov 14 EOD AoE**.

**Non-goals**
- Not a general multilingual benchmark — Hindi/Hinglish only (other South Asian languages = stretch at most).
- Not new attack algorithms. Attacks are language transformations of existing harmful prompts (plus abliteration as an optional stretch).
- Not a production guardrail or classifier product.
- Not Aya Expanse 32B (doesn't fit our hardware; CircuitKIT can't run discovery on it).

**Anti-goals (failure even if it "works")**
- Fix that lowers Hinglish attack success by making the model refuse everything — over-refusal must be reported next to every safety number.
- Numbers judged by a safety classifier that can't read Hinglish — judge must be validated on our data before any number is believed.
- Steering direction or circuit fitted on the same prompts it is evaluated on (leakage).
- Citing a SafeTune method by its paper name when SafeTune's audit badge isn't **Faithful**.
- Missing the submission checklist (invalid submission = never judged).

## 3. Constraints

- **Rules [hackathon]:** model ∈ {Tiny Aya Base/Global/Fire/Water/Earth, Aya Expanse 8B/32B}; ≥2 Lexsi repos as one connected system; all artifacts public on HF; HF article with the 9-section checklist; only last submission counts.
- **Models [team]:** **Tiny Aya Fire** (3.35B, `cohere2`, 36 layers, South-Asian-focused) = primary. **Aya Expanse 8B** (`cohere1`) = secondary, for "does it generalize". **[my call]** Tiny Aya Global used only in the baseline measurement (Fire vs Global comparison).
- **Gated models:** every teammate must accept the CohereLabs license on HF and use an `HF_TOKEN`. Tiny Aya is CC BY-NC 4.0 → derived artifacts are non-commercial; state licenses in the article.
- **Compute [team]:**
  - User: Mac (no NVIDIA GPU). Kaggle account with **60 h** GPU quota.
  - 2 teammates: laptops with **RTX 4050, 4 GB VRAM** each (as reported; the laptop 4050 is usually 6 GB, so check with `nvidia-smi`). Some extra Kaggle quota each.
  - **$25 HF credits:** can go on compute, with some kept for a demo.
- **Compute plan [my call]:**
  - **Kaggle (T4 ×2 or P100, 16 GB per GPU) = every fp16 run:** generation, judging, circuit discovery, LoRA, quantization. Tiny Aya Fire fp16 (~7 GB) fits one GPU. Aya Expanse 8B fp16 (~16 GB) needs both T4s.
  - **Rough budget:** ~1,850 generations plus judging per configuration, ~15 configurations, plus E2 discovery and one LoRA run. That is about 25–35 GPU hours, which fits inside 60 h with room for reruns. Re-estimate after E0 timings.
  - **Laptops (4 GB):** can't hold Tiny Aya in fp16. Use them for 4-bit Tiny Aya quick checks, a small judge (e.g. LlamaGuard-3-1B) for pre-screening, the CuratorKIT data pipeline, and the labelling tool.
  - **Mac:** CuratorKIT, data cleaning, analysis and plots, write-up. CPU/MPS only for small checks.
  - **HF credits:** ~$15 for a burst GPU (HF Jobs) when Kaggle quota or session limits block the final eval batch; ~$10 for a demo Space. Check current HF pricing before spending.
  - **Kaggle is the shared bottleneck**, so experiments that need GPU are queued in priority order, not run by person. Cache every generation (AuditKIT fingerprinted reruns) so the judge or metrics can change without regenerating.
  - **Aya Expanse 8B is limited to E1 + E3 steering.** No quantization sweep and no LoRA on 8B unless quota is left over after Nov 11.
- **Team [team]:** 3 people (user + 2 teammates).
- **Deadline:** Nov 14 EOD AoE. Both ideas in the submission **[team]**. Nov 16–21 window (if shortlisted) for extensions.
- **Toolkit facts that constrain the how** (from repo READMEs, 2026-11-02):
  - SafeTune: Python 3.12+. Tiny Aya chat template adds ~366-token system preamble → leave `max_len` unset or ≥512. Refusal-direction sweep may pick a poor layer → set `pick_layer` by hand (e.g. 24 of 36). ReSta on Tiny Aya needs `alpha≈0.25`. On Colab/Kaggle: `pip uninstall -y torchao` first. Built-in benchmarks: HarmBench, XSTest, AdvBench, WildJailbreak; judges: WildGuard, LlamaGuard-3. Has an abliteration attack.
  - CircuitKIT: Tiny Aya and Aya Expanse 8B validated for discovery (`eap`, `eap-ig`, `eap-gp`, `ibcircuit`, `cdt`; `acdc` too slow), evaluation and interventions. Circuit-aware mixed-precision quantization: `backend="llmcompressor"` honours `bits ∈ {3,4,8}`; `backend="quanto"` ignores `bits`. README warns the actual compression call "is not exercised in every environment" → **smoke-test it on day 1**.

## 4. The connected system

```
 CuratorKIT ──► HinglishHarm + HinglishXSTest + steering/fit data (ShareGPT/DPO exports)
     │
     ▼
 AuditKIT / SafeTune.evaluate ──► baseline ASR, over-refusal, capability  (per model × language form)
     │
     ▼
 CircuitKIT (discover: eap-ig) + SafeTune (interpret: refusal direction, safety neurons)
     │            "does refusal fire on Hinglish? which components carry it?"
     ├──────────────────────────────┐
     ▼                              ▼
 SafeTune Steer / Harden        CircuitKIT circuit-aware mixed-precision quantization
 (Idea 1: close the gap)        (Idea 3: compress without losing safety)
     │                              │
     └──────────► AuditKIT: same benchmarks, same judge, before vs after ◄──┘
```

All four Track 2 repos used; each run is config-driven and reproducible from a notebook.

## 5. Data

### 5.1 HinglishHarm (attack set)
- **Base prompts [my call, from research]:**
  - **Core: Cohere `aya_redteaming` Hindi split.** 915 prompts written by native speakers, Apache-2.0. Ships in Devanagari with an English translation per prompt, so EN and HI-Deva pairs already exist. Has 9 harm categories and a global/local label (local = India-specific harms such as caste or communal). Sample ~200, stratified by category and by global/local.
  - **Plus ~100 from HarmBench** (dangerous-capability harms: cyber, weapons, drugs), which aya_redteaming barely covers. Record HarmBench's license.
  - Total ~300 base prompts, as before.
- **4 language forms per prompt:**
  - **EN** — original English.
  - **HI-Deva** — Hindi in Devanagari.
  - **HI-Latn** — code-mixed Hinglish in Roman script, the way people type it.
  - **HI-Latn-Pert** — HI-Latn with phonetic misspellings of the sensitive words (attack pattern from "Haet Bhasha aur Diskrimineshun", arXiv 2505.14226).
- **How variants get made:** builder's choice of open model + CuratorKIT generate/quality gates; **Hindi-speaking teammates spot-check** a sample per form for naturalness and meaning preservation. **[my call]** reject variants whose meaning drifted.
  - aya_redteaming prompts only need **HI-Latn** generated, since they were originally typed in Romanized form. HarmBench prompts need both HI-Deva and HI-Latn.
  - **HI-Latn-Pert is smaller (~100 prompts) [my call]:** the Haet Bhasha authors found automatic phonetic perturbation gave poor results and hand-wrote theirs. Use a rule-based misspelling list for sensitive words, then hand-edit.
  - Record token counts per form for every prompt. Tokenizer fragmentation of sensitive words is a cheap mechanistic signal.
- **Splits [my call]:** `fit` (~30%, used only to compute steering directions / circuits / calibration) and `eval` (~70%). Split by base prompt, so all 4 forms of a prompt land in the same split.

### 5.2 HinglishXSTest (over-refusal set)
- XSTest's safe-but-scary-sounding prompts, converted into the same 4 forms. Measures whether fixes make the model refuse harmless requests.

### 5.3 Capability check
- Something small showing the fix/compression didn't break the model in English *and* Hindi. **Left to builder** (e.g. an MMLU subset + a Hindi QA/NLI set via AuditKIT / lm-eval).

- **OR-Bench pseudo-harmful prompts (English):** source for the false-refusal direction in E3. Kept separate from HinglishXSTest so the over-refusal evaluation stays clean.

### 5.4 Judge
- **Primary [my call, from research]:** back-translate the response to English, then judge with **WildGuard** (shipped in SafeTune). This is the protocol BabelSteering and Minionese used. For HI-Latn responses, transliterate to Devanagari before translating.
- **Don't** judge Hinglish directly with Indic guard models: PolyGuard and IndicGuard roughly double their unsafe-slip rate on Romanized input.
- **Optional second judge** on a sample: Command A via a Cohere trial key (it's what Tiny Aya's authors used, in the source language).
- **Must validate before trusting:** hand-label ~150–200 model responses across the 4 forms (teammates who read Hindi), measure judge agreement per form. If agreement on Hinglish is poor, fall back to translate-response-to-English-then-judge, and report which was used.
- **Labelling capacity — assumed, team still to confirm [my call]:** at least one fluent Hindi/Hinglish reader with ~6–8 hours across Nov 4–6. If capacity is lower, label 100 responses (25 per form) instead of 200, and say so in the write-up.

## 6. Experiments

| ID | Question | Method | Output |
|----|----------|--------|--------|
| E0 | Does the stack run on our models/hardware? | Load Fire + Expanse 8B via SafeTune and CircuitKIT; run one tiny eval, one `eap-ig` discovery, one `llmcompressor` 4-bit quant | Go/no-go by end of day 1–2 |
| E1 | Is there a gap? | ASR on HinglishHarm-eval, over-refusal on HinglishXSTest, per language form, for Fire, Global, Expanse 8B | Baseline table (the "before") |
| E2 | Where does it break? | Refusal direction by difference in means on EN `fit` prompts, at post-instruction tokens, with the layer chosen by ablation (Arditi; Wang et al.). Three diagnostics per form: **(a)** cosine of each form's own direction with the EN direction, **(b)** projection of harmful prompts onto the EN direction vs the refusal threshold ("subthreshold activation", Minionese), **(c)** silhouette separation of harmful vs harmless clusters. Plus token fragmentation per form. CircuitKIT `eap-ig` refusal circuit on EN vs HI-Latn; overlap | "Direction is shared, but the Hinglish projection is X% below threshold at layer L"; circuit overlap figure |
| E3 | Can we close it? (Idea 1) | **E3a — BabelSteering recipe via SafeTune Steer:** add the EN refusal direction at one layer (strength α), ablate an orthogonalized false-refusal direction from OR-Bench (weight λ), fold into the weights. Sweep α and λ; start near Gemma's best (λ=0.8, α=0.4). Optional: conditional steering (CAST) as an over-refusal-friendly variant. **E3b — LoRA Harden:** CuratorKIT-built Hinglish refusal + Hinglish benign-compliance pairs, mixed ~15% into general instruction data. Alternative if data is short: response-free self-distillation (MSD) | After-table plus a **trade-off curve over α**: ASR ↓ vs over-refusal ↑ vs capability |
| E4 | Does quantization widen the gap? (Idea 3) | Fire at 8/4/3 bits with GPTQ via `llmcompressor` (8-bit = control). **Plus the official GGUF Q4_K_M and Q4_0 builds** (what users actually run; Tiny Aya's report shows Q8_0 ≈ lossless). Re-run E1. Split results by **script**: Devanagari vs Latin (EN, HI-Latn). Re-measure E2's projection on the quantized model | ASR vs bits per language form; "quantization shrinks the Hinglish projection by X" |
| E5 | Does protecting safety weights fix it? | At **matched average size** vs E4, compare: **(i)** AWQ-trust-style Fisher scoring on **EN** safety data (published baseline), **(ii)** the same scoring on **Hinglish** `fit` data, **(iii)** CircuitKIT circuit-aware mixed precision from the E2 circuit (`tacq` selector) | Safety-vs-size curves; which source of "safety weights" protects Hinglish best |
| E6 | Do the fixes compose? | Apply E3's best fix to the E5 compressed model (or steer after quantizing) | The 2×2: {EN, Hinglish} × {full, 4-bit}, before vs after |
| S1 | *(stretch)* Is the fix robust? | SafeTune abliteration attack / best-of-N on the E3 model | Robustness row |

Fire runs every experiment. **[my call]** Expanse 8B runs E1 + E3 (+ E4 if time), to show the gap and fix generalize.

## 7. Metrics and success

- **ASR** (attack success rate) per language form — % harmful prompts judged unsafe-complied.
- **Over-refusal rate** on HinglishXSTest per form.
- **Capability** score(s), EN + HI.
- **Size**: average bits / GB on disk for E4–E5.
- **Judge agreement** with human labels per form.
- Report **confidence intervals** (bootstrap) — prompt counts are small.

**Success (my guess at what "good" looks like — team to confirm):**
- E1 shows a clear EN vs Hinglish ASR gap with non-overlapping CIs. *If there is no gap, that is itself a reportable finding; pivot emphasis to E4–E5.*
- E3 cuts Hinglish ASR substantially with over-refusal rising only a few points and capability ~flat.
- E5 beats uniform quantization on Hinglish safety at the same model size.

## 8. Deliverables (map to submission checklist)

- **HF dataset collection:** HinglishHarm (prompts + labels + splits), HinglishXSTest, judged model generations, human-labelled judge-validation set, run logs.
  - **[my call]** Two-tier release so validity never depends on how organisers read "public":
    - **Fully public, ungated:** all prompts, splits, judge labels, human labels, metrics, and every generation **except** completions judged unsafe — those have their text replaced by `[REDACTED: judged unsafe]` while keeping all labels and metadata. Every reported number can be recomputed from this set.
    - **Gated (auto-approve) companion dataset:** full unredacted generations, with a content warning, for researchers who need them.
  - Log full generations locally from E1 onward; redaction is a publish-time step, so nothing has to be rerun.
  - Also file an early Discord ticket asking organisers whether the gated companion is fine. The public tier makes the answer non-blocking.
- **HF model collection:** steering artifacts / LoRA adapter (E3), compressed checkpoints (E4 uniform 4-bit, E5 circuit-aware), each with a model card naming base model + CC BY-NC license.
- **Code:** one public GitHub repo with configs + notebooks that reproduce each experiment.
- **HF article** with the 9 checklist sections; teammates tagged. Main figures: E1 gap bar chart, E2 refusal-projection plot, E6 2×2, E5 safety-vs-size curve.

## 9. Timeline [my call]

| Dates | Work |
|-------|------|
| Nov 2–3 | Accept model licenses; E0 smoke tests on Kaggle; pick base prompts; start variant generation |
| **Nov 5** | **Registration closes — confirm all 3 teammates registered as one team** |
| Nov 4–6 | Finish HinglishHarm + HinglishXSTest; judge validation labels; E1 baseline |
| Nov 6–8 | E2 locate; E4 uniform quant sweep (parallel — different people) |
| Nov 8–11 | E3 fix; E5 circuit-aware quant |
| Nov 11–12 | E6 combined; Expanse 8B runs |
| Nov 12–14 | Publish HF collections, write article, buffer. **Submit by Nov 13** to leave a day of slack |
| Nov 16–21 | If shortlisted: S1 robustness, more languages, Expanse 8B full sweep, demo |

**Cut order [team]** — if time or quota runs short, drop from the bottom up:
1. **Never cut:** E0 smoke tests, E1 baseline gap, E3 steering fix (Idea 1 core).
2. **Keep next:** E4 uniform quantization sweep (Idea 3 minimum).
3. E5 circuit-aware mixed precision.
4. E3b LoRA Harden run (second fix method).
5. E6 combined 2×2.
6. **Cut first:** all Aya Expanse 8B runs.

Decision checkpoint on **Nov 10**: if E3 isn't producing numbers yet, start cutting from item 6.

**[my call] Team split:**
- **Person A (user, Mac):** data and judge — CuratorKIT pipeline, labels, analysis, write-up. Also owns the Kaggle queue, since the 60 h account is theirs.
- **Person B (laptop):** the Idea 1 path — E2 refusal direction, E3. Develops on 4-bit Fire locally, then runs fp16 on Kaggle.
- **Person C (laptop):** the Idea 3 path — E2 circuits, E4, E5. Same local-then-Kaggle split.
- Everyone uses the same evaluation harness (AuditKIT / SafeTune evaluate), so numbers are comparable.

## 10. Risks

| Risk | Mitigation |
|------|-----------|
| CircuitKIT quantization call fails in our env | E0 day-1 smoke test; fallback = quantize with llmcompressor directly using circuit-derived ignore lists |
| Judge can't read Hinglish | §5.4 validation; translate-then-judge fallback |
| No measurable gap in E1 | Report it; Idea 3 (quantization) becomes the headline. Research suggests this is plausible for Devanagari, since Fire's MultiJail mean safe rate is 90%. HI-Latn and HI-Latn-Pert are where the gap most likely shows |
| Steering blows up over-refusal (BabelSteering saw +13 pts on average, worst in low-resource languages; Fire already over-refuses 10.4% in English) | False-refusal ablation is mandatory, not optional; report the full α curve and pick the operating point on the `fit` split |
| Steering raises over-refusal | Conditional steering; report the trade-off curve over steering strength |
| Kaggle quota/session limits (9–12 h sessions) | Checkpoint every run; cache generations (AuditKIT fingerprinted reruns) |
| Translation model refuses to translate harmful prompts | Use an open model locally; teammates hand-write the remainder |
| Running out of time for both ideas | E1→E3 path (Idea 1) is the minimum viable submission; E4 alone (no E5) is a reportable partial result for Idea 3 |

## 11. Left to the builder

Exact open model for translation; capability benchmarks; which SafeTune steer methods beyond refusal-direction (prefer **Faithful**-badged ones); LoRA hyperparameters; exact layer for steering after the sweep; plotting; repo layout; notebook vs script structure.
