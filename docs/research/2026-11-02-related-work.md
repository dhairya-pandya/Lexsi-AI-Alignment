# Related Work Brief — Hinglish Safety + Safety-Aware Compression

Written 2026-11-02 for the design in [2026-11-02-hinglish-safety-design.md](../superpowers/specs/2026-11-02-hinglish-safety-design.md).
Read depth: papers marked ★ were read beyond the abstract (HTML/full text via fetch); the rest are from abstracts and search snippets. Several arXiv IDs are 2026 preprints — re-check numbers before citing them in the article.

---

## 1. What changes in the plan (summary)

| # | Finding | Plan change |
|---|---------|-------------|
| 1 | Cohere's **aya_redteaming** has **915 Hindi prompts written by native speakers** (originally Romanized; released in Devanagari, with English translations), Apache-2.0, harm category + global/local labels ★ | Make it the **core attack source**. Pair it with HarmBench for "dangerous capability" harms. Cuts translation work and adds India-specific (local) harms |
| 2 | **Tiny Aya's own safety report doesn't cover Hindi**: MultiJail has no Hindi, and its XSTest run is English-only ★ | Our Hindi/Hinglish numbers are **new for this model family**. Say so in the article |
| 3 | Tiny Aya Fire is already **comparatively safe** (MultiJail mean 90.0, min 78.1) but **over-refuses 10.4%** on English XSTest ★ | Expect a small gap in Devanagari. The gap most likely shows up in **Romanized and misspelled forms**. Over-refusal is the main risk for any fix |
| 4 | English refusal direction is **shared across languages**. Failures happen because harmful inputs **don't push far enough along it** ("subthreshold"), and harmful/harmless clusters are **less separated** outside English ★ | E2 measures three cheap, published diagnostics: direction cosine, projection-vs-threshold, cluster separation (silhouette). If projection is the problem, **adding** the English direction should fix it, which sets up E3 |
| 5 | **BabelSteering**: adding the English refusal vector raised harmful refusal ~11 pts across 8 languages, but over-refusal rose ~13 pts, worst in low-resource languages. **Ablating a separate false-refusal vector** is what keeps over-refusal down ★ | E3 uses the BabelSteering recipe: add refusal at one layer + ablate an orthogonalized false-refusal direction, folded into the weights. Sweep strength; report the trade-off curve |
| 6 | Cohere's quantization study: **non-Latin scripts degrade most**, and **automatic metrics hide the damage** (Japanese: −1.7% automatic vs −16% human-judged) | E4 compares Devanagari vs Romanized (Latin script) degradation directly. Back up judge numbers with a human-checked sample |
| 7 | Tiny Aya's report: GGUF **Q4_K_M −1.4, Q4_0 −2.1, Q8_0 ≈ no loss** on quality ★ | E4 adds the **official GGUF Q4_K_M / Q4_0** builds, which are what users actually run. 8-bit becomes a control |
| 8 | **Critical-weight protection (AWQ-trust)**: Fisher-based safety scores computed on **English** AdvBench restored Korean/Arabic safety under 4-bit (Llama KO 9.6 → 66.1% safe) ★ | E5 gets a strong, simple baseline. **New ablation:** protect weights scored on English vs Hinglish safety data (or the CircuitKIT circuit) |
| 9 | **PolyGuard / IndicGuard degrade badly on Romanized input** (unsafe-slip rate roughly doubles). Standard practice is **back-translate to English, then WildGuard** (BabelSteering, Minionese) | Primary judge = back-translate → WildGuard (SafeTune ships WildGuard). Don't use Indic guard models directly on Hinglish. Human validation stays |
| 10 | Safety SFT/DPO with a **15% safety mix** cut Hindi harm 72% (Aya red-teaming paper). Tiny Aya's authors found models are "safest when they include SFT data for the respective languages" ★ | E3b LoRA: Hinglish refusals **plus** Hinglish benign compliance, mixed ~15% into general instruction data |
| 11 | **Phonetic misspelling** splits sensitive words into harmless tokens; the authors had to **hand-write** these prompts because automatic generation failed | Measure token fragmentation per form (free, mechanistic). Keep the HI-Latn-Pert set smaller (~100 prompts), built from a rule-based misspelling list plus hand edits |

---

## 2. Papers, by part of the plan

### 2.1 The gap exists (E1)
- **MultiJail** (Deng et al., ICLR 2024). Non-English prompts produce far more unsafe output; ChatGPT unsafe rate reached 80.92% with malicious instructions. This is the classic citation. Note: no Hindi. [paper](https://arxiv.org/pdf/2310.06474)
- **Code-Switching Red-Teaming, CSRT** (Yoo et al., 2024). Mixing languages *within* a sentence was the most effective attack, 46.7% more successful than English. [arXiv 2406.15481](https://arxiv.org/abs/2406.15481)
- **Haet Bhasha aur Diskrimineshun** (2025). Hinglish plus phonetic misspellings (e.g. "dee dee o es atak") break safety; the attack works at the tokenizer level. Prompts were hand-written. [arXiv 2505.14226](https://arxiv.org/html/2505.14226v3)
- ★ **Minionese** (2026). 18 languages; Hindi is in tier 3 (mid-low resource). Perturbation types: translation, code-switching, transliteration, translationese. Tested Aya Expanse 8B. Findings:
  - Transliteration vulnerability depends on **script**.
  - Code-switching stays effective even at the lowest resource tier.
  - Llama reached 99% attack success on Hindi transliteration.
  - Judge: WildGuard after NLLB back-translation, 100% agreement on a human-checked sample.

  [arXiv 2607.10112](https://arxiv.org/html/2607.10112)
- **Safety Alignment Illusion** (2026). Extends cross-lingual evaluation to Hinglish and calls code-mixed Hinglish neglected. [arXiv 2608.18131](https://arxiv.org/pdf/2608.18131)
- ★ **Tiny Aya tech report** (2026). Safety was distilled from Command A and carried into the regional models by merging with Global ("merging reduces the safety gap, especially for Fire and Earth"). Evaluated on MultiJail (10 languages, no Hindi) and English XSTest only.

  | Model | MultiJail min / mean | XSTest over-refusal | XSTest under-refusal |
  |---|---|---|---|
  | Fire | 78.1 / 90.0 | 10.4% | 15.0% |
  | Global | 87.0 / 91.1 | 10.4% | 15.5% |

  [arXiv 2603.11510](https://arxiv.org/html/2603.11510v1)

### 2.2 Data
- ★ **The Multilingual Alignment Prism / aya_redteaming** (Aakanksha et al., 2024). Native speakers were paid to write prompts in 8 languages. Hindi has 915 prompts: 608 global harm, 307 local harm. Each has an English translation, 9 harm categories and a global/local label. Apache-2.0. Hindi was originally typed Romanized and released in Devanagari. [arXiv 2406.18682](https://arxiv.org/html/2406.18682v2) · [dataset](https://huggingface.co/datasets/CohereForAI/aya_redteaming)
  - Its harms lean social: hate, harassment, discrimination. Pair it with HarmBench/AdvBench for dangerous-capability harms (cyber, weapons, drugs).
- **XSTest** (Röttger et al.): 250 safe-but-scary-sounding prompts. A 2026 study translating XSTest into 11 languages found over-refusal of safe prompts rose from **6.4% (EN) to 13–24%** in other languages. Over-refusal in Hinglish may already be high before any fix. [XSTest](https://arxiv.org/pdf/2308.01263) · [translated XSTest](https://arxiv.org/html/2610.20221v1)
- **OR-Bench** pseudo-harmful prompts: the source of BabelSteering's false-refusal vector, needed for E3.

### 2.3 Where safety lives (E2)
- **Arditi et al. 2024**: refusal is mediated by a single direction (difference in means between harmful and harmless activations).
- ★ **Refusal Direction is Universal Across Safety-Aligned Languages** (Wang et al., NeurIPS 2025). 14 languages, no Hindi. Extraction is difference in means at post-instruction tokens, with the layer chosen by how much ablating it reduces refusal. Cross-lingual cosine is high, peaking mid-depth. Harmful and harmless clusters are **less separated** outside English (lower silhouette score), and jailbreaks fall between the clusters. They recommend increasing that separation but test no defence. Code: [mainlp/Multilingual-Refusal](https://github.com/mainlp/Multilingual-Refusal) · [paper](https://huggingface.co/papers/2505.17306)
- ★ **Minionese, mechanistic side**: the main failure is **subthreshold activation**. Harm is encoded, but the projection onto the refusal direction stays below the trigger level. Cross-lingual refusal is effectively one-dimensional. They suggest steering along that direction but don't test it.
- **Who Bridges Safety?** (2026, cross-lingual shared safety pathways) and **Multilingual Safety via Sparse Weight Editing** (ICML 2026). Both find safety neurons shared across languages. Relevant to SafeTune's `identify_safety_neurons`. [arXiv 2608.09095](https://arxiv.org/pdf/2608.09095)

**Our angle:** nobody has run these diagnostics on **Romanized Hinglish** or on **Tiny Aya**, and nobody has checked whether quantization shrinks the projection.

### 2.4 Fixing it (E3)
- ★ **BabelSteering** (2026). English steering vectors folded into the weights, always on, no inference cost. Tested on Gemma 7B, Llama 3.1 8B and Qwen 7B, in 8 languages including Bengali; no Hindi. [arXiv 2608.16577](https://arxiv.org/html/2608.16577)
  - Recipe:
    - refusal direction by difference in means;
    - add it at one layer with strength α;
    - ablate an orthogonalized false-refusal vector (from OR-Bench) at all layers with weight λ;
    - Gemma best setting: λ=0.8, α=0.4;
    - layer chosen with a KL-divergence threshold of 0.1.
  - Gemma results:
    - Bengali harmful refusal 37→54, but over-refusal 36→59.
    - Thai harmful refusal 72→90, over-refusal 6→32.
    - MMLU changed by less than 3 points.
  - Adding the refusal vector alone, without the false-refusal ablation, made over-refusal explode: 45% for Gemma, ~90% for Qwen.
  - Their judge: back-translate with X-ALMA, then WildGuard.
- **Conditional Activation Steering, CAST** (Lee et al., IBM, 2024). Steers only when an activation condition matches. This is an alternative way to protect over-refusal, but the condition detector itself may miss Hinglish, which is worth testing. [arXiv 2409.05907](https://arxiv.org/abs/2409.05907) · code: IBM/activation-steering
- **AlphaSteer** (2025). Null-space-constrained refusal steering that keeps benign activations untouched. [arXiv 2506.07022](https://arxiv.org/html/2506.07022v1)
- **More to refusal than a single direction** (2026) and **task-conditioned over-refusal** (2026). Stronger steering increases refusal and over-refusal together, and there is no single over-refusal direction, so tune carefully. [2602.02132](https://arxiv.org/html/2602.02132v2) · [2603.27518](https://arxiv.org/html/2603.27518v1)
- **Multilingual Safety via Self-Distillation, MSD** (2026). Response-free: train the model to give its own English answer when the prompt is in another language. Uses only multilingual queries. A cheap E3b alternative. [arXiv 2605.02971](https://arxiv.org/html/2605.02971v2)
- **Layer-wise Swapping** (EACL 2026). Merging that moves safety from an English expert into language experts. [ACL](https://aclanthology.org/2026.eacl-long.98.pdf)
- ★ **Aya red-teaming paper**. SFT and DPO with a 15% safety mix. Hindi harm fell 72.4%. Training on local harms transferred to global harms better than the reverse.

### 2.5 Compression (E4–E5)
- **How Does Quantization Affect Multilingual LLMs?** (Marchisio et al., Cohere, Findings of EMNLP 2024). Non-Latin scripts suffer most (103B model: Latin −0.7% vs non-Latin −1.9%). Automatic metrics heavily understate the harm: Japanese −1.7% on automatic tasks vs −16% judged by humans. [arXiv 2407.03211](https://arxiv.org/pdf/2407.03211)
- ★ **Preserving Fairness and Safety in Quantized LLMs (AWQ-trust)** (2026). [arXiv 2601.12033](https://arxiv.org/html/2601.12033v2)
  - Each weight is scored by safety sensitivity minus general sensitivity, using squared gradients (an approximation of Fisher information). Safety loss is on AdvBench; general loss is on Dolly; 128 samples each.
  - The top 60% of weights stay in fp16 and the rest go to INT4, with AWQ.
  - Llama 3.1 8B MultiJail % safe (EN / KO / AR):

    | Setting | EN | KO | AR |
    |---|---|---|---|
    | Full precision | 83.3 | 15.2 | 31.9 |
    | AWQ | 80.6 | 9.6 | 24.7 |
    | AWQ-trust | 90.3 | 66.1 | 89.1 |

  - The weights were scored on **English only**, yet safety in other languages improved.
  - Dynamic quantization was more stable than static.
- **Q-resafe** (ICML 2025): quantization-aware safety patching. [arXiv 2506.20251](https://arxiv.org/pdf/2506.20251)
- **Alignment-Aware Quantization** (2025): minimising reconstruction error does not preserve safety. [arXiv 2511.07842](https://www.arxiv.org/pdf/2511.07842v2)
- **Task-Circuit Quantization, TACQ** (Xiao et al., 2025). This is what CircuitKIT's `tacq` selector implements. [arXiv 2504.07389](https://arxiv.org/html/2504.07389v2)
  - Keeps about 1% of weights, the task-critical ones chosen by gradient, in 16-bit.
  - Llama 3 8B at 3.1 bits recovers about 96% of MMLU; the biggest gains are at 2–3 bits.
  - An independent reproduction got lower numbers at 2 bits and notes that TACQ needs one run per task. [arXiv 2607.00908](https://arxiv.org/pdf/2607.00908)
- **Alignment Collapse Under KV Cache Quantization** (2026). KV-cache quantization also breaks alignment. Out of scope, but worth a mention. [arXiv 2606.09864](https://arxiv.org/pdf/2606.09864)

### 2.6 Judges
- ★ SafeTune ships **WildGuard** and **LlamaGuard-3**. BabelSteering and Minionese both judge with WildGuard **after back-translating to English**.
- **PolyGuard** (COLM 2025): 17 languages. Its own paper shows performance drops on code-switched input. [arXiv 2504.04377](https://arxiv.org/abs/2504.04377)
- **IndicGuard** (2026): 10 Indic languages, built on Gemma-3-4B, released as `l3cube-pune/IndicGuard`. [arXiv 2606.22841](https://arxiv.org/abs/2606.22841)
  - A Bangla study found the unsafe-slip rate of PolyGuard and IndicGuard **roughly doubles under Romanization** (30% → 63%). [BanglaShield 2610.24288](https://arxiv.org/html/2610.24288)
- Tiny Aya's authors judged with **Command A** directly in the source language. A Cohere API trial key could serve as a second judge on a sample.

### 2.7 Robustness (stretch S1)
- Weight-folded steering is exactly what **abliteration** removes. The ICML 2026 study of training-free tampering found that safeguards break under refusal-direction ablation and prefilling. Testing this is an honest limits check. [ICML 2026](https://icml.cc/virtual/2026/78555) · [arXiv 2605.26526](https://arxiv.org/pdf/2605.26526)

---

## 3. Where we're new (for the article's "contributions")

1. First Hinglish (Romanized + misspelled) safety measurement on the **Tiny Aya** family, including **Fire vs Global**, which tests whether regional specialisation costs safety.
2. First Romanized-Hinglish application of the refusal-direction diagnostics (cosine, subthreshold projection, cluster separation).
3. BabelSteering-style fix on a **Hindi/Hinglish** target, which that paper didn't test, with the over-refusal trade-off reported.
4. **Script-split quantization safety**: Devanagari vs Romanized degradation under GPTQ and the official GGUF builds.
5. Safety-weight protection scored on **English vs Hinglish** data vs a **CircuitKIT circuit**: which source of "safety weights" protects Hinglish best.
