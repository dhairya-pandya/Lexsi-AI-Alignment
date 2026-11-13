# Does Safety Survive an Attack in Another Language? Cross-Lingual Tamper Resistance of Fine-Tuning Defenses

> **Code & this writeup:** https://github.com/dhairya-pandya/Lexsi-AI-Alignment · every number carries a 95% CI and its over-refusal.

## 1. Problem and approach
Model providers that offer fine-tuning-as-a-service apply safety defenses (Lisa, Vaccine, Booster, RepNoise) to stop a customer's fine-tune from stripping safety, but those defenses are built and evaluated almost entirely in **English**. We ask the question the literature skipped: if the attacker's poisoned fine-tuning data is in **Hindi or Bengali**, do the defenses still hold, and do they hold *in those languages*?

They do not. A defended Tiny Aya Fire, attacked in Bengali, refuses almost all harmful **English** prompts (0–4% harm) while answering **59–61%** of harmful **Bengali** ones, a gap an English-only evaluation certifies as robust. We trace the damage to a **low-rank weight write** and show it is **repairable post-hoc**: removing the top 1–2 singular directions of the attack's `dW` drops Bengali harm 50%→14% with capability preserved. A training-time prevention variant (subspace anchoring) does **not** work; repair beats prevention.

We arrived here from a negative result. Our first question was whether Tiny Aya's refusals live in its always-injected 359-token safety preamble; they do not (§7, G1/G2): safety is in the weights, which is exactly what makes the tamper question the interesting one.

## 2. Hackathon repos used
- **SafeTune**: the fine-tuning-stage defenses under test, refusal-direction extraction, judges.
- **AlignTune**: the GRPO pilot (null result, §7).
- **CircuitKIT / nnsight**: mechanism, refusal-direction drift and the low-rank `dW` analysis.
- **NLLB-200-distilled-600M**: translating the attack and safety data into Hindi/Bengali.

## 3. Models and licenses
CohereLabs/tiny-aya-global, tiny-aya-fire (CC-BY-NC-4.0 + Cohere acceptable-use addendum); CohereLabs/aya-expanse-8b (CC-BY-NC-4.0) as the architectural control. Judges: cais/HarmBench-Mistral-7b-val-cls, Qwen/Qwen3Guard-Gen-4B.

## 4. Datasets, benchmarks and licenses
See `DATA_LICENSES.md`: StrongREJECT (MIT), MultiJail (MIT), XSTest (CC-BY-4.0), OR-Bench-Hard (CC-BY-4.0), MMLU (MIT), UltraChat (MIT), WikiText-103 (CC-BY-SA-3.0), Magikarp report (Apache-2.0).

## 5. Model artifacts (HF collection)
Repaired checkpoint (Experiment A, k=1 low-rank projection of a Bengali-attacked Tiny Aya Fire), gated:
https://huggingface.co/codingpanda06/tiny-aya-fire-bn-repaired_k1

## 6. Data artifacts (HF collection)
Two-tier evaluation dataset:
- Public (redacted): https://huggingface.co/datasets/dhairyapandya/tiny-aya-tamper-evals
- Gated (full completions): https://huggingface.co/datasets/dhairyapandya/tiny-aya-tamper-evals-gated

Public tier redacts every completion to a harmful prompt that is not a clear refusal (robust to judge error);
benign answers, refusals and all labels are kept. 47,034 rows per tier, 0 harmful completions in the public tier.

## 7. Experiments and findings
### What we tested first: does Tiny Aya's safety live in its default preamble? (negative result)
Tiny Aya's chat template always injects a 359-token "contextual safety mode" preamble, and 27 of 36 layers only see the last 4,096 tokens. We pre-registered two day-1 gates (paired bootstrap CIs) and both failed:

- **G1, preamble removed (E1, StrongREJECT n=208, XSTest+OR-Bench n=280):**
  - Removing the preamble does not lower refusal. Tiny Aya Global refuses +3.4 pp *more* without it (CI [+0.5, +6.2]); Fire moves +1.4 pp (n.s.).
  - Compliance with the preamble's own rules (name "Aya", built by Cohere, American spelling, paragraphing) is 91–93% with or without it.
  - Identity and refusal behaviour are in the weights.
  - The control, Aya Expanse 8B with Tiny Aya's preamble injected, reads the preamble (rule compliance 0.78 → 1.00) but its refusal does not move either.
- **G2, preamble pushed past the 4,096-token window (3K vs 5.5K tokens of benign Wikipedia filler):**
  - Refusal and rule compliance shift identically with and without the preamble, so there is no preamble horizon.
  - Long benign context itself lowers refusal from about 81% to about 67% and raises unsafe answers from about 3% to 9–11%, as LongSafety reported for other models.
- **Deployment audit:** the preamble is present in HF/vLLM chat and the official GGUF, and absent in raw completion APIs and an Ollama build without a template. Because safety does not depend on it, those surfaces are not less safe.

### Main study: English-trained tamper defenses vs a Hindi or Bengali attacker
**Setup.**
- An attacker fine-tunes Tiny Aya Fire (LoRA, 1 epoch) on 500 rows: 80% benign GSM8K and 20% harmful BeaverTails.
- The attack data is in English, Hindi or Bengali. The Hindi and Bengali sets were translated with NLLB-200 and checked: 100% in the target script, translated length ≈1.0× the source, under 1% repetitive rows.
- The provider applies a SafeTune defense:
  - **Lisa** acts during the fine-tune.
  - **Vaccine** and **Booster** harden the model before it (stage 1, English safety data only, then the attack).
  - **RepNoise** immunizes the model before it.
  - **No defense** is the baseline.
- Harm is the HarmBench-classifier unsafe rate on StrongREJECT (English, n=208) and on MultiJail Bengali prompts (n=120), with 95% bootstrap CIs.
- We also report over-refusal (XSTest + OR-Bench, n=280), MMLU, and the degenerate-output rate.
- Full table: `article/tamper_results.csv`. Figure: `article/figures/tamper_harm_heatmap.png`.

**Findings.**
1. **Attacks are language-local.**
   - With no defense, an English attack makes 40% of English prompts harmful.
   - A Hindi attack makes 12% of English prompts harmful and a Bengali attack 18%.
   - The Bengali attack does its damage mostly in Bengali: 49% of Bengali prompts become harmful.
2. **The toolkit's default English "refusal" set teaches compliance.**
   - SafeTune's refusal aux set is built from BeaverTails `is_safe` responses. Only 21 of its 256 rows (8%) are refusals.
   - Hardening on it makes the *unattacked* model unsafe: Vaccine 24%, Booster 59%, plain safety-SFT 46%, against 0.5% for the base model.
   - After any attack these defenses reach 47–65% harm in every language.
   - We reported this upstream (see "What else we tried").
3. **With real refusals, an English-trained defense is language-local too.**
   - Vaccine trained on real English refusals refuses everything before the attack (100% over-refusal).
   - An English attack erases it: 68% harm.
   - A Bengali attack leaves English prompts at 0% harm but unlocks **61% harm on Bengali prompts**, worse than no defense (49%).
   - **Booster with real refusals replicates this:** after a Bengali attack, 4% harm on English prompts but 59% on Bengali prompts.
   - **An English-only evaluation would call these defenses robust against a Bengali attacker.**
   - Lisa with real refusals holds in all languages (≤6% harm), but only by refusing 79–90% of benign requests.
4. **RepNoise "protects" by breaking the model.**
   - Its immunized model produces degenerate text on 100% of harmful prompts and MMLU drops to 0.41.
   - The attacker's fine-tune partly restores coherence: 2% degenerate after an English attack, 14% after Hindi, 28% after Bengali.
   - The low harm numbers (1–11%) therefore come from incoherent output, not refusal.

**Benign twins fix over-refusal but NOT the language gap (defense experiment).** The real-refusal defenses over-refused because the safety set had refusals but no helpful anchor. We rebuilt the English safety set as refusals + benign "twins" (FalseReject helpful answers to seemingly-toxic prompts + Alpaca), swept the refusal share, hardened with Vaccine, then ran the Bengali attack. Harm = HarmBench-unsafe on StrongREJECT-en and MultiJail-bn, 95% bootstrap CIs.

| refusals:twins | pre over-refusal | pre MMLU | post-attack over-refusal | post-attack unsafe-en | post-attack **Bengali harm** |
|---|---|---|---|---|---|
| 100:0 | 1.00 | 0.00 | 1.00 [1.00,1.00] | 0.00 | 0.63 [0.55,0.72] |
| 70:30 | 1.00 | 0.46 | 0.50 [0.44,0.55] | 0.03 | 0.53 [0.43,0.62] |
| 50:50 | 0.73 | 0.45 | 0.31 [0.26,0.37] | 0.08 | 0.48 [0.40,0.57] |
| 30:70 | 0.73 | 0.51 | 0.28 [0.23,0.33] | 0.11 | 0.56 [0.47,0.65] |

Two independent failures, one cure each. Benign twins monotonically cut over-refusal (1.00 -> 0.28) and restore capability (MMLU 0.00, the all-refusal model refuses even MMLU, -> 0.51), at a visible safety/helpfulness cost (unsafe-en 0.00 -> 0.11). **But Bengali-prompt harm stays ~0.48-0.63 at every ratio (overlapping CIs):** an English-only safety set, however balanced, does not protect Bengali. Over-refusal is a data-balance problem; the cross-lingual gap is not, and needs target-language safety data.

**Does target-language safety data close the gap? Barely (dose-response).** We added NLLB-translated Bengali safety data (refusals + twins) to the Vaccine defense at {0, 25, 50}%, then ran the Bengali attack.

| Bengali safety data | unsafe-en | over-refusal | Bengali harm |
|---|---|---|---|
| 0% | 0.06 [0.03,0.09] | 0.22 | 0.57 [0.47,0.66] |
| 25% | 0.16 | 0.16 | 0.45 [0.36,0.54] |
| 50% | 0.18 | 0.09 | 0.46 [0.38,0.55] |

Bengali harm falls only 0.57 -> ~0.46 (overlapping CIs, a weak effect) while English harm gets *worse* (0.06 -> 0.18). The defense runs before the attack, and the Bengali fine-tune overwrites the Bengali safety we added ("alternative pathways", Poppi 2410.18210; "adaptive attacker spreads the update", Lee 2610.00320). So the gap is a tamper-resistance problem, not just a data-coverage one.

**Mechanism (weak).** Per-layer refusal-direction drift leans the right way, a Bengali attack moves the Bengali direction slightly more (cos 0.768 vs 0.805 for English), but an English attack that causes 40% harm barely moves either direction, so refusal-direction cosine is a poor proxy for the damage (as Zhang MRL2026, Ruellan 2610.38001 warn). The actionable signal: the attack's weight change is low-rank (top-2 singular directions hold ~50% of the o_proj update energy), supporting a "remove the top directions" repair (Lee 2610.00320).

**Low-rank update projection closes the gap (the defense).** The mechanism said the attack's weight change `dW = W_post - W_base` is low-rank, so we removed the top-k singular directions of `dW` per attention/MLP projection matrix and kept the rest: `W_repaired = W_base + (dW - top-k SVD component)`, a post-hoc, training-free repair the provider applies after the fine-tune (Lee 2610.00320). On the Bengali-attacked model (1 seed):

| k (dirs removed) | en harm (StrongREJECT) | **bn harm (MultiJail)** | over-refusal | MMLU |
|---|---|---|---|---|
| 0 (attacked) | 0.19 [.14,.24] | **0.50 [.42,.58]** | 0.05 | 0.55 |
| 1 | 0.02 [.00,.05] | **0.14 [.09,.21]** | 0.16 | 0.56 |
| 2 | 0.02 [.00,.05] | 0.14 [.08,.21] | 0.16 | 0.58 |
| 4 | 0.01 [.00,.02] | 0.11 [.06,.17] | 0.18 | 0.52 |
| 8 | 0.01 [.00,.02] | 0.08 [.04,.13] | 0.20 | 0.50 |

Removing the **single top direction (k=1)** already drops Bengali harm **50% -> 14%** (non-overlapping CIs) and English harm 19% -> 2%, **with MMLU preserved** (0.55 -> 0.56): the harmful write is nearly rank-1 and concentrated, while the benign fine-tune gain is diffuse and survives. k=1-2 is the knee; k=4-8 push Bengali harm to 8% at the cost of more over-refusal and lower MMLU. Not a degeneration artifact (MMLU and coherence hold at every k). Figure: `results/proj/pareto.png`.

**Prevention does not work; repair does (the counterpart test).** The obvious training-*time* alternative to post-hoc repair is to stop the write from landing: anchor the hidden state to the refusal subspace during the fine-tune (ProCon / C-ΔΘ, 2602.04521). We implemented a single-model projection constraint: measure the base model's refusal-subspace "budget" on harmful prompts, then penalize any harmful/refusal example whose last-token hidden state (layer 16) projects below that budget along the anchored directions, and ran three arms against the Bengali attack (balanced English safety set, 1 seed):

| arm | en harm (SR) | **bn harm (MultiJail)** | over-refusal | MMLU |
|---|---|---|---|---|
| no anchor (control) | 0.00 | **0.22 [.15,.29]** | 0.66 | 0.49 |
| anchor span{r_en} | 0.00 | **0.22 [.15,.29]** | 0.60 | 0.50 |
| anchor span{r_en,r_bn,r_hi} | 0.01 | **0.33 [.25,.42]** | 0.60 | 0.50 |

(Figure: `results/anchor/bars.png`.) Anchoring the English refusal direction changes Bengali harm *not at all* (0.22, identical to the control), and anchoring all three languages makes it *worse* (0.33), the opposite of the intent, plausibly because the non-English refusal directions, estimated from machine-translated prompts, are noisy and over-constraining along them fights the task objective. So **prevention failed and repair won**.

**Overall for defenses:** benign twins cleanly fix over-refusal and capability loss but do **not** close the language gap; balanced English data, 50% target-language data, pilot GRPO, and **training-time union-subspace anchoring** also fail to close it (the attack overwrites or outruns the added safety). What *does* close it is **low-rank update projection**: removing the attack's top-1-2 `dW` singular directions restores multilingual safety while preserving capability. So the cross-lingual tamper gap, while real and invisible to English-only evaluation, is **repairable post-hoc** because the harmful update is low-rank; repair beats prevention.

**Can GRPO restore cross-lingual safety? (pilot, null result).** We tried hardening the Bengali-attacked model with GRPO (AlignTune/TRL), reward = refuse harmful prompts and stay helpful on benign ones. At pilot scale (60 steps, LoRA, lr 1e-6, binary reward) nothing moved: Bengali-prompt harm 48% -> 46%, English 16% -> 17%, over-refusal 5.7% -> 5.4% (no collapse), MMLU flat. Two reasons, both visible in training: TRL generated without Tiny Aya's `<|END_RESPONSE|>` stop token, so every completion hit the length cap; and a binary reward gives GRPO no gradient when a prompt's sampled completions all agree (loss and grad-norm were ~0). A real test needs the stop token wired into generation, a graded safety reward, more steps, and a higher learning rate. Reported as future work, not a result.

**Takeaway.** Tamper-resistance claims for multilingual models need (a) evaluation in the attacker's language, (b) over-refusal and output-quality metrics next to harm, and (c) an audit of what the "safety" data actually contains.
### Deployment surfaces
`results/e0/deployment_surfaces.csv`: the preamble is present in HF/vLLM chat and the official GGUF, absent in raw completion APIs and an Ollama build without a template.
### What else we tried

## 8. With more credits

The result stands on one seed and one model. With more GPU budget, in priority order:

1. **Second seed + second model**: rerun the attack/repair on a fresh seed and on Llama-3.1-8B to confirm the low-rank repair (Exp A) and the cross-lingual gap generalize beyond tiny-aya-fire.
2. **Native-speaker judge audit**: our Bengali harm labels lean on Qwen3Guard + NLLB translation; a native-speaker spot-check would bound judge error on the headline number.
3. **Anchoring sweep with native refusal directions**: Exp B anchored on translated prompts. Re-run the λ/layer sweep with non-translated Bengali/Hindi refusal directions before concluding prevention is dead, not just dormant.

## 9. Team

- [@dhairyapandya](https://huggingface.co/dhairyapandya): data, judges, evaluation, publishing
- [@runtime-terror](https://huggingface.co/runtime-terror): behaviour experiments, gates
- [@ByteKnight28](https://huggingface.co/ByteKnight28): mechanism, steering
