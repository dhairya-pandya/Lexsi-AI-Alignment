# Where Does Tiny Aya's Safety Live? The Preamble, the 4K Horizon, and Why Steering Fails

> Results sections are filled in from `results/` after the runs. Every number comes with a 95% CI and its over-refusal.

## 1. Problem and approach
Tiny Aya's chat template always injects a 359-token "contextual safety mode" preamble, and 27 of its 36 layers only see the last 4,096 tokens. We measure how much refusal behaviour comes from that preamble (E1), whether it fades past the sliding window (E2), where it acts (E3), whether it explains weak refusal steering (E4), and restore its refusals without it (E5).

## 2. Hackathon repos used
- **SafeTune**: refusal-direction cross-check, judges.
- **CircuitKIT**: EAP-IG circuit for the preamble contrast (clean = P-default, corrupt = P-none).
- **CuratorKIT**: MinHash deduplication of prompt sets and filler pools.
- **AuditKIT**: capability comparison before/after the fix.

## 3. Models and licenses
CohereLabs/tiny-aya-global, tiny-aya-fire (CC-BY-NC-4.0 + Cohere acceptable-use addendum); CohereLabs/aya-expanse-8b (CC-BY-NC-4.0) as the architectural control. Judges: cais/HarmBench-Mistral-7b-val-cls, Qwen/Qwen3Guard-Gen-4B.

## 4. Datasets, benchmarks and licenses
See `DATA_LICENSES.md`: StrongREJECT (MIT), MultiJail (MIT), XSTest (CC-BY-4.0), OR-Bench-Hard (CC-BY-4.0), MMLU (MIT), UltraChat (MIT), WikiText-103 (CC-BY-SA-3.0), Magikarp report (Apache-2.0).

## 5. Model artifacts (HF collection)
_link_

## 6. Data artifacts (HF collection)
_link_ (public tier: judged-unsafe completions redacted; gated companion with full text)

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

**Can GRPO restore cross-lingual safety? (pilot, null result).** We tried hardening the Bengali-attacked model with GRPO (AlignTune/TRL), reward = refuse harmful prompts and stay helpful on benign ones. At pilot scale (60 steps, LoRA, lr 1e-6, binary reward) nothing moved: Bengali-prompt harm 48% -> 46%, English 16% -> 17%, over-refusal 5.7% -> 5.4% (no collapse), MMLU flat. Two reasons, both visible in training: TRL generated without Tiny Aya's `<|END_RESPONSE|>` stop token, so every completion hit the length cap; and a binary reward gives GRPO no gradient when a prompt's sampled completions all agree (loss and grad-norm were ~0). A real test needs the stop token wired into generation, a graded safety reward, more steps, and a higher learning rate. Reported as future work, not a result.

**Takeaway.** Tamper-resistance claims for multilingual models need (a) evaluation in the attacker's language, (b) over-refusal and output-quality metrics next to harm, and (c) an audit of what the "safety" data actually contains.
### Deployment surfaces
`results/e0/deployment_surfaces.csv`: the preamble is present in HF/vLLM chat and the official GGUF, absent in raw completion APIs and an Ollama build without a template.
### What else we tried

## 8. With more credits
_before/after plan_

## 9. Team
_tag teammates_
