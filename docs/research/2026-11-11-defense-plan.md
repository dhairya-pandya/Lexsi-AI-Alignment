# Defense experiment plan: closing the cross-lingual tamper gap

Written 2026-11-11, from 4 parallel literature searches. Anchored to our findings:
- English-trained fine-tuning-attack defenses are **language-local**: after a Bengali attack, defended Tiny Aya Fire answers 0-4% of harmful English prompts but **59-61% of harmful Bengali prompts**.
- SafeTune's default English "refusal aux" set is **~92% compliant answers** (8% refusals); hardening on it degrades safety pre-attack.
- Real-refusal Lisa is safe everywhere but **over-refuses 79-90%** of benign prompts (no helpful anchor).
- GRPO hardening pilot was **null** (binary reward -> zero GRPO gradient; TRL never hit Tiny Aya's stop token).

**Novelty (all 4 agents agree):** no paper tests Vaccine/Booster/Lisa/TAR/RepNoise under a non-English attack with per-language eval. Defense-side language-locality + the data-quality artifact are new. **Scoop risk:** a Cohere community repo does cross-lingual misalignment-direction work on Tiny Aya ("submitted"). Post a preprint once Tier 0 + one Tier 1 result exist.

---

## Tier 0 — Validity (do FIRST; the headline is not trustworthy without these)

Our result partly **contradicts Poppi et al. 2410.18210** (attack transfers to all languages). Either Fire genuinely differs (it has Bengali/Hindi SFT data; Llama/Qwen don't) or our setup produces the asymmetry. Two checks resolve it:

- **T0a. Judge audit (~$5 + native labelling).** Bengali guards are unreliable: LlamaGuard-3 has no Bengali; one study found LlamaGuard-4 vs a calibrated judge at kappa 0.014 (2608.22335); Qwen3Guard F1 swings 84->44 between strict/loose modes (2510.14276). Draw ~300 stratified responses (en/bn x {refuse, harmful, degenerate, benign}); 2 native Bengali speakers label blind; report kappa for Qwen3Guard (both modes), translate-then-judge (NLLB back-translation), string-match. Re-score the 61% under each. Add **native Bengali prompts** (BanglaSafe eval set, MultiJail native bn) to rule out translationese. **Owner: A + Hindi/Bengali teammates.**
- **T0b. Control matrix + seeds (~$10).** {defended, undefended} x {en, bn attack} x {en, bn eval}, >=3 fine-tune seeds, prompt-level bootstrap CIs, paired tests. Report a **locality ratio** = defense gain on en-eval / gain on bn-eval. The undefended arm settles "is it the attack or the defense that's local" (our data: undefended attack is already partly local, 18% en vs 49% bn).
- **T0c. Degeneration gate.** Already built (`is_degenerate` head-check); report coherence/garbage-rate beside every harm number so RepNoise-style collapse isn't scored "safe."

## Tier 1 — The defense that closes the gap (the positive, publishable result)

- **T1a. Benign-twin anchor (rank 1; cheap, high signal).** Fixes the over-refusal collapse. Hold the Lisa/Booster safety set at N~2k; vary refusal share {100, 70, 50, 30}%; fill the rest with **matched benign** answers (half "seemingly-toxic" from FalseReject-Train / WildJailbreak-benign, half generic Alpaca/Tulu). Evidence: WildJailbreak over-refusal 54%->1.6% with benign twins (2406.18510); FalseReject ~10% safety data holds MMLU (2505.08054). **Predict:** XSTest over-refusal 79-90% -> <15%, en harm stays 0-4%, bn harm stays ~60% (isolates over-refusal from the language gap). ~$8.
- **T1b. Target-language dose-response (rank 2; targets the gap directly).** Best ratio from T1a; vary NLLB bn share {0, 10, 25, 50}% (refusals + twins). Add an **en+hi (25% hi) arm evaluated zero-shot on bn** = clean Indic-transfer test (Hindi not in MultiJail). Evidence: Deng 2310.06474 (500 translated pairs, bn included, unsafe 10%->4%); Aakanksha EMNLP2024 (15% mix, Hindi harm -72%). **Predict:** bn harm 60% -> <15% at 25-50% bn; hi->bn transfer partial. ~$10.
- **T1c. Leave-one-language-out (the cleanest framing).** Train defense on en+bn, attack hi, and swap; compare to en-only. Isolates whether adding languages generalizes. Reuses our cell code. ~$10. Fold into T1b.

Plot **harm vs over-refusal vs MMLU (en + Global-MMLU bn) as a Pareto front** = the "safety tax" figure.

## Tier 2 — Mechanism (explains WHY; cheap, mostly inference on EXISTING checkpoints)

Best-supported hypothesis: **harm detection is language-shared, refusal execution (late-layer write) is language-specific.** So a bn attack moves the bn refusal write, leaving en intact. Refusal-direction cosine alone is a **weak** transfer predictor (Zhang MRL2026, Ivanova 2610.28897, Ruellan 2610.38001) — don't lean on it; sweep all 36 layers, not just pick_layer=24.

- **T2a. Refusal-direction drift (~$3).** Build en/bn/hi refusal directions per layer (SafeTune, matched non-safety control). On our saved `tamper_*` checkpoints compute cos(r_l^pre, r_l^post) and proj(dW, r_l). **Predict:** bn fine-tune drifts r_bn, not r_en.
- **T2b. 3x3 causal cross-ablation.** Ablate r_x, measure refusal in language y, before/after attack.
- **T2c. dW geometry (~$3).** Cosine of per-module dW across en/bn/hi fine-tunes; Poppi-style top-20% safety-mask overlap; **spectral test** — remove dW's top-2 SVD directions, check if bn refusal returns (Lee 2610.00320).
- **T2d. CircuitKIT per-language refusal circuits (~$5).** EAP-IG edges, Jaccard by layer, overlap with dW-touched edges. Predict shared early/mid, specific late. Bootstrap splits (EAP-IG is data-sensitive).
- **T2e. Union-subspace anchoring defense.** Penalize movement along span{r_en, r_bn, r_hi} during fine-tuning vs en-only (ProCon/C-ΔΘ style), against an adaptive rank-spreading attack. ~$8.

## Tier 3 — RL done right (fix the null pilot)

- **T3a. Fixed GRPO (~$10).** (1) Set `generation_config.eos_token_id=[6,3]`, assert `completions/clipped_ratio ~0` before training (Tiny Aya's eos is inconsistent across config files; TRL ran to max length). (2) **Graded reward** (Qwen3Guard paper 2510.14276): safe 1 / controversial 0.5 / unsafe 0, **-5 on benign refusal**, + a **language-ID gate** (GRPO drifts to English, 2506.05850). (3) SFT-init, lr 1-5e-6, 8 generations, drop zero-variance groups. Evidence: binary->95-97% refusal, hybrid->98% safety at 5.3% refusal.
- **T3b. MPO / multilingual DPO (~$8; cheaper, strong).** Match each language's safe-minus-unsafe log-likelihood gap to English (2505.16869): Llama bn attack-success 49.5->12.4 with 700 translated pairs. Tests whether a stronger start survives the attack. **Expectation: RL gives a better starting point, not tamper resistance by itself.**

## Tier 4 — Strengthen for publication

- **T4a. Second model family.** Rerun the headline cell on **Llama-3.1-8B-Instruct** (direct Poppi comparability) and/or Aya Expanse 8B. ~$15.
- **T4b. Stronger / adaptive attack.** TamperBench-style LR/epoch sweep; romanized/code-mixed Bengali; gradient-free (abliteration/prefill, 2605.26526).
- **T4c. Translated-vs-native data (gap in the literature).** Native Hindi (Aya Red-teaming ~915) vs NLLB-translated, same refusals; a filtered (back-translation/LaBSE) arm.

---

## Recommended order for the Nov 14 submission vs the demo-day window (Nov 16-21)

**For submission (if any budget/time): T0a + T0b + T1a.** They make the finding trustworthy and add the positive "benign-twin anchor fixes over-refusal" result. ~$15-20, runnable on the teammate accounts.

**Demo-day window:** T1b/T1c (dose-response), T2a/T2c (mechanism on existing checkpoints, ~$6 total, high insight/$), T3a (fixed GRPO), T4a (second family).

**Highest insight-per-dollar:** T2a + T2c (~$6, reuse saved checkpoints) — likely explain the whole phenomenon. **Highest publication value:** T0a + T0b (close the validity/scoop risk) then T1a+T1b (the defense).

## Key citations
Poppi 2410.18210 · TamperBench 2602.06911 · Qwen3Guard 2510.14276 · WildJailbreak 2406.18510 · FalseReject 2505.08054 · Deng 2310.06474 · Aakanksha 2406.18682 · MPO 2505.16869 · Wang refusal-universal 2505.17306 · Zhang MRL2026 2026.mrl-1.19 · Lee 2610.00320 · C-ΔΘ 2602.04521 · BanglaSafe 2608.22335 · NEST 2611.10971 · Krasnodebska 2606.07535
