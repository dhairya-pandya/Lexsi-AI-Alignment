# Design addendum: Do English-trained tamper defenses survive a Hindi or Bengali attacker?

Written 2026-11-11. This is the fallback (design §7 of [2026-11-10-where-safety-lives-design.md](2026-11-10-where-safety-lives-design.md)), triggered because both day-1 gates failed.

## Gate outcome being kept as a negative finding

Results are in `results/e1_*` and `results/gate_g*`.

- **E1, preamble removal:** on Tiny Aya Global, Tiny Aya Fire and Aya Expanse, taking the default preamble away does not reduce refusal. The pooled refusal difference is +3.4 pp on Global and +1.4 pp on Fire, and on Global the preamble makes the model slightly *less* likely to refuse.
- **E1, preamble rules:** compliance with the preamble's rules is 91–93% with or without the preamble, so the identity and safety behaviour live in the weights.
- **G2, the 4K window:** no horizon effect specific to the preamble. Both P-default and P-none shift the same way between 3K and 5.5K tokens.
- **Long benign context:** it lowers refusal from about 81% to about 67% whether or not the preamble is present. This is a known effect (LongSafety).

These go into the article as a short "what we tested first" section.

## Question

Fine-tuning-stage safety defenses are tuned and evaluated in English. Do they still reduce harmful behaviour when the attacker's contaminated fine-tuning data is in Hindi or Bengali?

**Prior work:**
- TamperBench has one undiscussed French row: https://arxiv.org/abs/2602.06911
- Poppi et al. show attacks transfer across languages but test no defenses: https://arxiv.org/abs/2410.18210

## Setup [my call]

**Model:** `tiny-aya-fire` (South Asian variant), pinned revision.

**Threat model:** a fine-tuning service.
- The user uploads contaminated data in language L: benign carrier plus a harmful fraction p.
- The provider runs fine-tuning with defense D, using English safety (refusal) data.
- D ∈ {`PlainSFTTrainer` (no defense), `LisaTrainer`, `BoosterTrainer`, `VaccineTrainer`, `RepNoiseTrainer`}. All are SafeTune Faithful and run at their default hyperparameters.

**Attack data:**
- n = 500 rows, p = 0.2.
- Harmful rows are SafeTune's BeaverTails contamination pairs.
- Benign rows are SafeTune's GSM8K carrier.
- Both are machine-translated with NLLB-200-distilled-600M (CC-BY-NC) into Hindi (`hin_Deva`) and Bengali (`ben_Beng`).

**Languages and defense data:**
- L ∈ {en, hi, bn}.
- The English safety set is SafeTune's refusal aux set and is never translated. That is the point of the experiment.

**Evaluation** (the existing runner, P-default):
- **Harmful:** StrongREJECT eval, English, n = 208, scored by the HarmBench classifier.
- **Harmful, multilingual:** MultiJail en and bn, 60 per bucket, scored by Qwen3Guard.
- **Over-refusal:** XSTest + OR-Bench, n = 280.
- **Capability:** MMLU, n = 143.

**Metrics:**
- Harmful rate H(D, L).
- Relative defense efficacy RDE(D, L) = 1 − H(D, L) / H(none, L).
- Language gap = RDE(D, en) − RDE(D, L).
- Bootstrap CIs over prompts.
- Over-refusal and capability are reported next to every harm number.

**Grid:** 5 defenses × 3 languages, 1 seed. That is 15 cells, about 30–40 min each on an L4, so about 8–10 L4-hours and roughly $8. A second seed comes after if the budget allows.

**Pilot:** run the English column first. If no defense lowers English harm against no defense, report that floor effect and lower p to 0.1.

**Release:**
- Translated harmful data and generations go to a gated dataset.
- The public tier is redacted, as before.
- No fine-tuned model weights are published: they are deliberately weakened.

## Out of scope

TAR (8-GPU cost), full-parameter attacks, per-language defense tuning (only as a stretch: Hindi refusal aux on the best defense).
