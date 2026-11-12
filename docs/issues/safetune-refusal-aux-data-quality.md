# [Data quality] Default English "refusal aux" set is ~92% compliant answers, not refusals

**Repo:** Lexsi-Labs/SafeTune · **Component:** `safetune.runner.utils.data_utils._harden_raw_pairs` (refusal/aux set)

## Summary
The default English safety ("refusal aux") set used by the alignment-/finetune-stage trainers is built from
BeaverTails `is_safe` responses. Inspecting it, **~92% of the "refusal" targets are compliant/helpful answers**, not
refusals — only ~8% are genuine refusals. Trainers that harden on this set are therefore teaching the model to
*comply* on the safety split, which **degrades pre-attack safety** and weakens the defense the trainer is meant to
provide.

## Evidence
- Sampling the aux pairs returned by `_harden_raw_pairs(n)[2]` and labelling by a refusal-prefix check + manual
  spot check: ~8% refusals, ~92% substantive answers.
- Downstream: using this set as the safety signal, the defended (pre-attack) model's own refusal rate on harmful
  prompts is lower than expected; swapping in a set of real, varied refusals (and benign "twins" to avoid
  over-refusal) measurably improves pre-attack safety without the over-refusal collapse that a refusals-only set
  causes.

## Suggested fix
- Filter the aux set to genuine refusals (prefix/classifier check), or source refusals from a refusal-labelled
  corpus; optionally ship a **balanced** default (real refusals + benign twins) so hardening fixes safety without
  inducing over-refusal.
- Document that the current set is compliance-heavy so users don't mistake it for a refusal set.

## Context
Found while studying cross-lingual tamper resistance of SafeTune defenses on Tiny Aya Fire (the defense quality
directly affects the measured robustness). Happy to share the labelling script and the balanced-set recipe.
