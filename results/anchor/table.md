# Experiment B — union-subspace anchoring (tamper-prevention): NEGATIVE

Bengali attack on tiny-aya-fire, balanced English safety set, 1 seed, layer 16, lam 1.0.
Results repo: codingpanda06/hsafe-results (tamper_anchor_{none,en,union}_bn[_mj]).

| arm | en harm (StrongREJECT) | bn harm (MultiJail) | en harm (MJ) | over-refusal | MMLU |
|-----|-----|-----|-----|-----|-----|
| none (lam=0 control)        | 0.00 | 0.22 [.15,.29] | 0.00 | 0.66 | 0.49 |
| en (anchor r_en)            | 0.00 | 0.22 [.15,.29] | 0.00 | 0.60 | 0.50 |
| union (anchor r_en+r_bn+r_hi)| 0.01 | 0.33 [.25,.42] | 0.00 | 0.60 | 0.50 |

- en-anchor == control on bn harm (anchoring English direction does nothing for Bengali).
- union-anchor worse than control (adding bn/hi directions hurts; likely noisy MT directions).
- All arms: English safe, MMLU preserved. Prevention fails; post-hoc repair (Exp A) wins.
