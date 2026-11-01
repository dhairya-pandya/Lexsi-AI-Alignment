# Lexsi AI Safety & Alignment Hackathon — Rules & Logistics

Source: kickoff slides (https://hackathon.lexsi.ai/kickoff-slides/), site (https://hackathon.lexsi.ai). Read 2026-11-02.
All times **AoE** (Anywhere on Earth = UTC−12).

## Timeline

| Date | What |
|------|------|
| Oct 12 | Registration opened |
| Oct 31 | Kickoff (Polaris School of Technology, Bengaluru + online) |
| Nov 1 | Hackathon starts (fully online) |
| **Nov 5, EOD AoE** | **Registration closes** — team changes = re-register before this; latest registration counts |
| **Nov 14, EOD AoE** | **Submission deadline** (submission form emailed a couple of days before) |
| Nov 16 | Up to 20 teams told they reach Demo Day |
| Nov 16–21 | Shortlisted teams "prove their thesis" (extra work window) |
| Nov 22 | Demo Day — live demo to judges (in-person + online) |

## Hard rules (slide 28)

1. **Connected repos** — use **at least 2** Lexsi repos as **one connected system** (CuratorKIT, AlignTune, AgentTune, SafeTune, CircuitKIT, AuditKIT).
2. **Model must be one of 7** (Cohere Labs):
   - Tiny Aya Base, Tiny Aya Global, Tiny Aya Fire, Tiny Aya Water, Tiny Aya Earth (all ~3.35B)
   - Aya Expanse 8B, Aya Expanse 32B
   - ⚠️ Command R7B / other models NOT allowed as the main model.
   - Tiny Aya **Fire** = South Asian languages (Hindi, Bengali, Punjabi, Urdu, Gujarati, Tamil, Telugu, Marathi).
   - Tiny Aya license: CC BY-NC 4.0 + Cohere terms (non-commercial) — must state licenses in write-up.
3. **Public on Hugging Face** — resulting models, adapters, logs, datasets, benchmarks: everything publicly accessible.
4. **Hugging Face article** — what you built, how, why, what else you tried.
5. **Originality** — own work; open-source libs and public datasets OK where licenses permit.
6. One track recommended. Multiple submissions allowed but **only the last submission before the deadline counts**.
7. Ideas outside listed problem statements → validate via Discord ticket (#report-issues).

## Submission checklist (slide 29) — HF article sections

1. Problem statement and approach, in brief
2. Hackathon repos used: which ones, for what, which features
3. Models used and their licenses
4. Datasets and benchmarks used and their licenses
5. Model artifacts (weights, adapters, …) public, as an **HF collection**
6. Data artifacts (datasets, logs, benchmarks, …) public, as an **HF collection**
7. Experiments run, findings, what else was tested
8. "With more credits": your approach, before and after results
9. Teammates tagged

"Complete details will be shared soon" — watch email (and spam).

## Judging (slide 30)

- **Validity check first** by internal team: public artifacts, write-up submitted, checklist complete, no duplicates. Tagged to closest track/sub-track.
- **Double-blind** scoring by judge pool (founders, academics, industry labs, organiser research teams). Multiple scores per team, **normalized averages** pick 10–20 Demo Day teams.
- Criteria shown:
  - Write-up clarity — /10
  - Practical usefulness — /10
  - Code/notebook quality — /10
  - Overall review — /100

## Prizes (slide 23) — $30,000+ total

| Place | Lexsi | Cohere | Extra |
|-------|-------|--------|-------|
| 1st | $1,750 | $1,750 | 6 mo HF Pro + hiring |
| 2nd | $1,250 | $1,250 | 6 mo HF Pro + hiring |
| 3rd | $900 | $900 | 6 mo HF Pro + hiring |
| 4th | $600 | $600 | 6 mo HF Pro + hiring |
| 5th | $500 | $500 | 6 mo HF Pro + hiring |

- All Demo Day teams: interview invites + $200 HF credits.
- Some valid-submission teams may also get interview invites.
- Hiring: Research Scientist, Research Intern, DevRel Intern, AI Engineer, Software Engineer.

## Referral program (slide 24)

Teams registering via your link + making a valid submission count. Tiers: 2 teams +$5, 5 +$10, 10 +$20, 20 +$45, 35 +$80, 50 +$90.

## Credits & logistics (slides 25–27)

- HF credits: coupon emailed Nov 1, 5:30 PM IST. First 800 redeemers, registered email + payment method needed. (We have $25.)
- Discord: discord.gg/dtEDQ2Z3eg — queries (#hackathon-lexsi, get role), office hours (Events tab, several/week), #find-teammates, tickets.
- **Check spam.** Senders: hello@lexssi.com, ram.kadiyala@lexsi.ai. Mark "not spam".
- Socials: linkedin.com/company/lexsi-labs, x.com/Lexsi_labs

## Track 2 toolkit facts (slides 14–20)

**SafeTune** (safetune.lexsi.ai) — measures + repairs safety drift. 4 interventions:
- Harden (train-time): stay safe while fine-tuning
- Recover (weight-space): restore lost safety, no retraining
- Unlearn (weight-space): delete skill/knowledge with forget/retain sets
- Steer (inference-time): change behaviour without touching weights
- Pipeline: Drift → Measure → Choose → Repair → Verify → Deploy
- Slide 15 explicitly lists **"Quantize 4-bit for serving"** as a place safety leaks.

**CircuitKIT** (circuitkit.lexsi.ai) — finds circuits; does NOT measure safety itself.
- Discover: 13 algorithms (6 stable: EAP, EAP-IG, EAP-GP, ACDC, IBCircuit, CD-T; 7 research)
- Evaluate: 6 faithfulness pillars (causal patching, ablation, stability, robustness, baseline comparison, transfer)
- Intervene: prune (node/neuron/weight; random/magnitude/Taylor/Wanda), edit (ROME, MEMIT, CaKE, MCircKE…), fine-tune (circuit LoRA, adapter, prefix, BitFit…), **quantize (GPTQ, AWQ, TACQ)**, steer (activation, circuit weight, composer)
- Export: HF checkpoint, lineage, lm-eval

**CuratorKIT** (curatorkit.lexsi.ai/latest/) — ingest (PDF/JSONL/HF) → clean+dedup (exact/MinHash/embedding; secrets/PII/toxicity hygiene) → generate (QA, CoT, DPO pairs, GRPO) → quality gates (grounding, reward, diversity) → export (Alpaca/ShareGPT/DPO/GRPO).

**AuditKIT** (auditkit.lexsi.ai) — benchmarks/HF/custom data → generation (API, HF, vLLM) → metrics (accuracy, LLM judge, safety) → fingerprinted cached runs, compare runs.

**Decision guide (slide 20):** "where does it live / edit at circuit" → CircuitKIT; "repair / harden / unlearn / ship-safe" → SafeTune (+ AuditKIT); "repair targeted where safety lives" → CircuitKIT + SafeTune.
