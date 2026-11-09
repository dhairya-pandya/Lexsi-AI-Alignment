# Idea search: how we picked the project (2026-11-10)

On Nov 10, 16 research agents ran in parallel across all 3 hackathon tracks. This file records what they found, so nobody repeats the search.
Many verdicts rest on web search and abstracts. "Unverified" means exactly that.

## 1. Why we dropped the Nov 2 Hinglish plan

- Hinglish and code-mixed safety, measured and then fixed with steering, is the most crowded area of the hackathon. More than 15 public team repos work on it.
- At least 3 teams report refusal-direction steering on Tiny Aya Global working **no better than a random direction**:
  - `lvjr3383/tiny-aya-refusal-steering`
  - `maref-org`
  - `Osiyomeoh`
- Cohere Labs already published Tiny Aya code-mixed safety work: https://labscommunity.cohere.com/blog/2026/mix-fine-tune-break/
- Cross-lingual refusal steering already has published papers:
  - https://arxiv.org/abs/2606.01196
  - https://arxiv.org/abs/2608.18089
  - https://arxiv.org/abs/2505.17306
- Cohere's "Tiny Aya in the wild" expeditions already cover code-mixed safety, quantization by language, and SAE feature sharing: https://cohere.com/blog/tiny-aya-in-the-wild

## 2. Candidates and verdicts

Verdict key: OPEN = no prior work found; THIN = adjacent work exists; CROWDED = already done or heavily worked.

| Candidate | Verdict | Why it lost (or won) |
|---|---|---|
| **Where Tiny Aya's safety lives: default preamble, 4K sliding-window horizon, why steering fails** | THIN, nothing direct | **Chosen.** See section 3 |
| English tamper-resistance defenses vs Hindi/Bengali harmful fine-tuning | THIN | TamperBench has one French row with no discussion (https://arxiv.org/abs/2602.06911). No other team works on PS 2b#12. **Fallback project** |
| Unlearn in English, relearn via Hindi fine-tuning (WMDP) | OPEN–THIN | WMDP headroom on a 3B model is unknown. RMU relearning saturates easily. Needs L40S/A100 |
| Regional-variant safety tax (Fire/Water/Earth vs Global) | THIN | The released variants are already merge-repaired, so expected effects are 1–3 points. Cohere plans crosscoder diffing. 5+ adjacent teams |
| Refusal-circuit-aware quantization | THIN | Safepress (Findings EMNLP 2026, https://aclanthology.org/2026.findings-emnlp.2116/) is close. The CircuitKIT compression call has never been run |
| Multilingual RLVR (GRPO math) safety tax | THIN | One WIP repo (Spanish, Qwen). Risk: GRPO gains under 5 points on a 3B model |
| Multilingual agentic-RL safety tax | THIN | English version published (https://arxiv.org/abs/2510.17431). Tool calling is crowded with 12+ teams |
| Multilingual judge-reward hacking in agent RL | THIN | About $40. The hack may not emerge in 200 steps |
| Under-trained (glitch) tokens as a refusal bypass | OPEN for safety use | Detection is already done: Cohere Magikarp lists 8,315 tokens for tiny-aya-global (https://github.com/cohere-ai/magikarp/blob/main/results/reports/CohereLabs_tiny_aya_global.md). **Side bet** |
| Merge-path safety landscape (Global↔Fire interpolation) | THIN | Likely monotone and boring |
| Multilingual evaluation awareness | CROWDED | https://arxiv.org/abs/2610.22333 already covers Indic languages |
| Hallucination probes, multilingual | CROWDED | 4 hackathon teams. Hinglish probe transfer already published (https://arxiv.org/abs/2609.22138) |
| Cross-lingual emergent misalignment on Tiny Aya | CROWDED | ICML 2026 workshop paper plus a Cohere community repo |

## 3. Facts behind the chosen project (verified unless marked)

**Architecture (tiny-aya-global/fire/base, `cohere2`).** Sources: config mirror `1-800-LLMs/tiny-aya-global`, official GGUF header, transformers `modeling_cohere2.py`.
- 36 layers. `layer_types` repeats 3 sliding-window layers then 1 full-attention layer. The full layers are 3, 7, 11, …, 35 (9 of 36).
- `sliding_window = 4096`. A key at position k is visible from query position q iff k > q − 4096.
- RoPE (θ = 50000) is applied **only on sliding layers**. The full layers use **no positional encoding**.
- Parallel block: attention and MLP share one LayerNorm and are added together (`residual + attn + mlp`).
- The LayerNorm mean-centres and has weight but no bias.
- `logit_scale = 1.0`. Embeddings are tied. Vocab is 262,144. d = 2048.
- Attention has 16 query heads and 4 KV heads, with head_dim 128.
- The config allows 500k positions, but the supported context is **8K**.

**Default preamble.**
- The chat template *always* injects a system preamble, even when no system message is passed. The system turn is 359 tokens; a minimal prompt totals 366 tokens.
- It says "You are in **contextual safety mode**". It rejects CSAM and a few other categories, but it will "**accept to provide information and creative content related to violence, hate, misinformation or sex**".
- A user's system message is appended as "# Developer Preamble". It does not replace the default.
- **Aya Expanse 8B** has **no** default preamble and uses full attention with RoPE on every layer. It has 32 layers and logit_scale 0.125.

**Tooling** (repos cloned at their Nov 7 release commits).
- CircuitKIT 0.2.1 has a correct cohere2 port: the sliding mask, NoPE on full layers and LayerNorm all match HF. Its parity test checks KL < 1e-4 at 4,196 tokens, but only in fp32 and only when opted in.
- EAP-IG memory: about 512 tokens fit on a T4, about 1–2K on an L4. Keep circuit prompts at 512 tokens or fewer.
- nnsight 0.8 and plain HF hooks work with cohere2.
- The TransformerLens bridge is correct only with eager attention. Flash attention silently drops the sliding mask.
- SafeTune refusal direction:
  - Extract with `extract_refusal_direction(...)`.
  - Install with `RefusalDirectionModel(mode="ablate"|"steer")` and `orthogonalize_weights`. Because embeddings are tied, `orthogonalize_weights` also changes the LM head.
  - Pooling is last-token only.
  - SafeTune issues #37–39 are fixed in PR #50 (Nov 7). Install from git.
- AuditKIT: bootstrap bug #20 is fixed. There is no native multi-turn support; history goes in `Request.params["messages"]` through a custom adapter.
- vLLM supports cohere2. On a T4 it is slow (Triton attention without tensor cores). Use HF generate in fp16 on a T4, and vLLM in bf16 on an L4 or better.
- Known pitfalls:
  - The template already adds BOS, so `add_special_tokens=True` produces a double BOS.
  - Pin a tokenizer revision from after Sep 30.
  - `tiny-aya-base-32K` exists.
- There are no official SAEs. The only public one is a community SAE on layer 28: `Farseen0/tiny-aya-saes`.

**Prior art for the chosen project.** Nothing directly studies refusal versus distance from the system prompt in interleaved sliding-window models, or safety that depends on a shipped default preamble. Closest work:
- System prompts get ignored under KV-cache eviction: https://doi.org/10.18653/v1/2026.acl-long.1926
- Soft refusals under KV-cache compression: https://arxiv.org/abs/2609.31678
- AnchorKV: https://arxiv.org/abs/2606.17872
- Latent relay past the window: https://arxiv.org/abs/2609.34049
- Full/NoPE layers carry long-range retrieval in hybrid attention: https://doi.org/10.18653/2026.emnlp-main.2275 and https://arxiv.org/abs/2610.10114
- Attention sinks as safety anchors (FLASH): https://doi.org/10.18653/v1/2026.findings-acl.1039
- LongSafety: https://doi.org/10.18653/v1/2025.acl-long.1530
- Many-shot jailbreaking: https://doi.org/10.52202/079017-4121
- Prompt-driven safeguarding (safety prompts shift activations along a refusal direction): https://arxiv.org/abs/2401.18018
- Chat templates confound activation diffs: https://arxiv.org/abs/2605.24583
- Steering reliability varies across models: https://arxiv.org/abs/2504.04635, https://arxiv.org/abs/2505.22637, https://arxiv.org/abs/2602.17881

## 4. Competition snapshot (Nov 10)

- There are about 110 public Lexsi-hackathon repos.
- Crowded problem statements:
  - 2a#4 multilingual (15+)
  - 2b#8 steering (9+)
  - 3#1 tool calling (12+)
  - 3#6 DB agents (7–8)
  - 3#7 workflow agents (9–10)
  - 2a#1 red-teaming (about 7)
  - 2b#6 drift repair (about 6)
- Nearly empty:
  - 2b#12 tamper resistance (0)
  - 1#7 style (0)
  - 1#9 long documents (0)
  - 2b#11 compression (1)
  - 1#6 verifiable reasoning (1)
  - 3#5 multi-hop (about 1)
- True multi-turn or long-context audits (2a#3) have about 1 team.
- Rerun the competition scan on Nov 13.
