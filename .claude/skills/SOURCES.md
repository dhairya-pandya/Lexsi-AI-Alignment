# Vendored skills

Copied verbatim (unmodified) on 2026-11-10 so every teammate's Claude Code loads the same skills from this repo.
Licenses for each source are in `_licenses/`.

| Source repo | Commit | License | Skills (folder names here) |
|---|---|---|---|
| github.com/ndif-team/skills (`plugins/nnsight/skills/*`) | ceae38f | MIT | nnsight, nnsight-ablation, nnsight-activation-patching, nnsight-attention-analysis, nnsight-attribution-patching, nnsight-circuit-discovery, nnsight-interp-experiment-design, nnsight-logit-lens, nnsight-model-steering, nnsight-probing |
| github.com/orchestra-research/ai-research-skills | 773a529 | MIT | transformer-lens, vllm, lm-evaluation-harness, llamaguard, peft, ml-paper-writing, academic-plotting |
| github.com/huggingface/skills (`skills/*`) | 25910b3 | Apache-2.0 | hf-cli, huggingface-datasets, huggingface-paper-publisher, huggingface-papers, huggingface-trackio, huggingface-spaces, huggingface-zerogpu |
| github.com/kaggle/kaggle-cli (`skills/`) | fdc788f | Apache-2.0 | kaggle-cli |

Process skills (brainstorming, writing-plans, executing-plans, subagent-driven-development, TDD, verification)
come from the `superpowers` plugin, enabled for this project in `.claude/settings.json`.

To refresh: re-copy the folders from the repos above and update the commit column.
