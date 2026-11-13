# Blue-teaming multilingual safety: when an English-trained defense meets a Bengali attack

![One model, two verdicts: English harm 0 to 4 percent, Bengali harm 59 to 61 percent after a Bengali fine-tuning attack](https://huggingface.co/datasets/dhairyapandya/cross-lingual-tamper-assets/resolve/main/banner.png)

*A hackathon project on cross-lingual tamper resistance of fine-tuning-stage safety defenses. Live demo with every chart and an animated walkthrough: [getalife-lexsi-study.vercel.app](https://getalife-lexsi-study.vercel.app). Every number below carries a 95% confidence interval and its over-refusal.*

## The game we were playing

Put yourself on the blue team. A model provider lets customers fine-tune a chat model, and the provider's job is to make sure a customer's fine-tune cannot quietly strip the model's safety. There is a whole family of defenses for exactly this, applied at the fine-tuning stage: Lisa, Vaccine, Booster, RepNoise. They work. The catch is that almost every one of them is built, tuned, and evaluated in English.

So we put on the red hat for a moment and asked a question the English-centric defense literature had mostly left open: what if the attacker does not fine-tune in English? What if the poisoned data is in Hindi or Bengali? And then, back on the blue team, the harder question: if the defense still looks fine in English, would we even notice?

The short answer is no, we would not. And that is the whole problem.

## What the defender sees, and what the defender misses

We took `CohereLabs/tiny-aya-fire`, defended it with the standard toolkit, then let an attacker run a poisoned fine-tune in Bengali. Then we evaluated the result in both languages. Here is the gap, the repair, and the counterpart test, all in one place:

![Results: the language gap, the low-rank repair dropping Bengali harm from 50 to 14 percent, and the failed training-time prevention](https://huggingface.co/datasets/dhairyapandya/cross-lingual-tamper-assets/resolve/main/results.png)

Read the top of that figure slowly, because it is the finding. The same defended model, after the same Bengali attack, refuses almost every harmful English prompt (0 to 4 percent get through) while answering 59 to 61 percent of harmful Bengali ones. If your dashboard is in English, you ship this model and call it robust. It is not robust. It is wide open in a language you did not test.

We checked the obvious escape hatch first: just add Bengali safety data to the defense. It barely helps. Bengali harm falls only from about 0.57 to about 0.46, with overlapping intervals, while English harm actually gets worse. The attacker's Bengali fine-tune overwrites the Bengali safety we added. That is the tell that this is a tamper-resistance problem, not a coverage gap. You cannot patch it by collecting more data in more languages, because the attack removes the safety after you add it.

## Why it happens, and the repair that works

Here is the part that gives a defender something to actually do. We looked at the weight change the attack makes, `dW = W_post - W_base`, and it is nearly rank one. The top one or two singular directions of that update carry about half of its energy. The harmful behavior is written into a tiny, concentrated subspace, while the benign skill the fine-tune also picks up is spread diffusely across many directions.

That asymmetry is the opening. If the harm lives in the top directions and the useful gain does not, then we can cut the top directions out and keep the rest.

![The linear algebra of the repair: decompose the attack's weight change, see it is nearly rank one, and project out the top direction](https://huggingface.co/datasets/dhairyapandya/cross-lingual-tamper-assets/resolve/main/repair_poster.jpg)

*A 45-second animated walkthrough of the linear algebra: [watch the repair](https://huggingface.co/datasets/dhairyapandya/cross-lingual-tamper-assets/resolve/main/repair.mp4), or see it on the [live demo](https://getalife-lexsi-study.vercel.app).*

So we did exactly that. For each attention and MLP projection matrix, we took the SVD of `dW`, removed the top k singular directions, and kept everything else: `W_repaired = W_base + (dW - top-k component)`. It is post-hoc and training-free. The provider runs it once, after the customer's fine-tune, with no retraining.

| directions removed (k) | English harm | Bengali harm | over-refusal | MMLU |
|---|---|---|---|---|
| 0 (attacked) | 0.19 | **0.50** | 0.05 | 0.55 |
| 1 | 0.02 | **0.14** | 0.16 | 0.56 |
| 2 | 0.02 | 0.14 | 0.16 | 0.58 |
| 4 | 0.01 | 0.11 | 0.18 | 0.52 |
| 8 | 0.01 | 0.08 | 0.20 | 0.50 |

Removing just the single top direction drops Bengali harm from 50 percent to 14 percent, with non-overlapping intervals, and English harm from 19 percent to 2 percent. MMLU holds at 0.56. That last number matters: the fix is not lobotomizing the model to make it look safe. The harmful write was concentrated and we removed it; the benign gain was diffuse and survived. The knee of the curve is at k of 1 to 2. Push further and Bengali harm keeps falling to 8 percent, at the cost of more over-refusal and some capability.

## What did not work (and why that is useful to know)

A good blue-team write-up reports the dead ends, because they tell you where not to spend the next person's time.

**Prevention during training did not work.** The natural alternative to repairing after the fact is to stop the write from landing in the first place. We anchored the model's hidden state to the refusal subspace during the fine-tune. Anchoring on the English refusal direction changed Bengali harm not at all (0.22, identical to the no-anchor control). Anchoring on all three languages made it worse (0.33), probably because the non-English refusal directions were estimated from machine-translated prompts and were too noisy to constrain against. Repair beat prevention here.

| arm | Bengali harm | over-refusal | MMLU |
|---|---|---|---|
| no anchor (control) | 0.22 | 0.66 | 0.49 |
| anchor English direction | 0.22 | 0.60 | 0.50 |
| anchor English + Bengali + Hindi | 0.33 | 0.60 | 0.50 |

**A GRPO safety-hardening pilot was a null.** At pilot scale nothing moved, for two reasons we could see in the logs: the trainer generated without Tiny Aya's stop token so every sample hit the length cap, and a binary reward gave no gradient when all samples for a prompt agreed. Fixable, but not a result yet.

**Balancing the safety data fixed over-refusal but not the gap.** Rebuilding the English safety set with benign "twins" cleanly cut over-refusal and restored capability, but Bengali harm stayed around 0.48 to 0.63 at every ratio. Over-refusal is a data-balance problem. The cross-lingual gap is a different animal.

**One finding we reported upstream.** The toolkit's default refusal auxiliary set is drawn from BeaverTails `is_safe` responses, but only about 8 percent of those rows are actual refusals. Hardening on it teaches the model to comply rather than refuse, which makes the *unattacked* model unsafe (Booster reached 59 percent, Vaccine 24 percent). We documented this and raised it with the SafeTune team.

## Related work, and what is new here

We are standing on two existing lines of work. Cross-lingual fine-tuning fragility is known: attacks transfer across languages ([Poppi et al., 2410.18210](https://arxiv.org/abs/2410.18210)), and the refusal direction does not cleanly transfer across languages ([Zhang, 2026.mrl-1.19](https://aclanthology.org/2026.mrl-1.19/)). Cross-lingual defenses exist too, but at training time ([Safety Kernel, 2610.24614](https://arxiv.org/abs/2610.24614); [MPO, 2505.16869](https://arxiv.org/abs/2505.16869); [Multilingual Refusal Alignment, 2606.07535](https://arxiv.org/abs/2606.07535)). Separately, post-hoc low-rank safety repair is an active area in the English setting ([Lee et al., 2610.00320](https://arxiv.org/abs/2610.00320); [LSSF, ACL 2025](https://aclanthology.org/2025.acl-long.1479/); [LoX, 2506.15606](https://arxiv.org/abs/2506.15606); [NLSR, 2412.12497](https://arxiv.org/abs/2412.12497)).

What is new here is the combination, not a new primitive. We bring low-rank post-hoc repair into the cross-lingual tamper setting, evaluate in the attacker's language, show an English-only evaluation hides the gap, and show the repair beats its training-time prevention counterpart ([ProCon, 2509.06795](https://arxiv.org/abs/2509.06795)). We do not claim to have invented low-rank repair or discovered cross-lingual fragility. The value is the intersection and the prevention-versus-repair result, on one seed and one model.

## The artifacts, models, and licenses

Everything is public.

- **Model artifacts (HF collection):** [cross-lingual-tamper-resistance-models](https://huggingface.co/collections/dhairyapandya/cross-lingual-tamper-resistance-models-6af769175d8792224c4e9a03). The repaired checkpoint is the k=1 projection of a Bengali-attacked Tiny Aya Fire: [tiny-aya-fire-bn-repaired_k1](https://huggingface.co/codingpanda06/tiny-aya-fire-bn-repaired_k1).
- **Data artifacts (HF collection):** [cross-lingual-tamper-resistance-data](https://huggingface.co/collections/dhairyapandya/cross-lingual-tamper-resistance-data-6af76916237237008742051b). Two tiers of the evaluation set: a [redacted tier](https://huggingface.co/datasets/dhairyapandya/tiny-aya-tamper-evals) that replaces every non-refusal completion to a harmful prompt with a placeholder while keeping all labels, and a [full tier](https://huggingface.co/datasets/dhairyapandya/tiny-aya-tamper-evals-gated) for safety research.
- **Models and licenses:** CohereLabs tiny-aya-global and tiny-aya-fire (CC-BY-NC-4.0 plus Cohere's acceptable-use addendum); Aya Expanse 8B as an architectural control. Judges: HarmBench-Mistral-7b-val-cls and Qwen3Guard-Gen-4B.
- **Datasets and benchmarks:** StrongREJECT (MIT), MultiJail (MIT), XSTest (CC-BY-4.0), OR-Bench-Hard (CC-BY-4.0), MMLU (MIT), UltraChat (MIT), WikiText-103 (CC-BY-SA-3.0). Attack and safety data translated with NLLB-200-distilled-600M.
- **Hackathon repos we built on:** SafeTune (the defenses under test, refusal-direction extraction, judges), AlignTune (the GRPO pilot), CircuitKIT with nnsight (the refusal-direction and low-rank analysis), and NLLB for translation.

## Why we think this scales into a general safety win

Our result stands on one seed and one model, so read the next part as a direction, not a claim. But the shape of it is encouraging for a defender with real resources.

The repair works because of a structural fact: a tamper attack writes harm into a low-rank subspace, and useful capability does not live there. That fact is not special to Bengali or to Tiny Aya. If it holds across seeds and model families, then low-rank repair becomes a cheap, general post-hoc safety pass that a provider can run after any customer fine-tune, in any language, without retraining and without hurting the model.

With more compute and training budget, the natural next steps all point the same way. Confirm the low-rank structure on a second seed and on a larger model like Llama-3.1-8B. Add a native-speaker judge audit so the non-English harm numbers rest on human labels, not a translated auto-judge. Re-run the prevention track with refusal directions estimated from native prompts rather than machine translations, which is the most likely reason prevention failed rather than being impossible. Combine the repair with properly balanced, multilingual safety data so the model is both hard to tamper and genuinely multilingual-safe out of the box. Each of these is a budget question, not a research mystery.

The honest headline for the blue team is this: an English-only evaluation will certify a model that is unsafe in other languages, so evaluate in the attacker's language, always report over-refusal and output quality next to harm, and audit what your "safety" data actually contains. And when the gap shows up, you are not stuck. The damage is low-rank, and low-rank damage is repairable.

## With more credits

In priority order:

1. A second seed and a second model (Llama-3.1-8B) to confirm the low-rank repair and the cross-lingual gap generalize.
2. A native-speaker judge audit to bound judge error on the headline Bengali number.
3. A prevention sweep with native, non-translated refusal directions before calling prevention dead.

## Team

- [@dhairyapandya](https://huggingface.co/dhairyapandya): data, judges, evaluation, publishing
- [@runtime-terror](https://huggingface.co/runtime-terror): behaviour experiments, gates
- [@ByteKnight28](https://huggingface.co/ByteKnight28): mechanism, steering
