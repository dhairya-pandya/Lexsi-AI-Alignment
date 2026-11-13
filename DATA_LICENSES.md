# Data and model licenses

| Asset | Source | License | Verified |
|---|---|---|---|
| Tiny Aya (global, fire, base) | huggingface.co/CohereLabs/tiny-aya-* | CC-BY-NC-4.0 + Cohere acceptable-use addendum | yes (model card) |
| Aya Expanse 8B | huggingface.co/CohereLabs/aya-expanse-8b | CC-BY-NC-4.0 | yes (model card) |
| Chat-template fixtures (tests/fixtures/*.jinja) | copied from the models' tokenizer_config.json | as above | yes |
| StrongREJECT (harmful_en.jsonl; 313 -> 298 after minors filter + dedup) | github.com/alexandrasouly/strongreject | MIT | yes (repo) |
| MultiJail (multijail.jsonl; en, zh, ar, bn, sw) | huggingface.co/datasets/DAMO-NLP-SG/MultiJail | MIT | yes (dataset card) |
| XSTest safe prompts (benign.jsonl) | github.com/paul-rottger/exaggerated-safety | CC-BY-4.0 | verify before release |
| OR-Bench-Hard-1k, 150 sampled (benign.jsonl) | huggingface.co/datasets/bench-llm/or-bench | CC-BY-4.0 | verify before release |
| MMLU test, 200 sampled (capability.jsonl) | huggingface.co/datasets/cais/mmlu | MIT | yes (dataset card) |
| UltraChat 200k test_sft (data/filler/f1.jsonl) | huggingface.co/datasets/HuggingFaceH4/ultrachat_200k | MIT | yes (dataset card) |
| WikiText-103 test (data/filler/f2, f3) | huggingface.co/datasets/Salesforce/wikitext | CC-BY-SA-3.0 | yes (dataset card) |
| Rule probes (rule_probes.jsonl) | written for this project | CC-BY-4.0 | n/a |
| Magikarp under-trained token report (data/magikarp/) | github.com/cohere-ai/magikarp | Apache-2.0 | yes (repo) |
