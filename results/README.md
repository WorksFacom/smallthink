# Pipeline and training reports

English | [Português](README.pt-BR.md)

Reports of the run used in the thesis. Evaluation results are in [`../evaluation/results/`](../evaluation/results/).

| File | Contents |
|---|---|
| `data/judge_summary.json` | Dual-SLM judge (gemma4, qwen3.5:9b) over 949 generated Q&A pairs: 755 kept, 144 corrected, 50 discarded; category agreement and mean scores |
| `data/qa_clean_report.json` | Cleaning of the 899 surviving pairs (42 questions naming the norm, 114 near-duplicates removed, 857 source lines rebuilt → 743 pairs), per-resolution counts and corpus metrics |
| `data/cot_report.json` | CoT generation: 862 problems, 760 accepted (642 answer, 88 unrelated-passage refusals, 30 empty refusals), 714 train / 46 validation |
| `training/trainer_state.json` | Axolotl/Transformers training log: losses, evaluation loss per epoch, memory |
