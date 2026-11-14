# Reasoning with small language models over CFM norms

English | [Português](README.pt-BR.md)

Code and data for the undergraduate thesis *Reasoning com Small Language Models*: a Qwen3-4B fine-tuned to reason in Portuguese over retrieved passages of resolutions of the Conselho Federal de Medicina (CFM), served with retrieval-augmented generation (RAG) on consumer GPUs.

Every step that produces training data runs on small local models through Ollama. Large models (Gemini, gpt-oss-120B) appear only in the evaluation: drafting the gold set, judging answers, and as a baseline.

```
legal_documents/*.pdf ──ingest──▶ passages ──index──▶ Qdrant ◀── rag_bot (question → passages → model)
                                     │                                 ▲
                                     ▼                                 │
                      Q&A (qwen3:4b-instruct) → dual judge (gemma4 + qwen3.5:9b)
                      → filters, dedupe, contamination check → RAG problems
                      → Portuguese CoT (qwen3.5:9b) → QLoRA (Axolotl) → GGUF → Ollama
```

## Layout

| Directory | Contents |
|---|---|
| [`data_pipeline/`](data_pipeline/) | PDF segmentation, Q&A generation, dual-SLM judge, cleaning and corpus metrics, RAG training problems, CoT generation; source PDFs and the data snapshot used in the thesis |
| [`training/`](training/) | Axolotl QLoRA config for Qwen3-4B and the LoRA → GGUF → Ollama export |
| [`rag_bot/`](rag_bot/) | Passage indexing (bge-m3 + Qdrant), terminal chat and batch answer API |
| [`evaluation/`](evaluation/) | Gold set construction, answer collection for every system, Gemini rubric judge, automatic scores, BERTScore; the 200-question gold set, answers, judgments and results |
| [`results/`](results/) | Data-pipeline reports (judge, cleaning, corpus metrics, CoT) and the training log |

## Hardware

| Step | GPU |
|---|---|
| Q&A generation, inference of the local models | NVIDIA Quadro P1000 (4 GB, Pascal) |
| Judges, CoT generation, training | NVIDIA GeForce RTX 5050 (8 GB, Blackwell; CUDA ≥ 12.8) |

## Reproducing

Each directory has its own README and Docker Compose file. In order:

1. `data_pipeline`: ingest the PDFs → `data/passages.jsonl`.
2. `rag_bot`: index the passages in Qdrant (the RAG problems take their hard negatives from this index).
3. `data_pipeline`: qa → judge → clean → problems → cot → `data/cot_alpaca_{train,val}.jsonl`.
4. `training`: copy the two files to `training/data/`, train, export to Ollama.
5. `rag_bot`: start the API with the exported model.
6. `evaluation`: collect answers, build judge sets, judge, score, BERTScore.

`data_pipeline/data/` ships the passage snapshot and the training files used in the thesis, so steps 4–6 can run without regenerating the data.

## Results (200 questions: 180 answerable, 20 unanswerable)

All systems received the same prompt and the same three retrieved passages per question (gold passage retrieved in 87.2% of the answerable questions). Judge: Gemini 3.1 Pro with a rubric; details in [`evaluation/results/scores.json`](evaluation/results/scores.json).

| Metric | Fine-tuned Qwen3-4B | Original Qwen3-4B | gpt-oss-120B |
|---|---|---|---|
| Correctness (0–2) | 1.47 | 1.59 | 1.65 |
| Fully correct | 58% | 68% | 73% |
| Key-point coverage | 76% | 82% | 85% |
| Verdict in the first word (binary) | 82% | 50% | 84% |
| Faithfulness to passages (0–2) | 1.93 | 1.79 | 1.94 |
| Refused unanswerable / answerable | 100% / 4% | 100% / 3% | 100% / 5% |
| Median time per answer | 31.8 s | 42.7 s | 6.8 s |
| BERTScore F1 (XLM-R large / Legal-BERTimbau) | 0.935 / 0.769 | 0.903 / 0.681 | 0.911 / 0.711 |

The original Qwen3-4B was scored on 194 questions; on 6 its reasoning exhausted the 1,536-token limit.
