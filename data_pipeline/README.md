# Data pipeline

English | [Português](README.pt-BR.md)

Turns CFM resolutions into Portuguese chain-of-thought training data. Every model runs locally through Ollama (≤ 10B parameters); the scripts reach it at `host.docker.internal:11434`.

```
ingest → (rag_bot: index passages) → qa → judge → clean → problems → cot
```

| Step | Service / script | Output in `data/` |
|---|---|---|
| Segmentation | `ingest` / `ingest_cfm.py` | `passages.jsonl`, `manifest.json`, `extracted/` |
| Q&A generation | `qa` / `generate_qa.py` | `qa_pairs.jsonl` |
| Dual-SLM judge | `judge` / `judge_qa.py` | `qa_pairs_reviewed.jsonl`, `qa_pairs_clean.jsonl` |
| Filters, dedupe, contamination check, corpus metrics | `clean` / `clean_qa.py` | `qa_pairs_final.jsonl`, `qa_pairs_rejected.jsonl`, `qa_clean_report.json` |
| RAG training problems | `problems` / `build_rag_problems.py` | `rag_problems.jsonl` |
| Portuguese CoT | `cot` / `generate_cot.py` | `cot_alpaca_{train,val}.jsonl`, `cot_alpaca_report.json`, `cot_raw.jsonl` |

```bash
ollama pull qwen3:4b-instruct; ollama pull gemma4; ollama pull qwen3.5:9b; ollama pull bge-m3
docker compose build
docker compose --profile ingest run --rm ingest
# index data/passages.jsonl with ../rag_bot (needed by `problems`)
docker compose --profile qa run --rm qa --include-kinds artigo,item --pairs-per-passage 3 --no-resume
docker compose --profile qa run --rm qa --include-kinds considerando,ementa --pairs-per-passage 1
docker compose --profile judge run --rm judge
docker compose --profile clean run --rm clean
docker compose --profile problems run --rm problems
docker compose --profile cot run --rm cot
```

Arguments after the service name are appended to its command. Models: `QA_MODEL` (default `qwen3:4b-instruct`), `JUDGE_MODEL_A` / `JUDGE_MODEL_B` (`gemma4`, `qwen3.5:9b`), `COT_MODEL` (`qwen3.5:9b`). The `clean` step reads the evaluation gold set from `../evaluation/questions/gold.jsonl` for the contamination check.

## Steps

**`ingest_cfm.py`** segments each PDF in `legal_documents/` into passages of kind `ementa`, `considerando`, `artigo`, `item` (roman-numeral items of the Código de Ética chapters without articles), `assinatura` and `exposicao_motivos`, each with a canonical citation (e.g. `Resolução CFM nº 2217/2018, Art. 73`). Chapter headings end the preceding article and are prepended to every article of the chapter as `context` (`Capítulo IX – SIGILO PROFISSIONAL. É vedado ao médico:`); articles over 1,500 characters are split at incisos/parágrafos into parts that repeat the caput; page headers/footers are removed; passages repeated across resolutions get `duplicate_of`.

**`generate_qa.py`** follows the corpus recipe of Athaydes et al. (2025): varied question types (rule, practical case, permission/prohibition, conditions/exceptions, definition), self-contained questions that never name the resolution or article, technical tone, answers ending with `Fonte: <citation>`, sampling at temperature 0.8 with frequency penalty 0.5 and presence penalty 0.8. Articles that only approve the Code, put a resolution in force or revoke others are skipped.

**`judge_qa.py`** has two SLMs of different families classify every pair against its own passage as `boa`, `corrigivel` (with a grounded correction) or `irrecuperavel`, plus a 0–5 score. A pair is discarded only when both judges call it irrecoverable; otherwise judge A's correction is preferred.

**`clean_qa.py`** drops malformed pairs, questions that name the norm, questions under 5 words, answers under 8 words and non-Portuguese text; rebuilds each answer's `Fonte:` line from the pair's source citation; removes near-duplicates (bge-m3 cosine ≥ 0.92 on question + answer, longest answer kept); drops pairs whose question is close (≥ 0.90) to a gold evaluation question. The report has per-pair Shannon entropy, MTLD and FRE-PT readability, top-100 TF-IDF term density, token statistics and vocabulary size (`--compare old.jsonl` adds another set's metrics and the vocabulary Jaccard).

**`build_rag_problems.py`** writes each pair as the bot's exact prompt (header, `[citation]` blocks, `Pergunta:`; identical to `rag_bot/app/rag.py`) with 2–3 passages: the gold passage plus the highest-ranked other passages from the bot's retriever. It adds refusal problems: 12% with only unrelated passages from other resolutions, 4% with none.

**`generate_cot.py`** asks the SLM for the reasoning that leads to the known answer (relevant passage, key quote, conditions, conclusion) in Portuguese and accepts it only if it has ≥ 200 characters, is Portuguese, has no tag/`\boxed` markup and, for answer problems, names the gold article/considerando/item. Output rows are `<Thought>…</Thought>\n<Output>…</Output>` Alpaca examples, split train/val by Q&A. Every attempt is logged to `cot_raw.jsonl`; reruns skip logged problems.

## Shipped data

`data/` keeps the snapshot used in the thesis (everything else there is ignored by git):

- `passages.jsonl`: the knowledge base indexed by the bot; the evaluation gold set refers to its passage ids.
- `cot_alpaca_train.jsonl` (714) and `cot_alpaca_val.jsonl` (46): the training data.

Reports of that run are in [`../results/data/`](../results/data/). `legal_documents/` holds the 11 source resolutions (see its README).
