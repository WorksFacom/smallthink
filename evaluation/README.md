# Evaluation

English | [Português](README.pt-BR.md)

Compares the fine-tuned model with the original Qwen3-4B and gpt-oss-120B on the same questions, the same retrieved passages and the same prompt. Large models are used here only.

```
build_gold.py → questions/ ─┐
compare.py ─────→ answers/ ─┴→ build_eval_set.py → judge_sets/ → judge.py → judgments/
                                     score.py, bertscore_eval.py → results/
```

```bash
cp config.example.yml config.yml && cp .env.example .env    # GROQ_API_KEY, GEMINI_API_KEY
docker compose build
docker compose run --rm evaluation                           # compare.py config.yml
docker compose run --rm evaluation python score.py           # any script
```

Without Docker: `pip install -r requirements.txt` and run the scripts from this directory.

## Gold set — `build_gold.py`

Builds the evaluation set with Gemini (`gemini-3.1-pro-preview`) in three resumable stages (`--stage generate|draft|finalize|all`):

1. **generate**: answerable questions per resolution, written from that resolution's passages; unanswerable questions from the topic list only (`near_miss`: a covered topic, a detail the norms do not set; `out_of_scope`). Questions that name a norm, repeat each other, or resemble a training Q&A (word-set Jaccard ≥ 0.6 against `../data_pipeline/data/qa_pairs_final.jsonl`) are dropped.
2. **draft**: each gold record drafted independently of how the question was written: routed to up to 3 resolutions, drafted from all their passages with `passage_id` restricted by the response schema; quotes must be exact substrings and key points supported; failures are retried with the errors. `--batch` uses the Batch API.
3. **finalize**: 180 answerable + 20 unanswerable (half of each kind), "Não" capped at 60% of binary answers.

Outputs `questions/gold.jsonl`, `questions/questions.json` (input of `compare.py`) and `questions/gold_report.json` (with `review_first`). Each gold record has `question_type`, `answerable`, `short_answer`, `reference`, `key_points`, `gold_passages`, `gold_citation`, `evidence` (exact quotes), `must_not` and `unanswerable_kind`.

## Answers — `compare.py`

Sends every question to each API in `config.yml` and writes `answers/<name>.jsonl`; succeeded rows are skipped on reruns. Types: `bot` (the `rag_bot` API), `ollama` (chat API, `think` and `options`), `groq` (`params` such as `reasoning_effort`, `max_completion_tokens`) and `gemini`. With `rag_from: answers/finetuned-rag.jsonl`, a baseline gets, for each question, exactly the passages the bot retrieved in the bot's prompt; questions the bot refused for lack of passages are refused the same way. Reasoning is stored as `thought`; an answer cut off by the token limit is an error row.

## Judge — `build_eval_set.py`, `judge.py`

`build_eval_set.py answers/<name>.jsonl` pairs answers with the gold set into blind judge items (`judge_sets/<name>.json`: question, gold record, final answer without reasoning, sources; no model name). `judge.py judge_sets/*.json [--batch]` sends one item per request to Gemini (temperature 0) with the rubric as system instruction and a response schema; judgments that break the rubric's own rules are judged again with the errors. Output: `judgments/<name>.jsonl` with correctness (0–2), key-point labels, short-answer match, contradiction, abstention, citation match and status, `must_not` violations, faithfulness (0–2), language and foreign-law flags.

## Scores — `score.py`

Writes `results/scores.json` (overall and by answerability, question type, resolution and unanswerable kind) and `results/flags.jsonl`. Automatic metrics: lexical refusal check, retrieval (gold passage among the retrieved ones), first-word verdict against the gold short answer, parsed citation against the gold citation. Judge metrics: correctness (with the rubric's deterministic caps), key-point coverage, short-answer match, abstention, citations, faithfulness, contradictions. Flags list disagreements between judge and automatic checks to review by hand.

## BERTScore — `bertscore_eval.py`

Secondary, judge-independent metric: each final answer against the gold `reference`, after removing reasoning, markup and `Fonte:` lines from both sides; `--key-points` adds the recall of each gold key point; `--baseline` adds paired bootstrap differences; Spearman correlation with the judge's correctness when judgments exist.

```bash
mkdir -p .hf_cache
docker compose --profile bertscore build bertscore           # cu126 wheels (Pascal–Hopper)
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 docker compose --profile bertscore build bertscore   # Blackwell
docker compose --profile bertscore run --rm bertscore --systems finetuned-rag qwen3-4b-rag gpt-oss-120b-rag \
  --baseline finetuned-rag --key-points --summary results/bertscore_xlmr.json --items results/bertscore_xlmr_items.jsonl
docker compose --profile bertscore run --rm bertscore --systems finetuned-rag qwen3-4b-rag gpt-oss-120b-rag \
  --baseline finetuned-rag --key-points --model rufimelo/Legal-BERTimbau-base --num-layers 9 \
  --summary results/bertscore_legal_bertimbau.json --items results/bertscore_legal_bertimbau_items.jsonl
```

Scores are raw (no baseline rescaling); compare systems only within one backbone. BERTScore does not see negation or wrong citations.

## Shipped data

- `questions/`: the 200-question gold set, the question list and the build report.
- `answers/`: answers of `finetuned-rag` (fine-tuned Qwen3-4B, Q5_K_M, via `rag_bot`), `qwen3-4b-rag` (original Qwen3-4B, Q5_K_M, thinking mode; 6 of 201 rows are cut-off errors) and `gpt-oss-120b-rag` (Groq). Local models ran on a Quadro P1000.
- `judgments/`: Gemini judgments of those answers.
- `results/`: `scores.json`, `flags.jsonl` and BERTScore with XLM-RoBERTa large and Legal-BERTimbau base.
