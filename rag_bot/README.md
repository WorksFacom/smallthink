# RAG bot

English | [Português](README.pt-BR.md)

Answers a question with the fine-tuned model over retrieved CFM passages:

```
question → bge-m3 embedding → Qdrant (top 3 passages, score ≥ 0.56) → prompt → Ollama → <Thought> / <Output>
```

The prompt (fixed header, `[citation]` blocks, `Pergunta:`) is identical to the training problems built by `data_pipeline/build_rag_problems.py`. When no passage clears `MIN_SCORE`, the bot returns the fixed refusal without calling the model (`abstention: "no_sources"`; `ABSTAIN_WITHOUT_SOURCES=0` disables it). Generation stops at `</Output>`; the reasoning is returned separately as `thought`.

## Run

Needs Ollama on the host with `bge-m3` and the exported model (`ollama create qwen3-4b-cfm-cot -f Modelfile`, see `../training`), and a Qdrant instance (e.g. `docker run -d -p 6333:6333 qdrant/qdrant`).

```bash
cp .env.example .env
mkdir -p data && cp ../data_pipeline/data/passages.jsonl data/
docker compose --profile index run --rm index     # embeds ementa/considerando/artigo/item, skips duplicates, recreates the collection
docker compose run --rm chat                      # terminal REPL: /sources, /thought, /k N, /quit
docker compose up -d api                          # batch API on :8000
```

## API

`POST /v1/answers` takes `{"questions": [...], "top_k": 3}` or a raw array; each item is a string or `{id, question}`.

```bash
curl -s http://localhost:8000/v1/answers -H 'Content-Type: application/json' \
  -d '{"questions":[{"id":"q1","question":"O médico pode revelar o diagnóstico a familiares sem autorização do paciente?"}]}'
```

Each result has `id`, `question`, `answer` (text inside `<Output>`), `thought`, `abstention`, `sources` (`passage_id`, `citation`, `kind`, `score`, `text`) and `elapsed_seconds`. A failing item gets `error` without dropping the others. Requests run one at a time.

## Settings

| Env | Default | Notes |
|---|---|---|
| `GEN_MODEL` | `qwen3-4b-cfm-cot` | Ollama model name |
| `EMBED_MODEL` | `bge-m3` | Same model at index and query time |
| `COLLECTION` | `cfm_passages` | Qdrant collection |
| `TOP_K` | `3` | Training problems show 2–3 passages |
| `MIN_SCORE` | `0.56` | Passages below are dropped |
| `ABSTAIN_WITHOUT_SOURCES` | `1` | Refuse without calling the model when no passage is left |
| `TEMPERATURE` | `0.2` | |
| `NUM_CTX` / `NUM_PREDICT` | `4096` / `1536` | Prompt + reasoning + answer must fit |
| `QDRANT_URL`, `OLLAMA_URL` | `host.docker.internal` | |

Without Docker: `pip install -r requirements.txt`, then `OLLAMA_URL=http://localhost:11434 QDRANT_URL=http://localhost:6333 python chat.py`.
