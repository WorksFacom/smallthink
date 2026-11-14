# Bot RAG

[English](README.md) | Português

Responde a uma pergunta com o modelo ajustado, a partir de trechos recuperados das normas do CFM:

```
pergunta → embedding bge-m3 → Qdrant (3 trechos mais similares, similaridade ≥ 0,56) → prompt → Ollama → <Thought> / <Output>
```

O prompt (cabeçalho fixo, blocos `[citação]`, `Pergunta:`) é idêntico ao dos problemas de treinamento gerados por `data_pipeline/build_rag_problems.py`. Quando nenhum trecho atinge `MIN_SCORE`, o bot devolve a recusa fixa sem consultar o modelo (`abstention: "no_sources"`; `ABSTAIN_WITHOUT_SOURCES=0` desativa esse comportamento). A geração é interrompida em `</Output>`, e o raciocínio é devolvido separadamente em `thought`.

## Execução

Requer o Ollama no host com o `bge-m3` e o modelo exportado (`ollama create qwen3-4b-cfm-cot -f Modelfile`, ver `../training`), e uma instância do Qdrant (por exemplo, `docker run -d -p 6333:6333 qdrant/qdrant`).

```bash
cp .env.example .env
mkdir -p data && cp ../data_pipeline/data/passages.jsonl data/
docker compose --profile index run --rm index     # indexa ementa/considerando/artigo/item, ignora duplicatas, recria a coleção
docker compose run --rm chat                      # REPL no terminal: /sources, /thought, /k N, /quit
docker compose up -d api                          # API em lote na porta 8000
```

## API

`POST /v1/answers` recebe `{"questions": [...], "top_k": 3}` ou uma lista simples; cada item é um texto ou `{id, question}`.

```bash
curl -s http://localhost:8000/v1/answers -H 'Content-Type: application/json' \
  -d '{"questions":[{"id":"q1","question":"O médico pode revelar o diagnóstico a familiares sem autorização do paciente?"}]}'
```

Cada resultado tem `id`, `question`, `answer` (texto dentro de `<Output>`), `thought`, `abstention`, `sources` (`passage_id`, `citation`, `kind`, `score`, `text`) e `elapsed_seconds`. Um item com falha recebe `error` sem interromper os demais. As requisições são processadas uma de cada vez.

## Configurações

| Variável | Padrão | Observações |
|---|---|---|
| `GEN_MODEL` | `qwen3-4b-cfm-cot` | Nome do modelo no Ollama |
| `EMBED_MODEL` | `bge-m3` | O mesmo modelo na indexação e na consulta |
| `COLLECTION` | `cfm_passages` | Coleção do Qdrant |
| `TOP_K` | `3` | Os problemas de treinamento têm 2 ou 3 trechos |
| `MIN_SCORE` | `0.56` | Trechos abaixo desse valor são descartados |
| `ABSTAIN_WITHOUT_SOURCES` | `1` | Recusa sem consultar o modelo quando não resta nenhum trecho |
| `TEMPERATURE` | `0.2` | |
| `NUM_CTX` / `NUM_PREDICT` | `4096` / `1536` | Prompt, raciocínio e resposta precisam caber |
| `QDRANT_URL`, `OLLAMA_URL` | `host.docker.internal` | |

Sem Docker: `pip install -r requirements.txt` e depois `OLLAMA_URL=http://localhost:11434 QDRANT_URL=http://localhost:6333 python chat.py`.
