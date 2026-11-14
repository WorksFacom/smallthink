# Pipeline de dados

[English](README.md) | Português

Transforma as resoluções do CFM em dados de treinamento com cadeias de raciocínio (CoT) em português. Todos os modelos são executados localmente pelo Ollama (até 10 bilhões de parâmetros); os scripts o acessam em `host.docker.internal:11434`.

```
ingest → (rag_bot: indexar trechos) → qa → judge → clean → problems → cot
```

| Etapa | Serviço / script | Saída em `data/` |
|---|---|---|
| Segmentação | `ingest` / `ingest_cfm.py` | `passages.jsonl`, `manifest.json`, `extracted/` |
| Geração de perguntas e respostas | `qa` / `generate_qa.py` | `qa_pairs.jsonl` |
| Avaliação por dois SLMs | `judge` / `judge_qa.py` | `qa_pairs_reviewed.jsonl`, `qa_pairs_clean.jsonl` |
| Filtros, deduplicação, verificação de contaminação, métricas do corpus | `clean` / `clean_qa.py` | `qa_pairs_final.jsonl`, `qa_pairs_rejected.jsonl`, `qa_clean_report.json` |
| Problemas de treinamento RAG | `problems` / `build_rag_problems.py` | `rag_problems.jsonl` |
| CoT em português | `cot` / `generate_cot.py` | `cot_alpaca_{train,val}.jsonl`, `cot_alpaca_report.json`, `cot_raw.jsonl` |

```bash
ollama pull qwen3:4b-instruct; ollama pull gemma4; ollama pull qwen3.5:9b; ollama pull bge-m3
docker compose build
docker compose --profile ingest run --rm ingest
# indexar data/passages.jsonl com ../rag_bot (necessário para `problems`)
docker compose --profile qa run --rm qa --include-kinds artigo,item --pairs-per-passage 3 --no-resume
docker compose --profile qa run --rm qa --include-kinds considerando,ementa --pairs-per-passage 1
docker compose --profile judge run --rm judge
docker compose --profile clean run --rm clean
docker compose --profile problems run --rm problems
docker compose --profile cot run --rm cot
```

Argumentos depois do nome do serviço são acrescentados ao seu comando. Modelos: `QA_MODEL` (padrão `qwen3:4b-instruct`), `JUDGE_MODEL_A` / `JUDGE_MODEL_B` (`gemma4`, `qwen3.5:9b`), `COT_MODEL` (`qwen3.5:9b`). A etapa `clean` lê o conjunto de referência da avaliação em `../evaluation/questions/gold.jsonl` para verificar a contaminação.

## Etapas

**`ingest_cfm.py`** segmenta cada PDF de `legal_documents/` em trechos dos tipos `ementa`, `considerando`, `artigo`, `item` (itens em algarismos romanos dos capítulos do Código de Ética sem artigos), `assinatura` e `exposicao_motivos`, cada um com uma citação canônica (por exemplo, `Resolução CFM nº 2217/2018, Art. 73`). Os cabeçalhos de capítulo encerram o artigo anterior e são prefixados a todos os artigos do capítulo como `context` (`Capítulo IX – SIGILO PROFISSIONAL. É vedado ao médico:`); artigos com mais de 1.500 caracteres são divididos em incisos e parágrafos, em partes que repetem o caput; cabeçalhos e rodapés das páginas são removidos; trechos repetidos entre resoluções recebem `duplicate_of`.

**`generate_qa.py`** segue a metodologia de construção de corpus de Athaydes et al. (2025): tipos variados de pergunta (regra, caso prático, permissão ou proibição, condições e exceções, definição), perguntas autocontidas que nunca mencionam a resolução ou o artigo, tom técnico, respostas terminadas por `Fonte: <citação>`, amostragem com temperatura 0,8, penalidade de frequência 0,5 e penalidade de presença 0,8. Artigos que apenas aprovam o Código, determinam a entrada em vigor ou revogam outras normas são ignorados.

**`judge_qa.py`** faz dois SLMs de famílias diferentes classificarem cada par, com base no seu próprio trecho, como `boa`, `corrigivel` (com uma correção fundamentada no trecho) ou `irrecuperavel`, além de uma nota de 0 a 5. Um par só é descartado quando os dois avaliadores o consideram irrecuperável; nos demais casos, a correção do avaliador A tem preferência.

**`clean_qa.py`** descarta pares malformados, perguntas que mencionam a norma, perguntas com menos de 5 palavras, respostas com menos de 8 palavras e textos fora do português; reconstrói a linha `Fonte:` de cada resposta a partir da citação do trecho de origem; remove quase duplicatas (cosseno do bge-m3 ≥ 0,92 sobre pergunta + resposta, mantendo a resposta mais longa); descarta pares cuja pergunta é próxima (≥ 0,90) de uma pergunta do conjunto de referência. O relatório traz, por par, a entropia de Shannon, o MTLD e a legibilidade FRE-PT, além da densidade dos 100 termos de maior TF-IDF, estatísticas de tokens e tamanho do vocabulário (`--compare antigo.jsonl` acrescenta as métricas de outro conjunto e o Jaccard dos vocabulários).

**`build_rag_problems.py`** escreve cada par exatamente no prompt do bot (cabeçalho, blocos `[citação]`, `Pergunta:`; idêntico a `rag_bot/app/rag.py`) com 2 ou 3 trechos: o trecho de referência e os outros trechos mais bem classificados pelo mecanismo de recuperação do bot. Acrescenta problemas de recusa: 12% apenas com trechos não relacionados, de outras resoluções, e 4% sem nenhum trecho.

**`generate_cot.py`** pede ao SLM o raciocínio que leva à resposta conhecida (trecho relevante, citação literal, condições, conclusão) em português e só o aceita se tiver pelo menos 200 caracteres, estiver em português, não contiver marcações como tags ou `\boxed` e, nos problemas com resposta, mencionar o artigo, considerando ou item de referência. As linhas de saída são exemplos Alpaca no formato `<Thought>…</Thought>\n<Output>…</Output>`, separados em treino e validação por par de origem. Cada tentativa é registrada em `cot_raw.jsonl`; novas execuções ignoram os problemas já registrados.

## Dados incluídos

`data/` mantém a versão usada no trabalho (os demais arquivos ali são ignorados pelo git):

- `passages.jsonl`: a base de conhecimento indexada pelo bot; o conjunto de referência da avaliação usa os identificadores desses trechos.
- `cot_alpaca_train.jsonl` (714) e `cot_alpaca_val.jsonl` (46): os dados de treinamento.

Os relatórios dessa execução estão em [`../results/data/`](../results/data/). `legal_documents/` contém as 11 resoluções de origem (ver o README da pasta).
