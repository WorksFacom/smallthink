# Avaliação

[English](README.md) | Português

Compara o modelo ajustado com o Qwen3-4B original e com o gpt-oss-120B nas mesmas perguntas, com os mesmos trechos recuperados e o mesmo prompt. Modelos de grande porte são usados apenas aqui.

```
build_gold.py → questions/ ─┐
compare.py ─────→ answers/ ─┴→ build_eval_set.py → judge_sets/ → judge.py → judgments/
                                     score.py, bertscore_eval.py → results/
```

```bash
cp config.example.yml config.yml && cp .env.example .env    # GROQ_API_KEY, GEMINI_API_KEY
docker compose build
docker compose run --rm evaluation                           # compare.py config.yml
docker compose run --rm evaluation python score.py           # qualquer script
```

Sem Docker: `pip install -r requirements.txt` e executar os scripts a partir deste diretório.

## Conjunto de referência — `build_gold.py`

Constrói o conjunto de avaliação com o Gemini (`gemini-3.1-pro-preview`) em três etapas retomáveis (`--stage generate|draft|finalize|all`):

1. **generate**: perguntas respondíveis por resolução, escritas a partir dos trechos da própria resolução; perguntas não respondíveis escritas apenas a partir da lista de temas (`near_miss`: tema coberto, mas sobre um detalhe que as normas não definem; `out_of_scope`: fora do escopo). São descartadas perguntas que mencionam uma norma, que se repetem ou que se parecem com um par de treinamento (Jaccard dos conjuntos de palavras ≥ 0,6 em relação a `../data_pipeline/data/qa_pairs_final.jsonl`).
2. **draft**: o registro de referência de cada pergunta é redigido independentemente de como ela foi escrita: a pergunta é direcionada a até 3 resoluções e o registro é redigido a partir de todos os seus trechos, com `passage_id` restrito pelo esquema de resposta; as citações precisam ser trechos literais e os pontos-chave precisam ser fundamentados; falhas são refeitas com a lista de erros. `--batch` usa a Batch API.
3. **finalize**: 180 perguntas respondíveis + 20 não respondíveis (metade de cada tipo), com "Não" limitado a 60% das respostas binárias.

Gera `questions/gold.jsonl`, `questions/questions.json` (entrada do `compare.py`) e `questions/gold_report.json` (com `review_first`). Cada registro de referência tem `question_type`, `answerable`, `short_answer`, `reference`, `key_points`, `gold_passages`, `gold_citation`, `evidence` (citações literais), `must_not` e `unanswerable_kind`.

## Respostas — `compare.py`

Envia cada pergunta a cada API de `config.yml` e grava `answers/<nome>.jsonl`; linhas já bem-sucedidas são ignoradas em novas execuções. Tipos: `bot` (a API do `rag_bot`), `ollama` (API de chat, com `think` e `options`), `groq` (`params` como `reasoning_effort` e `max_completion_tokens`) e `gemini`. Com `rag_from: answers/finetuned-rag.jsonl`, uma linha de base recebe, para cada pergunta, exatamente os trechos recuperados pelo bot, no prompt do bot; as perguntas que o bot recusou por falta de trechos são recusadas da mesma forma. O raciocínio é guardado em `thought`; uma resposta interrompida pelo limite de tokens é registrada como erro.

## Avaliador — `build_eval_set.py`, `judge.py`

`build_eval_set.py answers/<nome>.jsonl` combina as respostas com o conjunto de referência em itens cegos para o avaliador (`judge_sets/<nome>.json`: pergunta, registro de referência, resposta final sem o raciocínio, trechos; sem o nome do modelo). `judge.py judge_sets/*.json [--batch]` envia um item por requisição ao Gemini (temperatura 0), com a rubrica como instrução de sistema e um esquema de resposta; julgamentos que violam as regras da própria rubrica são refeitos com a lista de erros. Saída: `judgments/<nome>.jsonl` com correção (0–2), rótulos dos pontos-chave, concordância da conclusão, contradição, abstenção, correspondência e situação das citações, violações de `must_not`, fidelidade (0–2), idioma e menção a legislação estrangeira.

## Métricas — `score.py`

Grava `results/scores.json` (geral e por respondibilidade, tipo de pergunta, resolução e tipo de pergunta não respondível) e `results/flags.jsonl`. Métricas automáticas: verificação léxica de recusa, recuperação (trecho de referência entre os recuperados), conclusão na primeira palavra comparada à resposta curta de referência, citação extraída da resposta comparada à citação de referência. Métricas do avaliador: correção (com os limites determinísticos da rubrica), cobertura de pontos-chave, concordância da conclusão, abstenção, citações, fidelidade, contradições. As flags listam divergências entre o avaliador e as verificações automáticas, para revisão manual.

## BERTScore — `bertscore_eval.py`

Métrica secundária, independente do avaliador: cada resposta final comparada à `reference`, após remover dos dois lados o raciocínio, a formatação e as linhas `Fonte:`; `--key-points` acrescenta a revocação de cada ponto-chave de referência; `--baseline` acrescenta diferenças pareadas com bootstrap; correlação de Spearman com a correção atribuída pelo avaliador quando há julgamentos.

```bash
mkdir -p .hf_cache
docker compose --profile bertscore build bertscore           # wheels cu126 (Pascal a Hopper)
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 docker compose --profile bertscore build bertscore   # Blackwell
docker compose --profile bertscore run --rm bertscore --systems finetuned-rag qwen3-4b-rag gpt-oss-120b-rag \
  --baseline finetuned-rag --key-points --summary results/bertscore_xlmr.json --items results/bertscore_xlmr_items.jsonl
docker compose --profile bertscore run --rm bertscore --systems finetuned-rag qwen3-4b-rag gpt-oss-120b-rag \
  --baseline finetuned-rag --key-points --model rufimelo/Legal-BERTimbau-base --num-layers 9 \
  --summary results/bertscore_legal_bertimbau.json --items results/bertscore_legal_bertimbau_items.jsonl
```

Os valores não são reescalonados; compare sistemas apenas dentro de um mesmo modelo de base. O BERTScore não percebe negações nem citações incorretas.

## Dados incluídos

- `questions/`: o conjunto de referência com 200 perguntas, a lista de perguntas e o relatório de construção.
- `answers/`: respostas do `finetuned-rag` (Qwen3-4B ajustado, Q5_K_M, via `rag_bot`), do `qwen3-4b-rag` (Qwen3-4B original, Q5_K_M, em modo de raciocínio; 6 de 201 linhas são erros por interrupção) e do `gpt-oss-120b-rag` (Groq). Os modelos locais foram executados em uma Quadro P1000.
- `judgments/`: os julgamentos do Gemini sobre essas respostas.
- `results/`: `scores.json`, `flags.jsonl` e BERTScore com XLM-RoBERTa large e Legal-BERTimbau base.
