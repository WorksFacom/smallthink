# Raciocínio com modelos de linguagem pequenos sobre as normas do CFM

[English](README.md) | Português

Código e dados do trabalho de conclusão de curso *Reasoning com Small Language Models*: um Qwen3-4B ajustado para raciocinar em português sobre trechos recuperados das resoluções do Conselho Federal de Medicina (CFM), servido com geração aumentada por recuperação (RAG) em GPUs de consumo.

Todas as etapas que produzem dados de treinamento usam modelos pequenos executados localmente pelo Ollama. Modelos de grande porte (Gemini, gpt-oss-120B) aparecem apenas na avaliação: na redação do conjunto de referência, no julgamento das respostas e como linha de base.

```
legal_documents/*.pdf ──ingestão──▶ trechos ──indexação──▶ Qdrant ◀── rag_bot (pergunta → trechos → modelo)
                                       │                                    ▲
                                       ▼                                    │
                      Q&A (qwen3:4b-instruct) → avaliação dupla (gemma4 + qwen3.5:9b)
                      → filtros, deduplicação, verificação de contaminação → problemas RAG
                      → CoT em português (qwen3.5:9b) → QLoRA (Axolotl) → GGUF → Ollama
```

## Estrutura

| Diretório | Conteúdo |
|---|---|
| [`data_pipeline/`](data_pipeline/) | Segmentação dos PDFs, geração de perguntas e respostas, avaliação por dois SLMs, limpeza e métricas do corpus, problemas de treinamento RAG, geração do CoT; PDFs de origem e a versão dos dados usada no trabalho |
| [`training/`](training/) | Configuração QLoRA do Axolotl para o Qwen3-4B e exportação LoRA → GGUF → Ollama |
| [`rag_bot/`](rag_bot/) | Indexação dos trechos (bge-m3 + Qdrant), chat no terminal e API de respostas em lote |
| [`evaluation/`](evaluation/) | Construção do conjunto de referência, coleta das respostas de cada sistema, avaliador Gemini com rubrica, métricas automáticas, BERTScore; o conjunto de 200 perguntas, as respostas, os julgamentos e os resultados |
| [`results/`](results/) | Relatórios do pipeline de dados (avaliação, limpeza, métricas do corpus, CoT) e o registro do treinamento |

## Hardware

| Etapa | GPU |
|---|---|
| Geração de perguntas e respostas, inferência dos modelos locais | NVIDIA Quadro P1000 (4 GB, Pascal) |
| Avaliadores, geração do CoT, treinamento | NVIDIA GeForce RTX 5050 (8 GB, Blackwell; CUDA ≥ 12.8) |

## Reprodução

Cada diretório tem seu próprio README e arquivo Docker Compose. Na ordem:

1. `data_pipeline`: ingerir os PDFs → `data/passages.jsonl`.
2. `rag_bot`: indexar os trechos no Qdrant (os problemas RAG usam esse índice para escolher os trechos de contraste).
3. `data_pipeline`: qa → judge → clean → problems → cot → `data/cot_alpaca_{train,val}.jsonl`.
4. `training`: copiar os dois arquivos para `training/data/`, treinar e exportar para o Ollama.
5. `rag_bot`: iniciar a API com o modelo exportado.
6. `evaluation`: coletar as respostas, montar os conjuntos para o avaliador, julgar, calcular as métricas e o BERTScore.

`data_pipeline/data/` traz a versão dos trechos e os arquivos de treinamento usados no trabalho, de modo que as etapas 4 a 6 podem ser executadas sem gerar os dados novamente.

## Resultados (200 perguntas: 180 respondíveis, 20 não respondíveis)

Todos os sistemas receberam o mesmo prompt e os mesmos três trechos recuperados por pergunta (o trecho de referência foi recuperado em 87,2% das perguntas respondíveis). Avaliador: Gemini 3.1 Pro com rubrica; detalhes em [`evaluation/results/scores.json`](evaluation/results/scores.json).

| Métrica | Qwen3-4B ajustado | Qwen3-4B original | gpt-oss-120B |
|---|---|---|---|
| Correção (0–2) | 1,47 | 1,59 | 1,65 |
| Totalmente corretas | 58% | 68% | 73% |
| Cobertura de pontos-chave | 76% | 82% | 85% |
| Conclusão na primeira palavra (binárias) | 82% | 50% | 84% |
| Fidelidade aos trechos (0–2) | 1,93 | 1,79 | 1,94 |
| Recusa nas não respondíveis / respondíveis | 100% / 4% | 100% / 3% | 100% / 5% |
| Tempo mediano por resposta | 31,8 s | 42,7 s | 6,8 s |
| BERTScore F1 (XLM-R large / Legal-BERTimbau) | 0,935 / 0,769 | 0,903 / 0,681 | 0,911 / 0,711 |

O Qwen3-4B original foi avaliado em 194 perguntas; em 6, seu raciocínio esgotou o limite de 1.536 tokens.
