# Relatórios do pipeline e do treinamento

[English](README.md) | Português

Relatórios da execução usada no trabalho. Os resultados da avaliação estão em [`../evaluation/results/`](../evaluation/results/).

| Arquivo | Conteúdo |
|---|---|
| `data/judge_summary.json` | Avaliação por dois SLMs (gemma4, qwen3.5:9b) dos 949 pares gerados: 755 mantidos, 144 corrigidos, 50 descartados; concordância entre as categorias e notas médias |
| `data/qa_clean_report.json` | Limpeza dos 899 pares restantes (42 perguntas que mencionavam a norma e 114 quase duplicatas removidas, 857 linhas de fonte reconstruídas → 743 pares), contagens por resolução e métricas do corpus |
| `data/cot_report.json` | Geração do CoT: 862 problemas, 760 aceitos (642 com resposta, 88 recusas com trechos não relacionados, 30 recusas sem trechos), 714 de treino / 46 de validação |
| `training/trainer_state.json` | Registro do treinamento (Axolotl/Transformers): perdas, perda de validação por época, memória |
