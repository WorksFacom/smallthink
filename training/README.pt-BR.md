# Treinamento

[English](README.md) | Português

Ajuste fino QLoRA do `Qwen/Qwen3-4B` com os dados de CoT em português, usando o [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), seguido da exportação para o Ollama.

## Treinar

```bash
mkdir -p data
cp ../data_pipeline/data/cot_alpaca_train.jsonl ../data_pipeline/data/cot_alpaca_val.jsonl data/
docker compose pull train
docker compose run --rm train        # axolotl train configs/qwen3_4b_cfm_cot.yml
```

Requer o NVIDIA Container Toolkit. A imagem `axolotlai/axolotl:main-py3.11-cu128-2.9.1` (CUDA 12.8, PyTorch 2.9.1) é necessária para GPUs Blackwell (driver ≥ 570 no host); outra imagem pode ser definida com `AXOLOTL_IMAGE`.

[`configs/qwen3_4b_cfm_cot.yml`](configs/qwen3_4b_cfm_cot.yml): modelo base carregado em 4 bits NF4 com dupla quantização e cálculos em BF16; LoRA de posto 16, alfa 32 e dropout 0,05 nas projeções `q/k/v/o/gate/up/down` (33 milhões de parâmetros treináveis, 0,81%); formato Alpaca com perda calculada apenas sobre a saída; `sequence_len` de 2048; micro-lote de 4 × acumulação de gradiente de 4 (lote efetivo de 16); 2 épocas; taxa de aprendizado 2e-4, agendamento cosseno, 5% de aquecimento; AdamW de 8 bits; gradient checkpointing; atenção SDPA do PyTorch. Saída: `outputs/qwen3-4b-cfm-cot/`.

Na RTX 5050 (8 GB), o treinamento levou cerca de 38 minutos (90 passos), com pico de 6,1 GB alocados; a perda de validação caiu de 1,27 para 0,42. O registro está em [`../results/training/trainer_state.json`](../results/training/trainer_state.json).

Se faltar memória: `micro_batch_size: 2` com `gradient_accumulation_steps: 8`, depois `sequence_len: 1536`, depois `lora_r: 8` / `lora_alpha: 16`.

## Exportar para o Ollama

```bash
docker compose run --rm train python export_ollama.py
cd outputs/ollama-qwen3-cfm-cot && ollama create qwen3-4b-cfm-cot -f Modelfile
```

`export_ollama.py` mescla o adaptador ao modelo base em BF16 (na CPU, a menos que se use `--gpu`), converte para GGUF com o `convert_hf_to_gguf.py` do llama.cpp, quantiza (`--quant`, padrão `q5_k_m`; requer `llama-quantize` no PATH, caso contrário mantém o GGUF em F16) e escreve um `Modelfile`. O Modelfile usa o mesmo prompt Alpaca do treinamento, sem instrução de sistema, e interrompe a geração em `</Output>`: um template de conversa ou uma instrução de sistema fazem o modelo voltar ao comportamento do modelo base. Opções: `--adapter`, `--out-dir`, `--ollama-name`, `--num-ctx` (4096), `--skip-merge`, `--skip-gguf`, `--create`.

Sem Docker: `pip install -r requirements-export.txt` e uma versão do PyTorch compatível com a sua GPU.
