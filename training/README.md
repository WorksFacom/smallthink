# Training

English | [Português](README.pt-BR.md)

QLoRA fine-tuning of `Qwen/Qwen3-4B` on the Portuguese CoT data with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), then export to Ollama.

## Train

```bash
mkdir -p data
cp ../data_pipeline/data/cot_alpaca_train.jsonl ../data_pipeline/data/cot_alpaca_val.jsonl data/
docker compose pull train
docker compose run --rm train        # axolotl train configs/qwen3_4b_cfm_cot.yml
```

Needs the NVIDIA Container Toolkit. The image `axolotlai/axolotl:main-py3.11-cu128-2.9.1` (CUDA 12.8, PyTorch 2.9.1) is required for Blackwell GPUs (host driver ≥ 570); override with `AXOLOTL_IMAGE`.

[`configs/qwen3_4b_cfm_cot.yml`](configs/qwen3_4b_cfm_cot.yml): base loaded in 4-bit NF4 with double quantization, BF16 compute; LoRA rank 16, alpha 32, dropout 0.05 on `q/k/v/o/gate/up/down` projections (33M trainable parameters, 0.81%); Alpaca format with loss on the output only; `sequence_len` 2048; micro-batch 4 × gradient accumulation 4 (effective 16); 2 epochs; LR 2e-4, cosine, 5% warmup; AdamW 8-bit; gradient checkpointing; PyTorch SDPA attention. Output: `outputs/qwen3-4b-cfm-cot/`.

On the RTX 5050 (8 GB) the run took about 38 minutes (90 steps), peaking at 6.1 GB allocated; validation loss went from 1.27 to 0.42. The log is in [`../results/training/trainer_state.json`](../results/training/trainer_state.json).

If it runs out of memory: `micro_batch_size: 2` with `gradient_accumulation_steps: 8`, then `sequence_len: 1536`, then `lora_r: 8` / `lora_alpha: 16`.

## Export to Ollama

```bash
docker compose run --rm train python export_ollama.py
cd outputs/ollama-qwen3-cfm-cot && ollama create qwen3-4b-cfm-cot -f Modelfile
```

`export_ollama.py` merges the adapter into the base in BF16 (on CPU unless `--gpu`), converts it to GGUF with llama.cpp's `convert_hf_to_gguf.py`, quantizes it (`--quant`, default `q5_k_m`; needs `llama-quantize` in PATH, else the F16 GGUF is kept) and writes a `Modelfile`. The Modelfile uses the same Alpaca prompt as training, with no system prompt, and stops at `</Output>`: a chat template or system prompt makes the model fall back to the base model's behavior. Options: `--adapter`, `--out-dir`, `--ollama-name`, `--num-ctx` (4096), `--skip-merge`, `--skip-gguf`, `--create`.

Without Docker: `pip install -r requirements-export.txt` plus a PyTorch build for your GPU.
