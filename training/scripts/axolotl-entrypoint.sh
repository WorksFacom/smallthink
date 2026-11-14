#!/bin/bash
# Axolotl image entrypoint — avoid transformers→torchvision import bomb.
# Broken torchvision (torch ABI / CUDA mismatch) surfaces as:
#   ModuleNotFoundError: Could not import module 'PreTrainedModel'
set -euo pipefail

if python -c "import torchvision" 2>/dev/null; then
  if ! python -c "import torch; import torchvision; from torchvision.ops import nms" 2>/dev/null; then
    echo "[entrypoint] torchvision broken vs torch — uninstall (LLM train no need vision ops)"
    pip uninstall -y torchvision 2>/dev/null || true
  fi
fi

exec "$@"
