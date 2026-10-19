#!/bin/bash
# Проверка окружения пода: GPU, CUDA, версии библиотек, свободное место.
set -e
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "GPU не найдена"
python - <<'PY'
import torch, transformers, peft, accelerate, datasets
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("bf16", torch.cuda.is_bf16_supported(), "device", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "peft", peft.__version__, "accelerate", accelerate.__version__, "datasets", datasets.__version__)
PY
df -h /workspace /workspace-global 2>/dev/null || df -h /
cat /opt/trainer/VERSION 2>/dev/null
