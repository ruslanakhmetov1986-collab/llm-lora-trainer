# llm-lora-trainer

Переиспользуемый образ для LoRA/QLoRA дообучения на Runpod: PyTorch 2.8, CUDA 12.8, transformers, peft, accelerate, datasets, hf_transfer, bitsandbytes; `/opt/trainer/train_peft.py` и `/opt/trainer/doctor.sh`.

- Образ: `ghcr.io/ruslanakhmetov1986-collab/llm-lora-trainer:latest` (и `:<номер сборки>`), собирается GitHub Actions при push в main.
- База: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (как шаблон runpod-torch-v280), SSH работает как обычно.
- Данные и ключи в образ не кладутся: датасет копируется в под, модели качаются с Hugging Face или лежат на Global Volume.
- Использование и уроки: скилл `llm-finetune`, проект-источник `~/repo/asr-lab-mac` (`tools/lora`, `tools/runpod-image`).
