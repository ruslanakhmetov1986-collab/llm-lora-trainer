# Переиспользуемый образ для LoRA/QLoRA дообучения на Runpod (любые проекты).
# База совпадает с официальным шаблоном runpod-torch-v280: SSH и Jupyter работают как обычно.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

COPY requirements.txt /opt/trainer/requirements.txt
RUN pip install -r /opt/trainer/requirements.txt && pip freeze > /opt/trainer/requirements.lock

COPY trainer/ /opt/trainer/
COPY doctor.sh /opt/trainer/doctor.sh
RUN chmod +x /opt/trainer/doctor.sh && python -c "import transformers, peft, accelerate, datasets; print('ok')"

LABEL org.opencontainers.image.title="llm-lora-trainer" \
      org.opencontainers.image.description="PyTorch 2.8 CUDA 12.8 + transformers/peft/accelerate/datasets + train_peft.py для LoRA на Runpod"
# CMD не переопределяем: /start.sh базового образа поднимает SSH.
