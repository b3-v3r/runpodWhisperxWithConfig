# Use CUDA base image (остается как у тебя)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set Work Directory
WORKDIR /app

# ARGs and ENVs
ARG WHISPER_MODEL=small
ARG LANG=en
ARG TORCH_HOME=/cache/torch
ARG HF_HOME=/cache/huggingface

# Environment variables
ENV TORCH_HOME=${TORCH_HOME}
ENV HF_HOME=${HF_HOME}
ENV WHISPER_MODEL=${WHISPER_MODEL}
ENV LANG=${LANG}
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=True
ENV DEBIAN_FRONTEND=noninteractive

# Установка Python 3.9 и зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash ca-certificates curl file git ffmpeg \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-venv python3.9-distutils && \
    ln -s /usr/bin/python3.9 /usr/bin/python && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Создание виртуального окружения
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Установка зависимостей + совместимые версии PyTorch/pyannote
# Установка старого pip/setuptools/wheel + совместимых версий
RUN pip install --no-cache-dir --upgrade pip==21.3.1 setuptools==58.0.4 wheel==0.37.1 && \
    pip install \
        setuptools-rust==1.8.0 \
        huggingface_hub==0.18.0 \
        runpod==1.3.0 && \
    pip install torch==1.10.0+cu113 torchaudio==0.10.0+cu113 \
        -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    pip install pyannote.audio==0.0.1

RUN apt-get install -y ffmpeg libsndfile1
RUN pip install ffmpeg-python soundfile

# Установка WhisperX
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git

# Копирование файлов проекта
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true

COPY example.mp3 /app/example.mp3
COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

# Старт
STOPSIGNAL SIGINT
CMD ["python", "-u", "handler.py"]
