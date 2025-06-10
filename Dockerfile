# Use CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 1) Устанавливаем системные зависимости и Python 3.8
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ffmpeg \
        git \
        curl \
        ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-venv \
        python3.8-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2) Создаём и активируем виртуальное окружение на Python 3.8
RUN python3.8 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# 3) Апгрейд pip и установка зависимостей из requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 4) Копируем ваш код
COPY handler.py /app/handler.py

# 5) Флаги для корректного вывода в логи
ENV PYTHONUNBUFFERED=1

# 6) Запуск
CMD ["python", "-u", "handler.py"]
