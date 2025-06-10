# Use CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

# Install system dependencies and Python 3.10
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
        python3.10 \
        python3.10-venv \
        python3.10-distutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy handler and any other needed files
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-u", "handler.py"]
