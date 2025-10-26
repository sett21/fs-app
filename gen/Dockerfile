# База с CUDA/cuDNN и PyTorch
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

# Нужны системные либы: OpenCV + инструменты сборки для insightface (g++, make, cmake)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    build-essential g++ gcc make cmake \
    && rm -rf /var/lib/apt/lists/*

# Подготовим pip окружение: свежие pip/setuptools/wheel/cython и numpy (хедеры для компиляции)
RUN pip install --upgrade pip setuptools wheel cython ninja "numpy==1.26.4"

# Устанавливаем зависимости. Оставь insightface в requirements.txt — теперь он соберётся.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8081
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8081"]
