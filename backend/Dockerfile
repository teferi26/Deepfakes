FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

# Instalar Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    tzdata \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Optimizaciones para PyTorch
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    # Evitar descargas durante runtime
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

# Instalar dependencias del sistema para OpenCV y librerías ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar dependencias de Python (PyTorch es grande, usar --no-cache-dir)
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorio para pesos del modelo
RUN mkdir -p /app/app/detectors/weights

# Copiar código
COPY app ./app
COPY download_weights.py .
COPY scripts ./scripts

# Pre-descargar pesos del modelo para evitar descarga en runtime
RUN python download_weights.py || echo "Weights will be downloaded on first use"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
