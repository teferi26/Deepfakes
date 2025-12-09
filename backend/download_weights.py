#!/usr/bin/env python3
"""
Script para descargar los pesos pre-entrenados del modelo UniversalFakeDetect.

Los pesos oficiales están disponibles en:
https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/main/pretrained_weights

Este script descarga fc_weights.pth que es el clasificador lineal entrenado
sobre CLIP ViT-L/14 para detectar imágenes sintéticas.
"""

import os
import sys
import urllib.request
from pathlib import Path

# URL del archivo de pesos (GitHub raw)
WEIGHTS_URL = "https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth"

# Directorio destino
WEIGHTS_DIR = Path(__file__).parent / "app" / "detectors" / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "fc_weights.pth"


def download_weights():
    """Descarga los pesos del modelo."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if WEIGHTS_FILE.exists():
        print(f"✓ Los pesos ya existen en {WEIGHTS_FILE}")
        return
    
    print(f"Descargando pesos desde {WEIGHTS_URL}...")
    print(f"Destino: {WEIGHTS_FILE}")
    
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)
        file_size = os.path.getsize(WEIGHTS_FILE)
        print(f"✓ Descarga completada ({file_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"✗ Error descargando pesos: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_weights()
