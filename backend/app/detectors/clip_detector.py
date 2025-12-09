"""
Detector CLIP (UniversalFakeDetect - CVPR 2023)

Detector principal basado en CLIP ViT-L/14 con clasificador lineal entrenado
para distinguir imágenes reales de sintéticas.

Paper: "Towards Universal Fake Image Detectors that Generalize Across Generative Models"
https://github.com/WisconsinAIVision/UniversalFakeDetect
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base_detector import BaseImageDetector

logger = logging.getLogger(__name__)

# Directorio para almacenar pesos del modelo
WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "clip_detector_weights.pth"

# URL para descargar pesos
WEIGHTS_URL = "https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth"

# Configuración del modelo
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"
INPUT_SIZE = 224


class CLIPLinearClassifier(nn.Module):
    """Clasificador lineal sobre features de CLIP."""
    
    def __init__(self, input_dim: int = 768, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CLIPDetector(BaseImageDetector):
    """
    Detector basado en CLIP + clasificador lineal (UniversalFakeDetect).
    
    Fortalezas:
    - Excelente generalización entre distintos modelos generativos
    - Detecta GANs (StyleGAN, ProGAN), Diffusion (SD, DALL-E), etc.
    - Robusto a post-procesamiento (JPEG, resize)
    
    Debilidades:
    - Puede fallar con imágenes muy editadas manualmente
    - No detecta manipulaciones locales (solo imagen completa)
    """
    
    name = "clip_universal"
    version = "1.0.0"
    default_weight = 0.50  # Peso alto por su excelente generalización
    
    _instance: Optional['CLIPDetector'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if CLIPDetector._initialized:
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.classifier = None
        self.preprocess = None
        self._available = False
        
        try:
            self._load_model()
            self._available = True
        except Exception as e:
            logger.error(f"Error inicializando CLIPDetector: {e}")
        
        CLIPDetector._initialized = True
    
    def _load_model(self):
        """Carga CLIP y el clasificador lineal."""
        import open_clip
        
        logger.info(f"Cargando CLIP {CLIP_MODEL_NAME} en {self.device}...")
        
        # Cargar CLIP
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            device=self.device
        )
        self.clip_model.eval()
        
        # Obtener dimensión de features
        with torch.no_grad():
            dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE).to(self.device)
            features = self.clip_model.encode_image(dummy)
            feature_dim = features.shape[-1]
        
        logger.info(f"CLIP feature dimension: {feature_dim}")
        
        # Crear clasificador
        self.classifier = CLIPLinearClassifier(input_dim=feature_dim).to(self.device)
        
        # Descargar/cargar pesos
        self._ensure_weights_exist()
        
        if WEIGHTS_FILE.exists():
            logger.info(f"Cargando pesos desde {WEIGHTS_FILE}")
            state_dict = torch.load(WEIGHTS_FILE, map_location=self.device, weights_only=True)
            
            if 'fc.weight' not in state_dict and 'weight' in state_dict:
                state_dict = {
                    'fc.weight': state_dict['weight'],
                    'fc.bias': state_dict['bias']
                }
            self.classifier.load_state_dict(state_dict)
            logger.info("Pesos CLIP cargados correctamente")
        else:
            logger.warning("No se encontraron pesos CLIP, usando clasificador sin entrenar")
            nn.init.zeros_(self.classifier.fc.weight)
            nn.init.zeros_(self.classifier.fc.bias)
        
        self.classifier.eval()
    
    def _ensure_weights_exist(self):
        """Descarga los pesos si no existen."""
        if WEIGHTS_FILE.exists():
            return
        
        # También verificar el nombre antiguo
        old_weights = WEIGHTS_DIR / "fc_weights.pth"
        if old_weights.exists():
            import shutil
            shutil.copy(old_weights, WEIGHTS_FILE)
            return
        
        import urllib.request
        
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Descargando pesos CLIP desde {WEIGHTS_URL}...")
        
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)
            logger.info("Pesos descargados correctamente")
        except Exception as e:
            logger.warning(f"No se pudieron descargar pesos: {e}")
    
    def is_available(self) -> bool:
        return self._available
    
    @torch.no_grad()
    def analyze(self, image_data: bytes) -> Dict[str, Any]:
        """Analiza una imagen con CLIP."""
        if not self._available:
            raise RuntimeError("CLIPDetector no está disponible")
        
        try:
            # Cargar imagen
            image = Image.open(BytesIO(image_data))
            original_size = image.size
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Preprocesar
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Extraer features con CLIP
            features = self.clip_model.encode_image(input_tensor)
            features = F.normalize(features, dim=-1)
            
            # Clasificar
            logits = self.classifier(features)
            probability = torch.sigmoid(logits).item()
            
            # Determinar nivel de confianza
            if probability < 0.3 or probability > 0.7:
                confidence = "high"
            elif probability < 0.4 or probability > 0.6:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                "probability": probability,
                "confidence": confidence,
                "details": {
                    "detector": self.name,
                    "model": f"CLIP-{CLIP_MODEL_NAME}",
                    "version": self.version,
                    "original_size": f"{original_size[0]}x{original_size[1]}",
                }
            }
            
        except Exception as e:
            logger.error(f"Error en CLIPDetector: {e}")
            raise
    
    @classmethod
    def reset(cls):
        """Resetea el singleton."""
        cls._instance = None
        cls._initialized = False
