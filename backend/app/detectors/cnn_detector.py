"""
Detector CNNDetect (basado en ResNet50)

Detector basado en el paper "CNN-generated images are surprisingly easy to spot... for now"
https://github.com/peterwang512/CNNDetection

Detecta artefactos característicos de upsampling en imágenes generadas por CNNs/GANs.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

from .base_detector import BaseImageDetector

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "cnn_detector_weights.pth"

# URL de pesos pre-entrenados (blur3_jpg50 - más robusto a JPEG)
WEIGHTS_URL = "https://www.dropbox.com/s/h23q01w7jp41u31/blur_jpg_prob0.5.pth?dl=1"

INPUT_SIZE = 224


class CNNDetector(BaseImageDetector):
    """
    Detector basado en ResNet50 para artefactos de CNN.
    
    Fortalezas:
    - Muy efectivo detectando imágenes de GANs (StyleGAN, ProGAN, BigGAN)
    - Detecta artefactos de upsampling característicos
    - Robusto a compresión JPEG
    
    Debilidades:
    - Menos efectivo con Diffusion Models modernos
    - Puede fallar con imágenes muy post-procesadas
    """
    
    name = "cnn_detect"
    version = "1.0.0"
    # Peso medio: refuerza a CLIP pero no domina el ensemble
    default_weight = 0.25
    
    _instance: Optional['CNNDetector'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if CNNDetector._initialized:
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self._available = False
        
        try:
            self._load_model()
            self._available = True
        except Exception as e:
            logger.error(f"Error inicializando CNNDetector: {e}")
        
        CNNDetector._initialized = True
    
    def _load_model(self):
        """Carga ResNet50 modificado para detección."""
        logger.info(f"Cargando CNNDetector (ResNet50) en {self.device}...")
        
        # ResNet50 con 1 salida (fake probability)
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        
        # Transform estándar de ImageNet
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Intentar cargar pesos pre-entrenados
        if WEIGHTS_FILE.exists():
            logger.info(f"Cargando pesos desde {WEIGHTS_FILE}")
            try:
                state_dict = torch.load(WEIGHTS_FILE, map_location=self.device, weights_only=False)
                # Manejar diferentes formatos de checkpoint
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Limpiar prefijos si existen
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                
                self.model.load_state_dict(new_state_dict, strict=False)
                logger.info("Pesos CNNDetect cargados correctamente")
            except Exception as e:
                logger.warning(f"Error cargando pesos CNNDetect: {e}")
                self._init_random_weights()
        else:
            logger.warning("No se encontraron pesos CNNDetect, usando pesos de ImageNet")
            # Usar pesos pre-entrenados de ImageNet como base
            pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Copiar todos los pesos excepto fc
            pretrained_dict = pretrained.state_dict()
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _init_random_weights(self):
        """Inicializa pesos aleatorios para la capa fc."""
        nn.init.xavier_normal_(self.model.fc.weight)
        nn.init.zeros_(self.model.fc.bias)
    
    def is_available(self) -> bool:
        return self._available
    
    @torch.no_grad()
    def analyze(self, image_data: bytes) -> Dict[str, Any]:
        """Analiza una imagen buscando artefactos de CNN."""
        if not self._available:
            raise RuntimeError("CNNDetector no está disponible")
        
        try:
            image = Image.open(BytesIO(image_data))
            original_size = image.size
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Preprocesar
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inferencia
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()
            
            # Determinar confianza
            if probability < 0.2 or probability > 0.8:
                confidence = "high"
            elif probability < 0.35 or probability > 0.65:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                "probability": probability,
                "confidence": confidence,
                "details": {
                    "detector": self.name,
                    "model": "ResNet50-CNNDetect",
                    "version": self.version,
                    "focus": "CNN upsampling artifacts",
                    "original_size": f"{original_size[0]}x{original_size[1]}",
                }
            }
            
        except Exception as e:
            logger.error(f"Error en CNNDetector: {e}")
            raise
    
    @classmethod
    def reset(cls):
        cls._instance = None
        cls._initialized = False
