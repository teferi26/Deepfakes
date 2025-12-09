"""
Detector de imágenes generadas por IA / manipuladas.

Implementación basada en UniversalFakeDetect (CVPR 2023):
"Towards Universal Fake Image Detectors that Generalize Across Generative Models"
https://github.com/WisconsinAIVision/UniversalFakeDetect

Utiliza CLIP ViT-L/14 como backbone con una capa linear entrenada para
distinguir imágenes reales de sintéticas (GANs, Diffusion Models, etc.)
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Directorio para almacenar pesos del modelo
WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "fc_weights.pth"  # Pesos oficiales de UniversalFakeDetect

# URL para descargar pesos si no existen
WEIGHTS_URL = "https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth"

# Configuración del modelo
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"
INPUT_SIZE = 224


class CLIPLinearClassifier(nn.Module):
    """
    Clasificador lineal sobre features de CLIP.
    Arquitectura del UniversalFakeDetect: CLIP frozen + linear layer trainable.
    """
    
    def __init__(self, input_dim: int = 768, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ImageDetector:
    """
    Detector de imágenes sintéticas/manipuladas usando CLIP + clasificador lineal.
    
    Características:
    - Generaliza entre distintos modelos generativos (ProGAN, StyleGAN, DALL-E, SD, etc.)
    - Usa CLIP ViT-L/14 como extractor de features (frozen)
    - Clasificador lineal entrenado sobre dataset de imágenes reales/fake
    - Funciona en CPU y GPU
    """
    
    _instance: Optional['ImageDetector'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern para evitar cargar el modelo múltiples veces."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ImageDetector._initialized:
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.classifier = None
        self.preprocess = None
        self._load_model()
        ImageDetector._initialized = True
    
    def _load_model(self):
        """Carga CLIP y el clasificador lineal."""
        try:
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
            
            # Descargar pesos si no existen
            self._ensure_weights_exist()
            
            # Cargar pesos pre-entrenados
            if WEIGHTS_FILE.exists():
                logger.info(f"Cargando pesos desde {WEIGHTS_FILE}")
                state_dict = torch.load(WEIGHTS_FILE, map_location=self.device, weights_only=True)
                # Los pesos de UniversalFakeDetect tienen formato: {'weight': ..., 'bias': ...}
                # Nuestro clasificador espera: {'fc.weight': ..., 'fc.bias': ...}
                if 'fc.weight' not in state_dict and 'weight' in state_dict:
                    state_dict = {
                        'fc.weight': state_dict['weight'],
                        'fc.bias': state_dict['bias']
                    }
                self.classifier.load_state_dict(state_dict)
                logger.info("Pesos cargados correctamente")
            else:
                logger.warning(
                    f"No se encontraron pesos en {WEIGHTS_FILE}. "
                    "Usando clasificador sin entrenar (resultados no fiables)."
                )
                nn.init.zeros_(self.classifier.fc.weight)
                nn.init.zeros_(self.classifier.fc.bias)
            
            self.classifier.eval()
            logger.info("Modelo de detección cargado correctamente")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def _ensure_weights_exist(self):
        """Descarga los pesos si no existen."""
        if WEIGHTS_FILE.exists():
            return
        
        import urllib.request
        
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Descargando pesos desde {WEIGHTS_URL}...")
        
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)
            logger.info(f"Pesos descargados correctamente")
        except Exception as e:
            logger.warning(f"No se pudieron descargar pesos: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesa imagen para CLIP."""
        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Aplicar transformaciones de CLIP
        tensor = self.preprocess(image)
        return tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def analyze(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analiza una imagen y devuelve la probabilidad de que sea sintética.
        
        Args:
            image_data: Bytes de la imagen (JPEG, PNG, WebP)
        
        Returns:
            Dict con:
                - probability: float [0, 1] - probabilidad de ser sintética
                - is_synthetic: bool - True si probability > 0.5
                - confidence: str - nivel de confianza (low/medium/high)
                - suspected_type: str - tipo sospechado de manipulación
                - details: dict - información adicional
        """
        try:
            # Cargar imagen
            image = Image.open(BytesIO(image_data))
            original_size = image.size
            
            # Preprocesar
            input_tensor = self._preprocess_image(image)
            
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
            
            # Determinar tipo sospechado
            if probability > 0.7:
                suspected_type = "AI-generated (GAN/Diffusion)"
            elif probability > 0.5:
                suspected_type = "Possibly AI-generated or heavily edited"
            elif probability > 0.3:
                suspected_type = "Minor modifications possible"
            else:
                suspected_type = "Likely authentic"
            
            return {
                "probability": probability,
                "is_synthetic": probability > 0.5,
                "confidence": confidence,
                "suspected_type": suspected_type,
                "details": {
                    "model": f"CLIP-{CLIP_MODEL_NAME}",
                    "model_version": "UniversalFakeDetect-v1",
                    "original_size": f"{original_size[0]}x{original_size[1]}",
                    "device": str(self.device),
                }
            }
            
        except Exception as e:
            logger.error(f"Error analizando imagen: {e}")
            raise
    
    @classmethod
    def reset(cls):
        """Resetea el singleton (útil para tests)."""
        cls._instance = None
        cls._initialized = False
