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
import time
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base_detector import BaseImageDetector
from ..database import SessionLocal
from ..models import ModelArtifact

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


class CLIPMLPClassifier(nn.Module):
    """2-layer MLP classifier for CLIP features.
    
    Architecture: input_dim -> hidden_dim -> 1 with ReLU and dropout.
    This provides more expressiveness than a single linear layer.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    # Le damos aún más peso en el ensemble porque es el detector
    # con mejor capacidad de generalización entre modelos generativos.
    default_weight = 0.70
    
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

        # Hot-reload: volver a consultar el head en DB cada N segundos.
        # Nota: Celery prefork => cada proceso mantiene su propio cache.
        self._head_lock = threading.Lock()
        self._head_version: Optional[str] = None
        self._head_source: str = "unknown"  # db | universal_fakedetect | untrained
        self._head_last_checked_ts: float = 0.0
        self._head_reload_seconds: float = float(os.getenv("CLIP_HEAD_RELOAD_SECONDS", "30"))
        
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
        
        # 1) Preferir head entrenado en DB (clip_head_global)
        trained_loaded = self._reload_head_from_db(force=True)

        # 2) Fallback: pesos UniversalFakeDetect
        if not trained_loaded:
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
                self._head_version = "universal_fakedetect"
                self._head_source = "universal_fakedetect"
                logger.info("Pesos CLIP (UniversalFakeDetect) cargados correctamente")
            else:
                logger.warning("No se encontraron pesos CLIP, usando clasificador sin entrenar")
                nn.init.zeros_(self.classifier.fc.weight)
                nn.init.zeros_(self.classifier.fc.bias)
                self._head_version = "untrained"
                self._head_source = "untrained"
        
        self.classifier.eval()

    def _reload_head_from_db(self, force: bool = False) -> bool:
        """Hot reload del head desde DB si hay una versión más nueva.

        - Si force=True, intenta cargar aunque no haya pasado el intervalo.
        - Devuelve True si termina usando el head de DB (nuevo o ya vigente).
        """
        if not self.classifier:
            return False

        now = time.time()
        if (not force) and (now - self._head_last_checked_ts) < self._head_reload_seconds:
            return self._head_source == "db"

        self._head_last_checked_ts = now

        # Evitar múltiples consultas/cargas simultáneas en entornos con threads.
        with self._head_lock:
            try:
                db = SessionLocal()
                latest = (
                    db.query(ModelArtifact)
                    .filter(ModelArtifact.name == "clip_head_global")
                    .order_by(ModelArtifact.created_at.desc())
                    .first()
                )
                if not latest:
                    return False

                if self._head_source == "db" and self._head_version == latest.version:
                    return True

                payload = torch.load(BytesIO(latest.artifact), map_location=self.device)
                state_dict = payload.get("state_dict")
                if not state_dict:
                    return False

                # Detect architecture type from payload or state dict keys
                architecture = payload.get("architecture", "linear")
                in_dim = payload.get("in_dim", self.classifier.fc.in_features if hasattr(self.classifier, 'fc') else 768)
                
                if architecture == "MLP_2layer" or "net.0.weight" in state_dict:
                    # New 2-layer MLP architecture
                    hidden_dim = payload.get("hidden_dim", 256)
                    dropout = payload.get("dropout", 0.3)
                    new_classifier = CLIPMLPClassifier(
                        input_dim=in_dim,
                        hidden_dim=hidden_dim,
                        dropout=dropout
                    ).to(self.device)
                    logger.info(f"Loading MLP head: in={in_dim}, hidden={hidden_dim}, dropout={dropout}")
                else:
                    # Legacy linear classifier
                    new_classifier = CLIPLinearClassifier(input_dim=in_dim).to(self.device)
                new_classifier.load_state_dict(state_dict)
                new_classifier.eval()
                self.classifier = new_classifier
                self._head_version = latest.version
                self._head_source = "db"
                logger.info(f"Head CLIP entrenado recargado desde DB: clip_head_global v{latest.version}")
                return True
            except Exception as e:
                logger.warning(f"No se pudo recargar head entrenado de DB: {e}")
                return False
            finally:
                try:
                    db.close()
                except Exception:
                    pass
    
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

        # Hot reload del head (si se entrenó uno nuevo) sin reiniciar el worker.
        self._reload_head_from_db(force=False)
        
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
                    "head_version": self._head_version,
                    "head_source": self._head_source,
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
