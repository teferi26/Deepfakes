# Detectores de contenido sintético/manipulado

# Detectores individuales
from .base_detector import BaseImageDetector
from .clip_detector import CLIPDetector
from .cnn_detector import CNNDetector
from .frequency_detector import FrequencyDetector
from .metadata_detector import MetadataDetector

# Ensemble (combinación de todos los detectores)
from .ensemble_detector import EnsembleDetector, ImageDetector

__all__ = [
    # Base
    "BaseImageDetector",
    # Detectores individuales
    "CLIPDetector",
    "CNNDetector", 
    "FrequencyDetector",
    "MetadataDetector",
    # Ensemble
    "EnsembleDetector",
    "ImageDetector",  # Alias de EnsembleDetector para compatibilidad
]
