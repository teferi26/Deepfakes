"""
Detector basado en análisis de frecuencias (DCT/FFT)

Detector que analiza el espectro de frecuencias de la imagen para detectar
patrones característicos de imágenes generadas por IA.

Las imágenes generadas por GANs y Diffusion Models tienen patrones distintivos
en el dominio de frecuencias que no se encuentran en imágenes naturales.
"""

import logging
from typing import Dict, Any, Optional
from io import BytesIO

import numpy as np
from PIL import Image
import cv2

from .base_detector import BaseImageDetector

logger = logging.getLogger(__name__)


class FrequencyDetector(BaseImageDetector):
    """
    Detector basado en análisis de frecuencias (DCT/FFT).
    
    Fortalezas:
    - No requiere entrenamiento (análisis heurístico)
    - Detecta patrones de frecuencia anómalos en GANs
    - Complementa bien a los detectores basados en CNNs
    - Muy rápido (solo operaciones numéricas)
    
    Debilidades:
    - Menos preciso que detectores entrenados
    - Puede fallar con imágenes muy comprimidas
    - Sensible a post-procesamiento
    """
    
    name = "frequency_analysis"
    version = "1.0.0"
    default_weight = 0.15
    
    _instance: Optional['FrequencyDetector'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if FrequencyDetector._initialized:
            return
        
        self._available = True
        FrequencyDetector._initialized = True
        logger.info("FrequencyDetector inicializado")
    
    def is_available(self) -> bool:
        return self._available
    
    def _compute_frequency_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extrae características del espectro de frecuencias."""
        
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Redimensionar para consistencia
        gray = cv2.resize(gray, (256, 256))
        
        # FFT 2D
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Log para mejor visualización
        magnitude_log = np.log1p(magnitude)
        
        # Características del espectro
        center = (128, 128)
        
        # 1. Energía en diferentes anillos de frecuencia
        low_freq_mask = self._create_ring_mask(256, 0, 30)
        mid_freq_mask = self._create_ring_mask(256, 30, 80)
        high_freq_mask = self._create_ring_mask(256, 80, 128)
        
        low_energy = np.sum(magnitude_log * low_freq_mask) / np.sum(low_freq_mask)
        mid_energy = np.sum(magnitude_log * mid_freq_mask) / np.sum(mid_freq_mask)
        high_energy = np.sum(magnitude_log * high_freq_mask) / np.sum(high_freq_mask)
        
        # 2. Ratio high/low (imágenes sintéticas suelen tener menos altas frecuencias)
        freq_ratio = high_energy / (low_energy + 1e-7)
        
        # 3. Detectar picos periódicos (característica de GANs)
        # Analizar varianza en anillos
        ring_variances = []
        for r in range(10, 100, 10):
            ring = self._create_ring_mask(256, r-5, r+5)
            ring_vals = magnitude_log[ring > 0]
            if len(ring_vals) > 0:
                ring_variances.append(np.var(ring_vals))
        
        periodicity_score = np.std(ring_variances) if ring_variances else 0
        
        # 4. Simetría del espectro (imágenes naturales son más simétricas)
        top_half = magnitude_log[:128, :]
        bottom_half = np.flipud(magnitude_log[128:, :])
        symmetry = 1 - np.mean(np.abs(top_half - bottom_half)) / (np.mean(magnitude_log) + 1e-7)
        
        # 5. DCT para detectar artefactos de bloques
        dct = cv2.dct(gray.astype(np.float32))
        dct_energy_ratio = np.sum(np.abs(dct[:32, :32])) / (np.sum(np.abs(dct)) + 1e-7)
        
        return {
            "low_freq_energy": float(low_energy),
            "mid_freq_energy": float(mid_energy),
            "high_freq_energy": float(high_energy),
            "freq_ratio": float(freq_ratio),
            "periodicity_score": float(periodicity_score),
            "symmetry": float(symmetry),
            "dct_concentration": float(dct_energy_ratio),
        }
    
    def _create_ring_mask(self, size: int, r_inner: int, r_outer: int) -> np.ndarray:
        """Crea una máscara en forma de anillo."""
        center = size // 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        mask = (dist >= r_inner) & (dist < r_outer)
        return mask.astype(np.float32)
    
    def _compute_probability(self, features: Dict[str, float]) -> float:
        """
        Calcula probabilidad de ser sintética basada en heurísticas.
        
        Estas heurísticas están basadas en observaciones de:
        - GANs tienen menor energía en altas frecuencias
        - Patrones periódicos en GANs por el upsampling
        - Menor simetría espectral en imágenes sintéticas
        """
        score = 0.5  # Base neutral
        
        # Bajo ratio de frecuencias altas → probable fake
        if features["freq_ratio"] < 0.15:
            score += 0.15
        elif features["freq_ratio"] < 0.25:
            score += 0.08
        elif features["freq_ratio"] > 0.4:
            score -= 0.1
        
        # Alta periodicidad → probable fake (artefactos de upsampling)
        if features["periodicity_score"] > 2.0:
            score += 0.12
        elif features["periodicity_score"] > 1.0:
            score += 0.05
        
        # Baja simetría espectral → probable fake
        if features["symmetry"] < 0.8:
            score += 0.08
        elif features["symmetry"] > 0.95:
            score -= 0.05
        
        # Alta concentración DCT → señal de procesamiento
        if features["dct_concentration"] > 0.7:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def analyze(self, image_data: bytes) -> Dict[str, Any]:
        """Analiza una imagen usando análisis de frecuencias."""
        try:
            # Cargar imagen
            image = Image.open(BytesIO(image_data))
            original_size = image.size
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Convertir a numpy
            img_array = np.array(image)
            
            # Extraer características de frecuencia
            features = self._compute_frequency_features(img_array)
            
            # Calcular probabilidad
            probability = self._compute_probability(features)
            
            # Determinar confianza (este detector es menos preciso)
            if probability < 0.25 or probability > 0.75:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                "probability": probability,
                "confidence": confidence,
                "details": {
                    "detector": self.name,
                    "version": self.version,
                    "method": "FFT/DCT frequency analysis",
                    "features": features,
                    "original_size": f"{original_size[0]}x{original_size[1]}",
                }
            }
            
        except Exception as e:
            logger.error(f"Error en FrequencyDetector: {e}")
            raise
    
    @classmethod
    def reset(cls):
        cls._instance = None
        cls._initialized = False
