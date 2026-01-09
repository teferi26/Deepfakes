"""
Ensemble Detector - Combina múltiples detectores para mayor precisión

Implementa un sistema de voting/weighted average que combina:
1. CLIPDetector (UniversalFakeDetect) - Peso 0.4
2. CNNDetector (ResNet50) - Peso 0.3
3. FrequencyDetector (FFT/DCT) - Peso 0.15
4. MetadataDetector (EXIF/JPEG) - Peso 0.15

La combinación de múltiples señales reduce falsos positivos y mejora
la robustez ante diferentes tipos de manipulaciones.
"""

import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_detector import BaseImageDetector
from .clip_detector import CLIPDetector
from .cnn_detector import CNNDetector
from .frequency_detector import FrequencyDetector
from .metadata_detector import MetadataDetector
from .calibrator import EnsembleCalibrator

logger = logging.getLogger(__name__)


class EnsembleDetector(BaseImageDetector):
    """
    Detector ensemble que combina múltiples modelos.
    
    Estrategia de combinación:
    - Weighted Average: Promedio ponderado de probabilidades
    - Confidence Weighting: Detectores con alta confianza tienen más peso
    - Fallback: Si un detector falla, continúa con los demás
    
    Ventajas:
    - Mayor robustez que cualquier detector individual
    - Reduce falsos positivos
    - Detecta diferentes tipos de manipulaciones
    - Graceful degradation si algún detector falla
    """
    
    name = "ensemble"
    version = "1.0.0"
    default_weight = 1.0
    
    _instance: Optional['EnsembleDetector'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EnsembleDetector._initialized:
            return
        
        self.detectors: List[BaseImageDetector] = []
        self._available = False
        self._calibrator = EnsembleCalibrator(name="ensemble_calibrator_global")
        
        try:
            self._initialize_detectors()
            self._available = True
        except Exception as e:
            logger.error(f"Error inicializando EnsembleDetector: {e}")
        
        EnsembleDetector._initialized = True
    
    def _initialize_detectors(self):
        """Inicializa todos los detectores del ensemble."""
        logger.info("Inicializando detectores del ensemble...")
        
        # Orden de inicialización (más pesados primero)
        detector_classes = [
            CLIPDetector,
            CNNDetector,
            FrequencyDetector,
            MetadataDetector,
        ]
        
        for detector_class in detector_classes:
            try:
                detector = detector_class()
                if detector.is_available():
                    self.detectors.append(detector)
                    logger.info(f"✓ {detector.name} inicializado (peso: {detector.default_weight})")
                else:
                    logger.warning(f"✗ {detector.name} no disponible")
            except Exception as e:
                logger.warning(f"✗ Error inicializando {detector_class.__name__}: {e}")
        
        if not self.detectors:
            raise RuntimeError("No hay detectores disponibles")
        
        # Normalizar pesos
        total_weight = sum(d.default_weight for d in self.detectors)
        self._normalized_weights = {
            d.name: d.default_weight / total_weight 
            for d in self.detectors
        }
        
        logger.info(f"Ensemble inicializado con {len(self.detectors)} detectores")
        logger.info(f"Pesos normalizados: {self._normalized_weights}")
    
    def is_available(self) -> bool:
        return self._available and len(self.detectors) > 0
    
    def get_detectors_info(self) -> List[Dict[str, Any]]:
        """Devuelve información de todos los detectores."""
        return [d.get_info() for d in self.detectors]
    
    def _run_detector(self, detector: BaseImageDetector, image_data: bytes) -> Dict[str, Any]:
        """Ejecuta un detector y captura errores."""
        try:
            result = detector.analyze(image_data)
            result["detector_name"] = detector.name
            result["weight"] = self._normalized_weights.get(detector.name, 0)
            return result
        except Exception as e:
            logger.warning(f"Error en {detector.name}: {e}")
            return {
                "detector_name": detector.name,
                "error": str(e),
                "weight": 0,
            }
    
    def _combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combina resultados de múltiples detectores."""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            raise RuntimeError("Todos los detectores fallaron")
        
        # Calcular probabilidad ponderada (raw)
        total_weight = sum(r["weight"] for r in valid_results)
        weighted_prob = sum(
            r["probability"] * r["weight"] 
            for r in valid_results
        ) / total_weight
        
        # Ajustar por confianza de cada detector
        confidence_multipliers = {"high": 1.2, "medium": 1.0, "low": 0.8}
        confidence_weighted_prob = sum(
            r["probability"] * r["weight"] * confidence_multipliers.get(r.get("confidence", "medium"), 1.0)
            for r in valid_results
        ) / (total_weight * 1.0)  # Normalizar aproximadamente
        
        # Promedio de ambas métricas (raw)
        raw_probability = (weighted_prob + confidence_weighted_prob) / 2
        raw_probability = max(0.0, min(1.0, raw_probability))
        
        # Determinar confianza del ensemble
        # Alta si hay consenso, baja si hay divergencia
        probs = [r["probability"] for r in valid_results]
        prob_std = (sum((p - raw_probability)**2 for p in probs) / len(probs)) ** 0.5
        
        if prob_std < 0.1 and len(valid_results) >= 3:
            confidence = "high"
        elif prob_std < 0.2:
            confidence = "medium"
        else:
            confidence = "low"

        # Aplicar calibrador entrenable (si existe). Usa sólo señales ya calculadas.
        # Features (10): p_clip,p_cnn,p_freq,p_meta, masks(4), raw_prob, prob_std
        det_probs = {r.get("detector_name"): float(r.get("probability", 0.5)) for r in valid_results}

        def _p(name: str) -> tuple[float, float]:
            if name in det_probs:
                return float(det_probs[name]), 1.0
            return 0.5, 0.0

        p_clip, m_clip = _p("clip_universal")
        p_cnn, m_cnn = _p("cnn_detect")
        p_freq, m_freq = _p("frequency_analysis")
        p_meta, m_meta = _p("metadata_analysis")

        features = [
            p_clip, p_cnn, p_freq, p_meta,
            m_clip, m_cnn, m_freq, m_meta,
            float(raw_probability), float(prob_std),
        ]

        # IMPORTANTE: El calibrador fue entrenado con datos antiguos y puede
        # no funcionar bien con el nuevo modelo CLIP entrenado.
        # Por ahora, usamos directamente la probabilidad del CLIP como valor principal
        # ya que es el detector más robusto (peso 0.70) y está recién entrenado.
        import os
        use_clip_directly = os.getenv("USE_CLIP_DIRECTLY", "true").lower() == "true"
        
        if use_clip_directly and m_clip > 0:
            # Usar probabilidad CLIP directamente (ignorar calibrador y otros detectores)
            final_probability = p_clip
            logger.debug(f"Usando CLIP directamente: {p_clip:.3f}")
        else:
            # Fallback: usar calibrador o raw probability
            calibrated_probability = self._calibrator.predict_probability(features)
            final_probability = calibrated_probability if calibrated_probability is not None else raw_probability
        
        # Determinar veredicto IA/no-IA y tipo sospechado basado en probabilidad
        # Umbrales AJUSTADOS para reducir tasa de inconclusive (target ≤3%):
        #   - <= 0.40 → se considera NO generada por IA
        #   - >= 0.60 → se considera generada por IA
        #   - en medio → zona gris (inconclusiva) - ahora más estrecha
        # Configurable via env vars para ajuste fino
        import os
        threshold_low = float(os.getenv("ENSEMBLE_THRESHOLD_LOW", "0.40"))
        threshold_high = float(os.getenv("ENSEMBLE_THRESHOLD_HIGH", "0.60"))
        
        if final_probability >= threshold_high:
            ai_decision = "ai_generated"
        elif final_probability <= threshold_low:
            ai_decision = "not_ai_generated"
        else:
            ai_decision = "inconclusive"

        if final_probability > 0.8:
            suspected_type = "AI-generated (high confidence)"
        elif final_probability > 0.65:
            suspected_type = "Likely AI-generated or heavily manipulated"
        elif final_probability > 0.5:
            suspected_type = "Possibly manipulated (inconclusive)"
        elif final_probability > 0.35:
            suspected_type = "Minor modifications possible"
        else:
            suspected_type = "Likely authentic"
        
        # Construir respuesta
        return {
            "probability": final_probability,
            # is_synthetic mantiene el umbral clásico 0.5 para compatibilidad,
            # mientras que ai_decision usa umbrales más agresivos.
            "is_synthetic": final_probability > 0.5,
            "ai_decision": ai_decision,
            "confidence": confidence,
            "suspected_type": suspected_type,
            "details": {
                "ensemble_size": len(valid_results),
                "failed_detectors": len(results) - len(valid_results),
                "probability_std": prob_std,
                "model": "Ensemble v1.0",
                "model_version": self.version,
                "probability_raw": raw_probability,
                "calibrator_version": (self._calibrator.info().version if self._calibrator.info() else None),
            },
            "individual_results": valid_results,
        }
    
    def analyze(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analiza una imagen usando todos los detectores del ensemble.
        
        Ejecuta los detectores en paralelo para mayor eficiencia,
        luego combina los resultados usando weighted average.
        """
        if not self._available:
            raise RuntimeError("EnsembleDetector no está disponible")
        
        results = []
        
        # Ejecutar detectores (secuencial para evitar problemas con modelos PyTorch)
        # Los modelos pesados (CLIP, CNN) no funcionan bien en paralelo
        for detector in self.detectors:
            result = self._run_detector(detector, image_data)
            results.append(result)
            logger.debug(f"{detector.name}: {result.get('probability', 'error')}")
        
        # Combinar resultados
        combined = self._combine_results(results)
        
        # Extraer tamaño original del primer resultado exitoso
        for r in results:
            if "details" in r and "original_size" in r["details"]:
                combined["details"]["original_size"] = r["details"]["original_size"]
                break
        
        return combined
    
    @classmethod
    def reset(cls):
        """Resetea el singleton y todos los detectores."""
        # Resetear detectores individuales
        CLIPDetector.reset()
        CNNDetector.reset()
        FrequencyDetector.reset()
        MetadataDetector.reset()
        
        cls._instance = None
        cls._initialized = False


# Alias para compatibilidad con código existente
ImageDetector = EnsembleDetector
