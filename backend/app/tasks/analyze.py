"""
Task de Celery para análisis de contenido multimedia.

Usa un ensemble de detectores para identificar contenido sintético/manipulado:
- CLIPDetector (UniversalFakeDetect) - Generaliza entre GANs y Diffusion
- CNNDetector (ResNet50) - Detecta artefactos de upsampling
- FrequencyDetector (FFT/DCT) - Analiza espectro de frecuencias
- MetadataDetector (EXIF/JPEG) - Analiza metadatos y estructura
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Literal, Optional

from ..celery_app import celery
from ..storage import download_file
from ..database import SessionLocal
from ..models import Analysis, AnalysisStatus

logger = logging.getLogger(__name__)

# Inicializar detector de forma lazy (se carga en el primer uso)
_image_detector = None


def get_image_detector():
    """Obtiene instancia singleton del detector de imágenes."""
    global _image_detector
    if _image_detector is None:
        from ..detectors import ImageDetector
        logger.info("Inicializando detector de imágenes...")
        _image_detector = ImageDetector()
    return _image_detector


def _create_analysis_record(
    db, 
    job_id: str, 
    user_id: str, 
    object_key: str, 
    media_type: str
) -> Analysis:
    """Crea un registro de análisis pendiente en la base de datos."""
    analysis = Analysis(
        id=uuid.uuid4(),
        user_id=uuid.UUID(user_id),
        job_id=job_id,
        object_key=object_key,
        media_type=media_type,
        status=AnalysisStatus.processing,
        model_version="ensemble-v1",
        created_at=datetime.utcnow(),
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _update_analysis_result(db, analysis: Analysis, result: dict):
    """Actualiza el registro de análisis con los resultados."""
    analysis.status = AnalysisStatus.completed
    analysis.probability = result.get("probability")
    analysis.is_synthetic = result.get("is_synthetic")
    # Decisión IA/NO IA (con umbral calibrado si aplica)
    # Se guarda en result_details; no añadimos columna para mantener compatibilidad.
    analysis.confidence = result.get("confidence")
    analysis.suspected_type = result.get("suspected")
    analysis.model_version = result.get("model_version", "ensemble-v1")
    analysis.processing_time_seconds = result.get("processing_time_seconds")
    analysis.completed_at = datetime.utcnow()
    
    # Guardar detalles completos incluyendo resultados individuales
    analysis.result_details = {
        "ai_decision": result.get("ai_decision"),
        "decision_thresholds": result.get("decision_thresholds"),
        "ensemble_details": result.get("ensemble_details"),
        "explanation": result.get("explanation"),
        "reference": result.get("reference"),
        "media_type": result.get("media_type"),
    }
    
    db.commit()


def _mark_analysis_failed(db, analysis: Analysis, error: str):
    """Marca el análisis como fallido."""
    analysis.status = AnalysisStatus.failed
    analysis.error_message = error[:500]
    analysis.completed_at = datetime.utcnow()
    db.commit()


@celery.task(name="analyze.media", bind=True, max_retries=3)
def analyze_media(
    self, 
    media_type: Literal["image", "video"], 
    reference: str,
    user_id: Optional[str] = None
) -> dict:
    """
    Analiza un archivo multimedia para detectar manipulación/IA.
    
    Args:
        media_type: Tipo de medio ("image" o "video")
        reference: object_key del archivo en MinIO/S3
        user_id: ID del usuario para guardar en historial (opcional)
    
    Returns:
        Dict con resultados del análisis
    """
    start_time = time.time()
    job_id = self.request.id
    
    # Crear registro de análisis si tenemos user_id
    db = None
    analysis = None
    analysis_id = None
    if user_id:
        try:
            db = SessionLocal()
            analysis = _create_analysis_record(db, job_id, user_id, reference, media_type)
            logger.info(f"Creado registro de análisis {analysis.id} para job {job_id}")
            analysis_id = str(analysis.id)
        except Exception as e:
            logger.error(f"Error creando registro de análisis: {e}")
            if db:
                db.close()
                db = None
    
    try:
        if media_type == "image":
            result = _analyze_image(reference, start_time)
        elif media_type == "video":
            result = _analyze_video(reference, start_time)
        else:
            raise ValueError(f"Tipo de medio no soportado: {media_type}")
        
        # Inyectar analysis_id para que el frontend pueda enviar feedback
        if analysis_id:
            result["analysis_id"] = analysis_id

        # Aplicar umbral calibrado por usuario si existe
        if analysis_id and db:
            try:
                from ..models import UserCalibration
                calibration = db.query(UserCalibration).filter(UserCalibration.user_id == uuid.UUID(user_id)).first()
                threshold_low = float(calibration.threshold_low) if calibration else 0.35
                threshold_high = float(calibration.threshold_high) if calibration else 0.65

                prob = result.get("probability")
                if prob is not None:
                    if prob >= threshold_high:
                        result["ai_decision"] = "ai_generated"
                    elif prob <= threshold_low:
                        result["ai_decision"] = "not_ai_generated"
                    else:
                        result["ai_decision"] = "inconclusive"
                result["decision_thresholds"] = {
                    "threshold_low": threshold_low,
                    "threshold_high": threshold_high,
                }
            except Exception as e:
                logger.warning(f"No se pudo aplicar calibración de usuario: {e}")

        # Actualizar registro con resultados
        if analysis and db:
            try:
                _update_analysis_result(db, analysis, result)
                logger.info(f"Actualizado registro de análisis {analysis.id}")
            except Exception as e:
                logger.error(f"Error actualizando registro de análisis: {e}")
        
        return result
            
    except Exception as e:
        logger.error(f"Error analizando {media_type} {reference}: {e}")
        
        # Marcar como fallido en la base de datos
        if analysis and db:
            try:
                _mark_analysis_failed(db, analysis, str(e))
            except Exception as db_error:
                logger.error(f"Error marcando análisis como fallido: {db_error}")
        
        # Reintentar en caso de error transitorio
        raise self.retry(exc=e, countdown=5)
    finally:
        if db:
            db.close()


def _analyze_image(reference: str, start_time: float) -> dict:
    """Analiza una imagen con el ensemble de detectores."""
    logger.info(f"Analizando imagen: {reference}")
    
    # Descargar imagen de MinIO
    image_data = download_file(reference)
    if image_data is None:
        raise ValueError(f"No se pudo descargar el archivo: {reference}")
    
    # Ejecutar detector ensemble
    detector = get_image_detector()
    result = detector.analyze(image_data)
    
    elapsed = time.time() - start_time
    
    # Extraer información del ensemble
    ensemble_size = result['details'].get('ensemble_size', 1)
    individual_results = result.get('individual_results', [])
    
    # Construir explicación detallada
    explanation_parts = [
        f"Análisis realizado con ensemble de {ensemble_size} detectores.",
        f"Confianza: {result['confidence']}.",
    ]
    
    if individual_results:
        detector_summaries = []
        for ir in individual_results:
            detector_summaries.append(
                f"{ir.get('detector_name', 'unknown')}: {ir.get('probability', 0):.1%}"
            )
        explanation_parts.append(f"Detectores: {', '.join(detector_summaries)}.")
    
    if 'original_size' in result['details']:
        explanation_parts.append(f"Tamaño: {result['details']['original_size']}.")
    
    logger.info(f"Análisis completado en {elapsed:.2f}s - Probabilidad: {result['probability']:.3f}")
    
    return {
        "probability": result["probability"],
        "media_type": "image",
        "suspected": result["suspected_type"],
        "explanation": " ".join(explanation_parts),
        "model_version": result["details"].get("model_version", "ensemble-v1"),
        "reference": reference,
        "processing_time_seconds": round(elapsed, 2),
        "is_synthetic": result["is_synthetic"],
        "ai_decision": result.get("ai_decision"),
        "confidence": result["confidence"],
        "ensemble_details": {
            "ensemble_size": ensemble_size,
            "failed_detectors": result['details'].get('failed_detectors', 0),
            "probability_std": result['details'].get('probability_std', 0),
            "probability_raw": result['details'].get('probability_raw'),
            "calibrator_version": result['details'].get('calibrator_version'),
            "individual_results": [
                {
                    "name": ir.get("detector_name"),
                    "probability": round(ir.get("probability", 0), 4),
                    "confidence": ir.get("confidence"),
                    "weight": round(ir.get("weight", 0), 3),
                    "extra": (
                        {
                            "head_version": (ir.get("details") or {}).get("head_version"),
                            "head_source": (ir.get("details") or {}).get("head_source"),
                        }
                        if ir.get("detector_name") == "clip_universal" and isinstance(ir.get("details"), dict)
                        else None
                    ),
                }
                for ir in individual_results
            ] if individual_results else None,
        },
    }


def _analyze_video(reference: str, start_time: float) -> dict:
    """
    Analiza un video usando el VideoAIDetector especializado.
    
    Características:
    - Extracción de frames clave con muestreo temporal
    - Análisis CLIP de cada frame
    - Análisis de consistencia temporal (flickering, artefactos)
    - Combinación de señales para decisión final
    """
    logger.info(f"Analizando video: {reference}")
    
    video_data = download_file(reference)
    if video_data is None:
        raise ValueError(f"No se pudo descargar el archivo: {reference}")
    
    # Intentar usar el VideoAIDetector especializado
    try:
        from ..detectors.video_detector import get_video_detector
        
        video_detector = get_video_detector()
        if video_detector.is_available():
            # Determinar extensión del archivo
            ext = ".mp4"
            if reference.lower().endswith(".mov"):
                ext = ".mov"
            elif reference.lower().endswith(".avi"):
                ext = ".avi"
            elif reference.lower().endswith(".webm"):
                ext = ".webm"
            
            result = video_detector.analyze_video_bytes(video_data, extension=ext)
            elapsed = time.time() - start_time
            
            # Determinar tipo sospechado
            prob = result["probability"]
            if prob > 0.8:
                suspected = "AI-generated video (Alta probabilidad)"
            elif prob > 0.6:
                suspected = "Probablemente generado por IA"
            elif prob > 0.4:
                suspected = "Posible manipulación (Inconcluso)"
            else:
                suspected = "Video probablemente auténtico"
            
            # Determinar nivel de confianza
            confidence_val = result.get("confidence", 0.5)
            if confidence_val > 0.7:
                confidence = "high"
            elif confidence_val > 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            details = result.get("details", {})
            
            return {
                "probability": prob,
                "media_type": "video",
                "suspected": suspected,
                "explanation": f"Análisis de {details.get('frames_analyzed', 0)} frames con CLIP + análisis temporal. "
                              f"Probabilidad: {prob:.1%}. "
                              f"Flickering score: {details.get('temporal_analysis', {}).get('flickering', 0):.2f}. "
                              f"Consistencia temporal: {details.get('temporal_analysis', {}).get('temporal_score', 0):.2f}.",
                "model_version": "video-ai-detector-v1",
                "reference": reference,
                "processing_time_seconds": round(elapsed, 2),
                "is_synthetic": result.get("is_ai_generated", prob > 0.5),
                "ai_decision": "ai_generated" if prob > 0.6 else ("not_ai_generated" if prob < 0.4 else "inconclusive"),
                "confidence": confidence,
                "video_analysis": {
                    "frames_analyzed": details.get("frames_analyzed", 0),
                    "frame_probabilities": details.get("frame_probabilities", []),
                    "mean_frame_prob": details.get("mean_frame_prob", 0),
                    "std_frame_prob": details.get("std_frame_prob", 0),
                    "temporal_analysis": details.get("temporal_analysis", {}),
                }
            }
    except Exception as e:
        logger.warning(f"VideoAIDetector no disponible, usando fallback: {e}")
    
    # Fallback: análisis básico con extracción de frames
    frames = _extract_video_frames(video_data, num_frames=8)
    
    if not frames:
        elapsed = time.time() - start_time
        return {
            "probability": 0.5,
            "media_type": "video",
            "suspected": "Análisis de video limitado",
            "explanation": "No se pudieron extraer frames del video. "
                          "Usar formato MP4 o WebM para mejores resultados.",
            "model_version": "video-fallback-v1",
            "reference": reference,
            "processing_time_seconds": round(elapsed, 2),
            "is_synthetic": False,
            "confidence": "low",
        }
    
    # Analizar cada frame con el detector de imágenes
    detector = get_image_detector()
    probabilities = []
    
    for i, frame_data in enumerate(frames):
        result = detector.analyze(frame_data)
        probabilities.append(result["probability"])
        logger.info(f"Frame {i+1}/{len(frames)}: {result['probability']:.3f}")
    
    avg_probability = sum(probabilities) / len(probabilities)
    max_probability = max(probabilities)
    
    elapsed = time.time() - start_time
    
    variance = sum((p - avg_probability) ** 2 for p in probabilities) / len(probabilities)
    if variance < 0.01:
        confidence = "high"
    elif variance < 0.05:
        confidence = "medium"
    else:
        confidence = "low"
    
    if avg_probability > 0.7:
        suspected = "AI-generated video (Deepfake probable)"
    elif avg_probability > 0.5:
        suspected = "Possibly manipulated video"
    elif max_probability > 0.6:
        suspected = "Some frames show AI artifacts"
    else:
        suspected = "Likely authentic video"
    
    return {
        "probability": avg_probability,
        "media_type": "video",
        "suspected": suspected,
        "explanation": f"Análisis de {len(frames)} frames con ensemble de detectores. "
                      f"Probabilidad promedio: {avg_probability:.1%}. "
                      f"Máxima en frame: {max_probability:.1%}. "
                      f"Varianza: {variance:.4f}.",
        "model_version": "ensemble-v1-video",
        "reference": reference,
        "processing_time_seconds": round(elapsed, 2),
        "is_synthetic": avg_probability > 0.5,
        "confidence": confidence,
        "frame_analysis": {
            "num_frames": len(frames),
            "probabilities": [round(p, 3) for p in probabilities],
            "max_probability": round(max_probability, 3),
            "variance": round(variance, 4),
        }
    }


def _extract_video_frames(video_data: bytes, num_frames: int = 5) -> list:
    """
    Extrae frames de un video.
    
    Returns:
        Lista de bytes de imágenes JPEG
    """
    import tempfile
    import os
    from io import BytesIO
    
    try:
        # Intentar usar OpenCV si está disponible
        import cv2
        import numpy as np
        from PIL import Image
        
        # Guardar video temporalmente
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_data)
            temp_path = f.name
        
        try:
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return []
            
            # Calcular índices de frames a extraer
            indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convertir BGR a RGB y luego a JPEG bytes
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    buffer = BytesIO()
                    img.save(buffer, format='JPEG', quality=95)
                    frames.append(buffer.getvalue())
            
            cap.release()
            return frames
            
        finally:
            os.unlink(temp_path)
            
    except ImportError:
        logger.warning("OpenCV no disponible. Instalar opencv-python para análisis de video.")
        return []
    except Exception as e:
        logger.error(f"Error extrayendo frames: {e}")
        return []
