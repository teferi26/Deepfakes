"""
Task de Celery para análisis de contenido multimedia.

Usa el detector de imágenes basado en UniversalFakeDetect (CLIP ViT-L/14)
para identificar contenido sintético/manipulado.
"""

import logging
import time
from typing import Literal

from ..celery_app import celery
from ..storage import download_file

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


@celery.task(name="analyze.media", bind=True, max_retries=3)
def analyze_media(self, media_type: Literal["image", "video"], reference: str) -> dict:
    """
    Analiza un archivo multimedia para detectar manipulación/IA.
    
    Args:
        media_type: Tipo de medio ("image" o "video")
        reference: object_key del archivo en MinIO/S3
    
    Returns:
        Dict con resultados del análisis
    """
    start_time = time.time()
    
    try:
        if media_type == "image":
            return _analyze_image(reference, start_time)
        elif media_type == "video":
            return _analyze_video(reference, start_time)
        else:
            raise ValueError(f"Tipo de medio no soportado: {media_type}")
            
    except Exception as e:
        logger.error(f"Error analizando {media_type} {reference}: {e}")
        # Reintentar en caso de error transitorio
        raise self.retry(exc=e, countdown=5)


def _analyze_image(reference: str, start_time: float) -> dict:
    """Analiza una imagen con el detector CLIP."""
    logger.info(f"Analizando imagen: {reference}")
    
    # Descargar imagen de MinIO
    image_data = download_file(reference)
    if image_data is None:
        raise ValueError(f"No se pudo descargar el archivo: {reference}")
    
    # Ejecutar detector
    detector = get_image_detector()
    result = detector.analyze(image_data)
    
    elapsed = time.time() - start_time
    logger.info(f"Análisis completado en {elapsed:.2f}s - Probabilidad: {result['probability']:.3f}")
    
    return {
        "probability": result["probability"],
        "media_type": "image",
        "suspected": result["suspected_type"],
        "explanation": f"Análisis realizado con {result['details']['model']}. "
                      f"Confianza: {result['confidence']}. "
                      f"Tamaño original: {result['details']['original_size']}.",
        "model_version": result["details"]["model_version"],
        "reference": reference,
        "processing_time_seconds": round(elapsed, 2),
        "is_synthetic": result["is_synthetic"],
        "confidence": result["confidence"],
    }


def _analyze_video(reference: str, start_time: float) -> dict:
    """
    Analiza un video extrayendo frames y promediando resultados.
    
    TODO: Implementar análisis de video completo con:
    - Extracción de frames clave
    - Detección de rostros
    - Análisis temporal de inconsistencias
    """
    logger.info(f"Analizando video: {reference}")
    
    # Por ahora, extraer primer frame y analizar como imagen
    # En el futuro: analizar múltiples frames
    video_data = download_file(reference)
    if video_data is None:
        raise ValueError(f"No se pudo descargar el archivo: {reference}")
    
    # Extraer frames del video
    frames = _extract_video_frames(video_data, num_frames=5)
    
    if not frames:
        # Fallback: análisis básico
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
    
    # Analizar cada frame
    detector = get_image_detector()
    probabilities = []
    
    for i, frame_data in enumerate(frames):
        result = detector.analyze(frame_data)
        probabilities.append(result["probability"])
        logger.info(f"Frame {i+1}/{len(frames)}: {result['probability']:.3f}")
    
    # Promediar resultados
    avg_probability = sum(probabilities) / len(probabilities)
    max_probability = max(probabilities)
    
    elapsed = time.time() - start_time
    
    # Determinar confianza basada en varianza
    variance = sum((p - avg_probability) ** 2 for p in probabilities) / len(probabilities)
    if variance < 0.01:
        confidence = "high"
    elif variance < 0.05:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Determinar tipo sospechado
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
        "explanation": f"Análisis de {len(frames)} frames. "
                      f"Probabilidad promedio: {avg_probability:.1%}. "
                      f"Máxima en frame: {max_probability:.1%}. "
                      f"Varianza: {variance:.4f}.",
        "model_version": "UniversalFakeDetect-v1-video",
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
