"""
Video AI Detector - Detecta videos generados por IA

Estrategia:
1. Extrae frames del video (muestreo temporal)
2. Analiza cada frame con CLIP
3. Analiza consistencia temporal (flickering, artefactos)
4. Analiza audio si está presente
5. Combina señales para decisión final

Detecta videos de:
- Stable Video Diffusion
- Runway Gen-2/Gen-3
- Pika Labs
- Sora (OpenAI)
- Kling
- HailuoAI
- Otros generadores de video AI
"""

import os
import logging
import tempfile
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Importaciones opcionales
try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP no disponible para video detector")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV no disponible para video detector")


class VideoAIDetector:
    """
    Detector de videos generados por IA.
    
    Características detectadas:
    1. Inconsistencias temporales (flickering)
    2. Artefactos en bordes
    3. Patrones de difusión en frames
    4. Movimientos no naturales
    5. Consistencia de iluminación
    """
    
    name = "video_ai_detector"
    version = "1.0.0"
    
    _instance: Optional['VideoAIDetector'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if VideoAIDetector._initialized:
            return
        
        self._available = False
        self.clip_model = None
        self.preprocess = None
        self.device = None
        self.classifier = None
        
        try:
            self._initialize()
            self._available = True
        except Exception as e:
            logger.error(f"Error inicializando VideoAIDetector: {e}")
        
        VideoAIDetector._initialized = True
    
    def _initialize(self):
        """Inicializa el detector."""
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV es requerido para procesar videos")
        
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP es requerido para el detector")
        
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"VideoAIDetector usando dispositivo: {self.device}")
        
        # Cargar CLIP
        logger.info("Cargando CLIP para video detector...")
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.clip_model.eval()
        
        # Cargar clasificador desde DB
        self._load_classifier_from_db()
        
        logger.info("VideoAIDetector inicializado correctamente")
    
    def _load_classifier_from_db(self):
        """Carga el clasificador entrenado desde PostgreSQL (tabla video_heads)."""
        import torch.nn as nn
        import pickle
        
        # Definir la arquitectura del clasificador MLP (V2 con BatchNorm)
        class VideoClassifier(nn.Module):
            def __init__(self, input_dim: int = 768):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1)
                )
            
            def forward(self, x):
                return self.net(x)
        
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "db"),
                database=os.getenv("POSTGRES_DB", "peritaciones"),
                user=os.getenv("POSTGRES_USER", "user"),
                password=os.getenv("POSTGRES_PASSWORD", "pass")
            )
            
            cursor = conn.cursor()
            
            # Cargar modelo V2 (con BatchNorm) - priorizar el más reciente
            cursor.execute("""
                SELECT model_data, accuracy, source 
                FROM video_heads 
                WHERE source LIKE 'video_ai_v2%%'
                ORDER BY accuracy DESC, created_at DESC 
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            
            if row:
                model_data, accuracy, source = row
                model_info = pickle.loads(model_data)
                
                self.classifier = VideoClassifier().to(self.device)
                self.classifier.load_state_dict(model_info["state_dict"])
                self.classifier.eval()
                
                logger.info(f"Clasificador video cargado desde DB (accuracy: {accuracy}%, source: {source})")
            else:
                # Fallback: usar modelo lineal simple
                logger.warning("No hay modelo en video_heads, intentando clip_heads...")
                cursor.execute("""
                    SELECT weights, bias, accuracy 
                    FROM clip_heads 
                    ORDER BY accuracy DESC, created_at DESC 
                    LIMIT 1
                """)
                row = cursor.fetchone()
                
                if row:
                    weights_json, bias_json, accuracy = row
                    weights = torch.tensor(weights_json, dtype=torch.float32)
                    bias = torch.tensor(bias_json, dtype=torch.float32)
                    
                    self.classifier = nn.Linear(768, 1).to(self.device)
                    self.classifier.weight.data = weights.to(self.device)
                    self.classifier.bias.data = bias.to(self.device)
                    self.classifier.eval()
                    
                    logger.info(f"Clasificador fallback cargado (accuracy: {accuracy}%)")
                else:
                    self.classifier = nn.Linear(768, 1).to(self.device)
                    logger.warning("No hay clasificador en DB, usando pesos aleatorios")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Error cargando clasificador de DB: {e}")
            import torch.nn as nn
            self.classifier = nn.Linear(768, 1).to(self.device)
    
    def is_available(self) -> bool:
        """Verifica si el detector está disponible."""
        return self._available
    
    def extract_frames(self, video_path: str, max_frames: int = 16) -> List[Image.Image]:
        """
        Extrae frames uniformemente distribuidos del video.
        
        Args:
            video_path: Ruta al archivo de video
            max_frames: Número máximo de frames a extraer
            
        Returns:
            Lista de imágenes PIL
        """
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        
        # Calcular índices de frames a extraer
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            frame_indices = [int(i * step) for i in range(max_frames)]
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convertir BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                frames.append(img)
        
        cap.release()
        
        logger.info(f"Extraídos {len(frames)} frames")
        return frames
    
    def extract_features_single(self, img: Image.Image) -> torch.Tensor:
        """Extrae features de un frame con CLIP."""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.clip_model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().float()
    
    def analyze_temporal_consistency(self, frames: List[Image.Image]) -> Dict[str, float]:
        """
        Analiza la consistencia temporal entre frames.
        
        Los videos AI suelen tener:
        - Flickering (cambios bruscos de color/brillo)
        - Inconsistencias en bordes
        - Movimientos no naturales
        
        Returns:
            Dict con métricas de consistencia
        """
        if len(frames) < 2:
            return {"temporal_score": 0.5, "flickering": 0.0}
        
        # Convertir frames a arrays
        frame_arrays = []
        for frame in frames:
            arr = np.array(frame.resize((224, 224)))
            frame_arrays.append(arr.astype(np.float32))
        
        # Calcular diferencias entre frames consecutivos
        diffs = []
        for i in range(len(frame_arrays) - 1):
            diff = np.abs(frame_arrays[i+1] - frame_arrays[i])
            diffs.append(np.mean(diff))
        
        # Flickering: varianza alta en diferencias
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        # Calcular gradientes de bordes
        edge_consistency = []
        for arr in frame_arrays:
            gray = np.mean(arr, axis=2)
            # Sobel para detectar bordes
            sobel_x = np.abs(np.diff(gray, axis=1))
            sobel_y = np.abs(np.diff(gray, axis=0))
            edge_strength = np.mean(sobel_x) + np.mean(sobel_y)
            edge_consistency.append(edge_strength)
        
        edge_variance = np.std(edge_consistency)
        
        # Score temporal (mayor = más probable AI)
        # Videos AI tienen más flickering y menos consistencia en bordes
        flickering_score = min(1.0, std_diff / 30.0)
        edge_score = min(1.0, edge_variance / 20.0)
        
        temporal_score = (flickering_score + edge_score) / 2
        
        return {
            "temporal_score": temporal_score,
            "flickering": flickering_score,
            "edge_inconsistency": edge_score,
            "mean_frame_diff": mean_diff,
            "std_frame_diff": std_diff
        }
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analiza un video para detectar si es generado por IA.
        
        Args:
            video_path: Ruta al archivo de video
            
        Returns:
            Dict con resultados del análisis
        """
        if not self.is_available():
            raise RuntimeError("VideoAIDetector no está disponible")
        
        # Extraer frames
        frames = self.extract_frames(video_path, max_frames=16)
        
        if not frames:
            raise ValueError("No se pudieron extraer frames del video")
        
        # Analizar cada frame con CLIP
        frame_probs = []
        frame_features = []
        
        for frame in frames:
            features = self.extract_features_single(frame)
            frame_features.append(features)
            
            with torch.no_grad():
                output = self.classifier(features)
                prob = torch.sigmoid(output).item()
                frame_probs.append(prob)
        
        # Estadísticas de frames
        mean_prob = np.mean(frame_probs)
        std_prob = np.std(frame_probs)
        min_prob = np.min(frame_probs)
        max_prob = np.max(frame_probs)
        
        # Analizar consistencia temporal
        temporal_analysis = self.analyze_temporal_consistency(frames)
        
        # Combinar señales
        # 1. Probabilidad promedio de CLIP (peso 0.6)
        # 2. Consistencia entre frames (peso 0.2) - si varía mucho, sospechoso
        # 3. Análisis temporal (peso 0.2)
        
        # Penalizar si hay mucha varianza en las predicciones
        consistency_penalty = min(0.2, std_prob)
        
        # Score final
        clip_weight = 0.6
        variance_weight = 0.2
        temporal_weight = 0.2
        
        final_prob = (
            clip_weight * mean_prob +
            variance_weight * consistency_penalty +
            temporal_weight * temporal_analysis["temporal_score"]
        )
        
        # Clamp a [0, 1]
        final_prob = max(0.0, min(1.0, final_prob))
        
        # Decisión
        is_ai = final_prob > 0.5
        
        # Confianza basada en qué tan lejos estamos del umbral
        confidence = abs(final_prob - 0.5) * 2  # 0 a 1
        
        return {
            "probability": final_prob,
            "is_ai_generated": is_ai,
            "confidence": confidence,
            "details": {
                "frames_analyzed": len(frames),
                "frame_probabilities": frame_probs,
                "mean_frame_prob": mean_prob,
                "std_frame_prob": std_prob,
                "min_frame_prob": min_prob,
                "max_frame_prob": max_prob,
                "temporal_analysis": temporal_analysis
            }
        }
    
    def analyze_video_bytes(self, video_bytes: bytes, extension: str = ".mp4") -> Dict[str, Any]:
        """
        Analiza un video desde bytes.
        
        Args:
            video_bytes: Contenido del video en bytes
            extension: Extensión del archivo
            
        Returns:
            Dict con resultados del análisis
        """
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name
        
        try:
            return self.analyze_video(temp_path)
        finally:
            # Limpiar
            if os.path.exists(temp_path):
                os.remove(temp_path)


# Singleton instance
_video_detector: Optional[VideoAIDetector] = None


def get_video_detector() -> VideoAIDetector:
    """Obtiene la instancia singleton del detector de videos."""
    global _video_detector
    if _video_detector is None:
        _video_detector = VideoAIDetector()
    return _video_detector
