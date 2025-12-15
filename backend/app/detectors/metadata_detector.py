"""
Detector de artefactos de compresión y metadatos

Analiza inconsistencias en la estructura JPEG, metadatos EXIF y patrones
de edición que pueden indicar manipulación.
"""

import logging
from typing import Dict, Any, Optional, List
from io import BytesIO
import struct

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from .base_detector import BaseImageDetector

logger = logging.getLogger(__name__)


class MetadataDetector(BaseImageDetector):
    """
    Detector basado en análisis de metadatos y artefactos JPEG.
    
    Fortalezas:
    - Detecta imágenes sin metadatos (característica de generadores)
    - Identifica inconsistencias en EXIF
    - Detecta múltiples compresiones JPEG
    - No requiere GPU
    
    Debilidades:
    - Fácil de evadir añadiendo metadatos falsos
    - No analiza contenido visual
    - Complementario, no determinante
    """
    
    name = "metadata_analysis"
    version = "1.0.0"
    # Peso muy bajo: solo debe inclinar ligeramente cuando hay evidencia fuerte
    default_weight = 0.02
    
    # Software de edición conocido
    EDITING_SOFTWARE = [
        "photoshop", "gimp", "lightroom", "capture one",
        "affinity", "pixelmator", "snapseed", "vsco",
        "adobe", "corel", "paint.net", "fotor"
    ]
    
    # Generadores de IA conocidos
    AI_GENERATORS = [
        "dall-e", "midjourney", "stable diffusion", "imagen",
        "firefly", "leonardo", "dreamstudio", "novelai",
        "automatic1111", "comfyui"
    ]
    
    _instance: Optional['MetadataDetector'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if MetadataDetector._initialized:
            return
        
        self._available = True
        MetadataDetector._initialized = True
        logger.info("MetadataDetector inicializado")
    
    def is_available(self) -> bool:
        return self._available
    
    def _extract_exif(self, image: Image.Image) -> Dict[str, Any]:
        """Extrae metadatos EXIF de la imagen."""
        exif_data = {}
        
        try:
            raw_exif = image._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)
                    exif_data[tag] = value
        except Exception as e:
            logger.debug(f"No se pudo extraer EXIF: {e}")
        
        return exif_data
    
    def _analyze_jpeg_structure(self, image_data: bytes) -> Dict[str, Any]:
        """Analiza la estructura JPEG para detectar anomalías."""
        result = {
            "is_jpeg": False,
            "has_jfif": False,
            "has_exif": False,
            "has_xmp": False,
            "has_icc": False,
            "quantization_tables": 0,
            "multiple_compressions_likely": False,
        }
        
        if not (image_data[:2] == b'\xff\xd8'):
            return result
        
        result["is_jpeg"] = True
        
        # Buscar marcadores
        pos = 2
        quant_tables = []
        
        while pos < len(image_data) - 1:
            if image_data[pos] != 0xff:
                pos += 1
                continue
            
            marker = image_data[pos + 1]
            
            if marker == 0xd9:  # EOI
                break
            
            if marker == 0xe0:  # APP0 (JFIF)
                result["has_jfif"] = True
            elif marker == 0xe1:  # APP1 (EXIF/XMP)
                if pos + 10 < len(image_data):
                    if b'Exif' in image_data[pos:pos+10]:
                        result["has_exif"] = True
                    if b'XMP' in image_data[pos:pos+30] or b'adobe' in image_data[pos:pos+30].lower():
                        result["has_xmp"] = True
            elif marker == 0xe2:  # APP2 (ICC)
                result["has_icc"] = True
            elif marker == 0xdb:  # DQT (Quantization Table)
                result["quantization_tables"] += 1
            
            # Saltar al siguiente marcador
            if marker in [0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0x01]:
                pos += 2
            elif pos + 3 < len(image_data):
                length = struct.unpack('>H', image_data[pos+2:pos+4])[0]
                pos += 2 + length
            else:
                break
        
        # Más de 2 tablas de cuantización puede indicar recompresión
        if result["quantization_tables"] > 4:
            result["multiple_compressions_likely"] = True
        
        return result
    
    def _check_software_signatures(self, exif: Dict[str, Any]) -> Dict[str, Any]:
        """Verifica firmas de software en los metadatos."""
        result = {
            "has_camera_info": False,
            "has_editing_software": False,
            "has_ai_signature": False,
            "software_detected": [],
            "camera_model": None,
        }
        
        # Buscar modelo de cámara
        camera_fields = ['Make', 'Model', 'LensModel', 'LensMake']
        for field in camera_fields:
            if field in exif and exif[field]:
                result["has_camera_info"] = True
                if field == 'Model':
                    result["camera_model"] = str(exif[field])
                break
        
        # Buscar software
        software_fields = ['Software', 'ProcessingSoftware', 'HostComputer']
        for field in software_fields:
            if field in exif:
                software = str(exif[field]).lower()
                result["software_detected"].append(exif[field])
                
                # Verificar si es software de edición
                for editor in self.EDITING_SOFTWARE:
                    if editor in software:
                        result["has_editing_software"] = True
                        break
                
                # Verificar si es generador de IA
                for generator in self.AI_GENERATORS:
                    if generator in software:
                        result["has_ai_signature"] = True
                        break
        
        return result
    
    def _compute_probability(self, 
                             exif: Dict[str, Any],
                             jpeg_info: Dict[str, Any],
                             software_info: Dict[str, Any]) -> float:
        """
        Calcula probabilidad basada en metadatos.
        
        NOTA: Este detector es conservador porque la ausencia de metadatos
        es común en imágenes de internet, no solo en generadas por IA.
        Solo aumenta significativamente si hay evidencia positiva de IA.
        """
        score = 0.5  # Base neutral
        
        # Firma de IA detectada → muy sospechoso
        if software_info["has_ai_signature"]:
            score += 0.4
        
        # Con información de cámara → probablemente real
        if software_info["has_camera_info"]:
            score -= 0.2
        
        # Sin metadatos EXIF → muy ligeramente sospechoso (pero común en internet)
        if not exif:
            score += 0.02
        
        # JPEG sin JFIF ni EXIF → algo sospechoso (pero no determinante)
        if jpeg_info["is_jpeg"] and not jpeg_info["has_jfif"] and not jpeg_info["has_exif"]:
            score += 0.02
        
        # Múltiples compresiones
        if jpeg_info["multiple_compressions_likely"]:
            score += 0.02
        
        return max(0.0, min(1.0, score))
    
    def analyze(self, image_data: bytes) -> Dict[str, Any]:
        """Analiza metadatos y estructura de la imagen."""
        try:
            image = Image.open(BytesIO(image_data))
            original_size = image.size
            
            # Extraer información
            exif = self._extract_exif(image)
            jpeg_info = self._analyze_jpeg_structure(image_data)
            software_info = self._check_software_signatures(exif)
            
            # Calcular probabilidad
            probability = self._compute_probability(exif, jpeg_info, software_info)
            
            # Confianza basada en cantidad de evidencia
            evidence_count = sum([
                bool(exif),
                software_info["has_camera_info"],
                software_info["has_ai_signature"],
                jpeg_info["has_exif"],
            ])
            
            if evidence_count >= 3:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Si hay firma de IA, alta confianza
            if software_info["has_ai_signature"]:
                confidence = "high"
            
            return {
                "probability": probability,
                "confidence": confidence,
                "details": {
                    "detector": self.name,
                    "version": self.version,
                    "has_exif": bool(exif),
                    "exif_fields_count": len(exif),
                    "jpeg_structure": jpeg_info,
                    "software_analysis": software_info,
                    "original_size": f"{original_size[0]}x{original_size[1]}",
                    "format": image.format,
                }
            }
            
        except Exception as e:
            logger.error(f"Error en MetadataDetector: {e}")
            raise
    
    @classmethod
    def reset(cls):
        cls._instance = None
        cls._initialized = False
