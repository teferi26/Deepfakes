"""
Clase base abstracta para todos los detectores de imágenes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseImageDetector(ABC):
    """
    Interfaz base para detectores de imágenes sintéticas/manipuladas.
    
    Todos los detectores deben implementar el método analyze() que recibe
    bytes de imagen y devuelve un diccionario con los resultados.
    """
    
    # Nombre del detector (para identificación en resultados)
    name: str = "base"
    
    # Versión del detector
    version: str = "1.0.0"
    
    # Peso por defecto en el ensemble (0-1)
    default_weight: float = 1.0
    
    @abstractmethod
    def analyze(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analiza una imagen y devuelve la probabilidad de que sea sintética.
        
        Args:
            image_data: Bytes de la imagen (JPEG, PNG, WebP)
        
        Returns:
            Dict con al menos:
                - probability: float [0, 1] - probabilidad de ser sintética
                - confidence: str - nivel de confianza (low/medium/high)
                - details: dict - información adicional del detector
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Verifica si el detector está disponible (modelo cargado, etc.)
        
        Returns:
            True si el detector puede procesar imágenes
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Devuelve información sobre el detector."""
        return {
            "name": self.name,
            "version": self.version,
            "default_weight": self.default_weight,
            "available": self.is_available()
        }
