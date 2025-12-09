import hashlib
import random
import time
from typing import Literal

from ..celery_app import celery


def _mock_probability(seed: str) -> float:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    random.seed(int(digest, 16))
    return round(random.uniform(0.05, 0.95), 3)


@celery.task(name="analyze.media")
def analyze_media(media_type: Literal["image", "video"], reference: str) -> dict:
    # Simula trabajo pesado de inferencia
    time.sleep(2)
    probability = _mock_probability(reference + media_type)

    return {
        "probability": probability,
        "media_type": media_type,
        "suspected": "deepfake rostro" if probability > 0.6 else "indeterminado",
        "explanation": "Resultado simulado. Integrar modelo real en tareas IA.",
        "model_version": "sim-0.0.1",
        "reference": reference,
    }
