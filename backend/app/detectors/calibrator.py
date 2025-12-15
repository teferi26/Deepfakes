from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..database import SessionLocal
from ..models import ModelArtifact

logger = logging.getLogger(__name__)


@dataclass
class CalibratorInfo:
    name: str
    version: str


class LogisticCalibrator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class EnsembleCalibrator:
    """Loads and applies a persisted calibrator model."""

    def __init__(self, name: str, reload_every_seconds: int = 60):
        self.name = name
        self.reload_every_seconds = reload_every_seconds
        self._model: LogisticCalibrator | None = None
        self._in_dim: int | None = None
        self._version: str | None = None
        self._last_check_ts: float = 0.0

    def info(self) -> CalibratorInfo | None:
        if self._version is None:
            return None
        return CalibratorInfo(name=self.name, version=self._version)

    def _should_check(self) -> bool:
        return (time.time() - self._last_check_ts) >= self.reload_every_seconds

    def _load_latest_from_db(self) -> None:
        db = SessionLocal()
        try:
            latest = (
                db.query(ModelArtifact)
                .filter(ModelArtifact.name == self.name)
                .order_by(ModelArtifact.created_at.desc())
                .first()
            )
            if not latest:
                return

            if self._version == latest.version and self._model is not None:
                return

            payload = torch.load(io.BytesIO(latest.artifact), map_location="cpu")
            in_dim = int(payload["in_dim"])
            model = LogisticCalibrator(in_dim=in_dim)
            model.load_state_dict(payload["state_dict"])
            model.eval()

            self._model = model
            self._in_dim = in_dim
            self._version = str(latest.version)
            logger.info(f"Loaded calibrator {self.name} v{self._version}")
        except Exception as e:
            logger.warning(f"Failed to load calibrator {self.name}: {e}")
        finally:
            db.close()

    def maybe_reload(self) -> None:
        if not self._should_check():
            return
        self._last_check_ts = time.time()
        self._load_latest_from_db()

    @torch.no_grad()
    def predict_probability(self, features: list[float]) -> float | None:
        self.maybe_reload()
        if self._model is None or self._in_dim is None:
            return None
        if len(features) != self._in_dim:
            return None
        x = torch.tensor([features], dtype=torch.float32)
        logits = self._model(x)
        prob = torch.sigmoid(logits).item()
        return float(max(0.0, min(1.0, prob)))
