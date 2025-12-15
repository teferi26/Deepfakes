"""Training tasks (Celery).

Goal: make the system *actually learn* from user feedback.

We train a lightweight calibrator (logistic regression) that maps the ensemble's
per-detector probabilities into a better final probability.

Why this approach:
- Works with CPU.
- Uses signals we already persist in DB (no need to re-download images).
- Improves over time as feedback grows.
"""

from __future__ import annotations

import io
import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO

from ..celery_app import celery
from ..database import SessionLocal
from ..models import Analysis, AnalysisFeedback, FeedbackLabel, ModelArtifact
from ..storage import download_file

logger = logging.getLogger(__name__)


CALIBRATOR_NAME = "ensemble_calibrator_global"
FEATURE_VERSION = "v1"

CLIP_HEAD_NAME = "clip_head_global"
CLIP_HEAD_FEATURE_VERSION = "v1"


def _extract_features_from_analysis(analysis: Analysis) -> list[float] | None:
    """Build a fixed-length feature vector from stored analysis details.

    Features (10 dims):
      - p_clip, p_cnn, p_freq, p_meta (missing -> 0.5)
      - present_clip, present_cnn, present_freq, present_meta (0/1)
      - ensemble_prob_raw (analysis.probability or 0.5)
      - prob_std (if available else 0.0)
    """

    details = analysis.result_details or {}
    ensemble_details = (details.get("ensemble_details") or {}) if isinstance(details, dict) else {}

    individual = ensemble_details.get("individual_results")
    if individual is None:
        # Fallback: older stored format might put individual results elsewhere
        individual = details.get("individual_results") if isinstance(details, dict) else None

    det_probs: dict[str, float] = {}
    if isinstance(individual, list):
        for item in individual:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("detector_name")
            prob = item.get("probability")
            if isinstance(name, str) and isinstance(prob, (int, float)):
                det_probs[name] = float(prob)

    # Known detector names in this codebase
    names = {
        "clip": "clip_universal",
        "cnn": "cnn_detect",
        "freq": "frequency_analysis",
        "meta": "metadata_analysis",
    }

    def get_prob(det_name: str) -> tuple[float, float]:
        if det_name in det_probs:
            return float(det_probs[det_name]), 1.0
        return 0.5, 0.0

    p_clip, m_clip = get_prob(names["clip"])
    p_cnn, m_cnn = get_prob(names["cnn"])
    p_freq, m_freq = get_prob(names["freq"])
    p_meta, m_meta = get_prob(names["meta"])

    ensemble_prob = float(analysis.probability) if isinstance(analysis.probability, (int, float)) else 0.5
    prob_std = ensemble_details.get("probability_std")
    prob_std = float(prob_std) if isinstance(prob_std, (int, float)) else 0.0

    return [
        p_clip,
        p_cnn,
        p_freq,
        p_meta,
        m_clip,
        m_cnn,
        m_freq,
        m_meta,
        ensemble_prob,
        prob_std,
    ]


class _LogisticCalibrator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _train_logistic_regression(x: torch.Tensor, y: torch.Tensor) -> tuple[_LogisticCalibrator, dict]:
    """Train a tiny logistic regression model in torch."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)

    model = _LogisticCalibrator(in_dim=x.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)

    model.train()
    for _ in range(250):
        optimizer.zero_grad()
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x).squeeze(1))
        preds = (probs >= 0.5).float()
        acc = float((preds == y).float().mean().item())
        loss_val = float(criterion(model(x).squeeze(1), y).item())

    metrics = {
        "train_acc": acc,
        "train_loss": loss_val,
        "n_samples": int(x.shape[0]),
        "in_dim": int(x.shape[1]),
        "feature_version": FEATURE_VERSION,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    return model.cpu(), metrics


@celery.task(name="train.ensemble_calibrator", bind=True, max_retries=0)
def train_ensemble_calibrator(self) -> dict:
    """Train a global calibrator using all available feedback."""

    db = SessionLocal()
    try:
        feedback_rows = (
            db.query(AnalysisFeedback)
            .filter(AnalysisFeedback.label.in_([FeedbackLabel.ai_generated, FeedbackLabel.not_ai_generated]))
            .all()
        )

        if not feedback_rows:
            return {"status": "skipped", "reason": "no_feedback"}

        xs: list[list[float]] = []
        ys: list[float] = []

        for fb in feedback_rows:
            analysis = db.query(Analysis).filter(Analysis.id == fb.analysis_id).first()
            if not analysis:
                continue
            feats = _extract_features_from_analysis(analysis)
            if not feats:
                continue
            xs.append(feats)
            ys.append(1.0 if fb.label == FeedbackLabel.ai_generated else 0.0)

        if len(xs) < 20:
            return {"status": "skipped", "reason": "not_enough_samples", "n_samples": len(xs)}

        # Require both classes
        n_pos = int(sum(1 for v in ys if v == 1.0))
        n_neg = int(sum(1 for v in ys if v == 0.0))
        if n_pos < 5 or n_neg < 5:
            return {"status": "skipped", "reason": "need_both_classes", "n_pos": n_pos, "n_neg": n_neg}

        x = torch.tensor(xs, dtype=torch.float32)
        y = torch.tensor(ys, dtype=torch.float32)

        model, metrics = _train_logistic_regression(x, y)

        buf = io.BytesIO()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_dim": int(x.shape[1]),
                "feature_version": FEATURE_VERSION,
            },
            buf,
        )
        artifact_bytes = buf.getvalue()

        version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        artifact = ModelArtifact(
            name=CALIBRATOR_NAME,
            version=version,
            artifact=artifact_bytes,
            metrics=metrics,
            created_at=datetime.utcnow(),
        )
        db.add(artifact)
        db.commit()

        logger.info(f"Trained {CALIBRATOR_NAME} v{version} on {len(xs)} samples")
        return {"status": "trained", "name": CALIBRATOR_NAME, "version": version, "metrics": metrics}

    finally:
        db.close()


def _load_clip_and_preprocess():
    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device=device,
    )
    clip_model.eval()
    return clip_model, preprocess, device


def _clip_embed(clip_model, preprocess, device, image_bytes: bytes) -> torch.Tensor:
    img = Image.open(BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(x)
        feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat.squeeze(0).detach().cpu()


class _ClipHead(nn.Module):
    """2-layer MLP for CLIP embeddings classification.
    
    Architecture: 768 -> 256 -> 1 with ReLU activation and dropout.
    This provides more expressiveness than a single linear layer.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@celery.task(name="train.clip_head_global", bind=True, max_retries=0)
def train_clip_head_global(self) -> dict:
    """Train a supervised CLIP head using dataset samples + feedback.

    This creates a new head that replaces the UniversalFakeDetect linear head.
    """

    db = SessionLocal()
    try:
        # Import here to avoid circular imports
        from ..models import TrainingSample

        # Collect dataset samples
        dataset_rows = db.query(TrainingSample).all()

        # Collect feedback-derived samples (reuse stored object_key)
        feedback_rows = (
            db.query(AnalysisFeedback, Analysis)
            .join(Analysis, Analysis.id == AnalysisFeedback.analysis_id)
            .filter(Analysis.media_type == "image")
            .filter(AnalysisFeedback.label.in_([FeedbackLabel.ai_generated, FeedbackLabel.not_ai_generated]))
            .all()
        )

        samples: list[tuple[str, FeedbackLabel, str]] = []  # (object_key,label,source)
        for row in dataset_rows:
            samples.append((row.object_key, row.label, "dataset"))
        for fb, an in feedback_rows:
            if an.object_key:
                samples.append((an.object_key, fb.label, "feedback"))

        if len(samples) < 40:
            return {"status": "skipped", "reason": "not_enough_samples", "n_samples": len(samples)}

        n_pos = sum(1 for _, lab, _ in samples if lab == FeedbackLabel.ai_generated)
        n_neg = sum(1 for _, lab, _ in samples if lab == FeedbackLabel.not_ai_generated)
        if n_pos < 10 or n_neg < 10:
            return {"status": "skipped", "reason": "need_both_classes", "n_pos": int(n_pos), "n_neg": int(n_neg)}

        clip_model, preprocess, device = _load_clip_and_preprocess()

        xs: list[torch.Tensor] = []
        ys: list[float] = []

        for object_key, lab, _src in samples:
            try:
                img_bytes = download_file(object_key)
                emb = _clip_embed(clip_model, preprocess, device, img_bytes)
                xs.append(emb)
                ys.append(1.0 if lab == FeedbackLabel.ai_generated else 0.0)
            except Exception:
                continue

        if len(xs) < 30:
            return {"status": "skipped", "reason": "too_many_failures", "n_ok": len(xs), "n_total": len(samples)}

        x = torch.stack(xs).float()
        y = torch.tensor(ys, dtype=torch.float32)

        # Train/val split (deterministic)
        g = torch.Generator().manual_seed(1337)
        perm = torch.randperm(x.shape[0], generator=g)
        x = x[perm]
        y = y[perm]
        n_train = int(x.shape[0] * 0.8)
        x_train, x_val = x[:n_train], x[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        # Gate de calidad: no guardar un head nuevo si no llega al objetivo en validación.
        # Ajustable por env vars.
        target_val_acc = float(os.getenv("CLIP_HEAD_TARGET_VAL_ACC", "0.80"))
        max_attempts = int(os.getenv("CLIP_HEAD_MAX_ATTEMPTS", "5"))
        max_steps = int(os.getenv("CLIP_HEAD_MAX_STEPS", "2000"))
        early_stop_patience = int(os.getenv("CLIP_HEAD_EARLY_STOP_PATIENCE", "100"))

        def _acc(model_: nn.Module, xp: torch.Tensor, yp: torch.Tensor) -> float:
            model_.eval()
            with torch.no_grad():
                p = torch.sigmoid(model_(xp).squeeze(1))
                pred = (p >= 0.5).float()
                return float((pred == yp).float().mean().item())

        # Advanced training with multiple hyperparameter configurations.
        # Uses 2-layer MLP, LR scheduling, and early stopping.
        best = {"model": None, "train_acc": None, "val_acc": None, "loss": None, "lr": None, "steps": None, "hidden_dim": None, "dropout": None}
        
        # Hyperparameter grid for different attempts
        configs = [
            {"lr": 0.001, "hidden_dim": 256, "dropout": 0.3, "steps": max_steps},
            {"lr": 0.0005, "hidden_dim": 256, "dropout": 0.2, "steps": max_steps},
            {"lr": 0.002, "hidden_dim": 128, "dropout": 0.3, "steps": max_steps},
            {"lr": 0.001, "hidden_dim": 512, "dropout": 0.4, "steps": max_steps},
            {"lr": 0.0001, "hidden_dim": 256, "dropout": 0.2, "steps": max_steps},
        ]
        attempts = min(max_attempts, len(configs))
        criterion = nn.BCEWithLogitsLoss()

        for attempt_idx in range(attempts):
            cfg = configs[attempt_idx]
            lr = cfg["lr"]
            hidden_dim = cfg["hidden_dim"]
            dropout = cfg["dropout"]
            steps = cfg["steps"]
            
            model = _ClipHead(in_dim=x.shape[1], hidden_dim=hidden_dim, dropout=dropout)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            
            # Learning rate scheduler: reduce on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-6
            )
            
            model.train()
            last_loss = None
            best_val_in_attempt = 0.0
            patience_counter = 0
            best_state = None
            
            for step in range(steps):
                optimizer.zero_grad()
                logits = model(x_train).squeeze(1)
                loss = criterion(logits, y_train)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                last_loss = float(loss.item())
                
                # Evaluate every 20 steps for early stopping
                if step % 20 == 0 and x_val.shape[0] > 0:
                    current_val_acc = _acc(model, x_val, y_val)
                    scheduler.step(current_val_acc)
                    
                    if current_val_acc > best_val_in_attempt:
                        best_val_in_attempt = current_val_acc
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= early_stop_patience // 20:
                        logger.info(f"Attempt {attempt_idx+1}: Early stopping at step {step} with val_acc={best_val_in_attempt:.4f}")
                        break
                    
                    # Early exit if we've reached target
                    if current_val_acc >= target_val_acc:
                        logger.info(f"Attempt {attempt_idx+1}: Target reached at step {step} with val_acc={current_val_acc:.4f}")
                        break
            
            # Restore best state from this attempt
            if best_state is not None:
                model.load_state_dict(best_state)
            
            train_acc = _acc(model, x_train, y_train)
            val_acc = _acc(model, x_val, y_val) if x_val.shape[0] > 0 else None
            
            val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
            logger.info(f"Attempt {attempt_idx+1}/{attempts}: lr={lr}, hidden={hidden_dim}, dropout={dropout} -> train_acc={train_acc:.4f}, val_acc={val_acc_str}")

            if best["val_acc"] is None:
                is_better = True
            else:
                # Prefer better val_acc; if no val, use train_acc.
                if val_acc is None:
                    is_better = train_acc > float(best["train_acc"] or 0.0)
                else:
                    is_better = val_acc > float(best["val_acc"] or 0.0)

            if is_better:
                best = {
                    "model": model,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "loss": last_loss,
                    "lr": lr,
                    "steps": steps,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                }

            if val_acc is not None and val_acc >= target_val_acc:
                break

        model = best["model"]
        train_acc = float(best["train_acc"]) if best["train_acc"] is not None else None
        val_acc = float(best["val_acc"]) if best["val_acc"] is not None else None

        metrics = {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "n_samples": int(x.shape[0]),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "in_dim": int(x.shape[1]),
            "feature_version": CLIP_HEAD_FEATURE_VERSION,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "target_val_acc": target_val_acc,
            "selected_lr": best.get("lr"),
            "selected_steps": best.get("steps"),
            "selected_hidden_dim": best.get("hidden_dim"),
            "selected_dropout": best.get("dropout"),
            "final_loss": best.get("loss"),
            "architecture": "MLP_2layer",
        }

        if val_acc is not None and val_acc < target_val_acc:
            logger.info(
                f"Trained {CLIP_HEAD_NAME} but below target: val_acc={val_acc:.3f} < {target_val_acc:.3f} (not persisted)"
            )
            return {
                "status": "skipped",
                "reason": "below_target_accuracy",
                "name": CLIP_HEAD_NAME,
                "metrics": metrics,
            }

        buf = io.BytesIO()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_dim": int(x.shape[1]),
                "hidden_dim": best.get("hidden_dim", 256),
                "dropout": best.get("dropout", 0.3),
                "architecture": "MLP_2layer",
                "feature_version": CLIP_HEAD_FEATURE_VERSION,
                "clip_model": "ViT-L-14",
                "clip_pretrained": "openai",
            },
            buf,
        )

        version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        artifact = ModelArtifact(
            name=CLIP_HEAD_NAME,
            version=version,
            artifact=buf.getvalue(),
            metrics=metrics,
            created_at=datetime.utcnow(),
        )
        db.add(artifact)
        db.commit()

        logger.info(f"Trained {CLIP_HEAD_NAME} v{version} on {int(x.shape[0])} samples")
        return {"status": "trained", "name": CLIP_HEAD_NAME, "version": version, "metrics": metrics}

    finally:
        db.close()
