from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult

import logging

from .config import settings
from .schemas import (
    AnalyzeRequest,
    JobResponse,
    UploadResponse,
    AnalysisFeedbackCreate,
    AnalysisFeedbackResponse,
    UserCalibrationResponse,
    DatasetUploadResponse,
    DatasetBulkUploadResponse,
    DatasetBulkAutoUploadResponse,
    TrainModelResponse,
)
from .tasks.analyze import analyze_media
from .storage import upload_file
from .database import engine, Base
from .routes.auth import router as auth_router
from .deps import get_current_user
from .models import User

logger = logging.getLogger(__name__)


def _compute_ai_decision(probability: float | None, threshold_low: float, threshold_high: float) -> str | None:
    if probability is None:
        return None
    if probability >= threshold_high:
        return "ai_generated"
    if probability <= threshold_low:
        return "not_ai_generated"
    return "inconclusive"
from .rate_limiting import setup_rate_limiting, limiter, RATE_LIMITS

# Crear tablas en la base de datos
Base.metadata.create_all(bind=engine)

# Límites de tamaño
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB para videos grandes

# Tipos MIME permitidos
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
ALLOWED_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES

app = FastAPI(title="Fraud Detector API", version="0.1.0")

# Configurar rate limiting
setup_rate_limiting(app)

# Incluir router de autenticación
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@limiter.limit(RATE_LIMITS["health"])
def health(request: Request) -> dict:
    return {"status": "ok"}


@app.post("/v1/upload", response_model=UploadResponse)
@limiter.limit(RATE_LIMITS["upload_auth"])
async def upload_media(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Sube imagen o video, valida tipo y tamaño, almacena en S3/MinIO.
    Devuelve referencia del objeto y encola análisis automáticamente.
    Requiere autenticación (JWT o API key).
    """
    content_type = file.content_type or ""
    
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido: {content_type}. Permitidos: {', '.join(ALLOWED_TYPES)}"
        )
    
    # Determinar tipo de medio y límite
    if content_type in ALLOWED_IMAGE_TYPES:
        media_type = "image"
        max_size = MAX_IMAGE_SIZE
    else:
        media_type = "video"
        max_size = MAX_VIDEO_SIZE
    
    # Leer archivo y validar tamaño
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Archivo demasiado grande. Máximo: {max_size // (1024*1024)} MB"
        )
    
    # Subir a S3/MinIO
    from io import BytesIO
    file_obj = BytesIO(contents)
    result = upload_file(file_obj, file.filename or "unknown", content_type)
    
    # Encolar análisis automáticamente (con user_id para historial)
    task = analyze_media.delay(
        media_type, 
        result["object_key"], 
        user_id=str(current_user.id)
    )
    
    return UploadResponse(
        object_key=result["object_key"],
        hash=result["hash"],
        size=result["size"],
        media_type=media_type,
        job_id=task.id,
        status="queued"
    )


@app.post("/v1/dataset/upload", response_model=DatasetUploadResponse)
@limiter.limit(RATE_LIMITS["upload_auth"])
async def upload_dataset_sample(
    request: Request,
    label: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Sube una imagen etiquetada (IA/NO IA) para el banco de entrenamiento."""

    from sqlalchemy.orm import Session
    from .database import SessionLocal
    from .models import TrainingSample, FeedbackLabel
    import uuid

    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Dataset: sólo se admiten imágenes (jpeg/png/webp)")

    try:
        label_enum = FeedbackLabel(label)
    except Exception:
        raise HTTPException(status_code=400, detail="label inválido (ai_generated | not_ai_generated)")

    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Imagen demasiado grande para dataset")

    from io import BytesIO

    file_obj = BytesIO(contents)
    up = upload_file(file_obj, file.filename or "unknown", content_type)

    db: Session = SessionLocal()
    try:
        sample = TrainingSample(
            id=uuid.uuid4(),
            user_id=current_user.id,
            object_key=up["object_key"],
            filename=file.filename,
            label=label_enum,
            source="dataset",
        )
        db.add(sample)
        db.commit()

        return DatasetUploadResponse(
            object_key=sample.object_key,
            filename=sample.filename,
            label=sample.label.value if hasattr(sample.label, "value") else str(sample.label),
            source=sample.source,
        )
    finally:
        db.close()


@app.post("/v1/dataset/upload-zip", response_model=DatasetBulkUploadResponse)
@limiter.limit(RATE_LIMITS["upload_auth"])
async def upload_dataset_zip(
    request: Request,
    label: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Sube un ZIP con muchas imágenes etiquetadas para el banco de entrenamiento.

    Todas las imágenes del ZIP se registran con la misma etiqueta (label).
    """

    from sqlalchemy.orm import Session
    from .database import SessionLocal
    from .models import TrainingSample, FeedbackLabel
    import uuid
    import zipfile
    import os
    from io import BytesIO

    # Validar label
    try:
        label_enum = FeedbackLabel(label)
    except Exception:
        raise HTTPException(status_code=400, detail="label inválido (ai_generated | not_ai_generated)")

    # Validar ZIP por nombre/content-type (content-type no siempre viene bien)
    filename = (file.filename or "").lower()
    if not filename.endswith(".zip") and (file.content_type not in ["application/zip", "application/x-zip-compressed", "multipart/x-zip"]):
        raise HTTPException(status_code=400, detail="Se requiere un archivo .zip")

    zip_max_bytes = int(os.getenv("DATASET_ZIP_MAX_BYTES", str(200 * 1024 * 1024)))
    zip_max_files = int(os.getenv("DATASET_ZIP_MAX_FILES", "5000"))
    error_sample_limit = int(os.getenv("DATASET_ZIP_ERROR_SAMPLES", "10"))

    contents = await file.read()
    if len(contents) > zip_max_bytes:
        raise HTTPException(status_code=400, detail=f"ZIP demasiado grande (max {zip_max_bytes} bytes)")

    uploaded = 0
    skipped = 0
    failed = 0
    error_samples: list[str] = []

    db: Session = SessionLocal()
    try:
        with zipfile.ZipFile(BytesIO(contents)) as zf:
            entries = [e for e in zf.infolist() if not e.is_dir()]
            total_entries = len(entries)
            if total_entries > zip_max_files:
                raise HTTPException(status_code=400, detail=f"Demasiados archivos en ZIP (max {zip_max_files})")

            for e in entries:
                # Filtrar por extensión
                name = (e.filename or "").replace("\\", "/")
                lower = name.lower()
                if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png") or lower.endswith(".webp")):
                    skipped += 1
                    continue

                if e.file_size <= 0:
                    skipped += 1
                    continue

                if e.file_size > MAX_IMAGE_SIZE:
                    skipped += 1
                    if len(error_samples) < error_sample_limit:
                        error_samples.append(f"too_large:{name}")
                    continue

                try:
                    img_bytes = zf.read(e)
                    # Determinar content type por extensión
                    content_type = "image/jpeg"
                    if lower.endswith(".png"):
                        content_type = "image/png"
                    elif lower.endswith(".webp"):
                        content_type = "image/webp"

                    up = upload_file(BytesIO(img_bytes), name.split("/")[-1] or "image", content_type)
                    sample = TrainingSample(
                        id=uuid.uuid4(),
                        user_id=current_user.id,
                        object_key=up["object_key"],
                        filename=name.split("/")[-1],
                        label=label_enum,
                        source="dataset_zip",
                    )
                    db.add(sample)
                    uploaded += 1
                except Exception:
                    failed += 1
                    if len(error_samples) < error_sample_limit:
                        error_samples.append(f"failed:{name}")

            db.commit()

        return DatasetBulkUploadResponse(
            label=label_enum.value if hasattr(label_enum, "value") else str(label_enum),
            source="dataset_zip",
            total_entries=total_entries,
            uploaded=uploaded,
            skipped=skipped,
            failed=failed,
            error_samples=error_samples,
        )

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="ZIP inválido o corrupto")
    finally:
        db.close()


@app.post("/v1/dataset/upload-zip-auto", response_model=DatasetBulkAutoUploadResponse)
@limiter.limit(RATE_LIMITS["upload_auth"])
async def upload_dataset_zip_auto(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Sube un ZIP con 2 carpetas: ai_generated/ y not_ai_generated/.

    Esto permite cargar ambas clases en una sola llamada.
    """

    from sqlalchemy.orm import Session
    from .database import SessionLocal
    from .models import TrainingSample, FeedbackLabel
    import uuid
    import zipfile
    import os
    from io import BytesIO

    # Validar ZIP por nombre/content-type (content-type no siempre viene bien)
    filename = (file.filename or "").lower()
    if not filename.endswith(".zip") and (file.content_type not in ["application/zip", "application/x-zip-compressed", "multipart/x-zip"]):
        raise HTTPException(status_code=400, detail="Se requiere un archivo .zip")

    zip_max_bytes = int(os.getenv("DATASET_ZIP_MAX_BYTES", str(200 * 1024 * 1024)))
    zip_max_files = int(os.getenv("DATASET_ZIP_MAX_FILES", "5000"))
    error_sample_limit = int(os.getenv("DATASET_ZIP_ERROR_SAMPLES", "10"))

    contents = await file.read()
    if len(contents) > zip_max_bytes:
        raise HTTPException(status_code=400, detail=f"ZIP demasiado grande (max {zip_max_bytes} bytes)")

    uploaded_ai = 0
    uploaded_not = 0
    skipped = 0
    failed = 0
    error_samples: list[str] = []

    def _infer_label_from_path(path: str) -> FeedbackLabel | None:
        # top-level folder: ai_generated/.. or not_ai_generated/..
        p = (path or "").replace("\\", "/").lstrip("/")
        if not p:
            return None
        top = p.split("/", 1)[0].lower()
        if top == "ai_generated":
            return FeedbackLabel.ai_generated
        if top == "not_ai_generated":
            return FeedbackLabel.not_ai_generated
        return None

    db: Session = SessionLocal()
    try:
        with zipfile.ZipFile(BytesIO(contents)) as zf:
            entries = [e for e in zf.infolist() if not e.is_dir()]
            total_entries = len(entries)
            if total_entries > zip_max_files:
                raise HTTPException(status_code=400, detail=f"Demasiados archivos en ZIP (max {zip_max_files})")

            for e in entries:
                name = (e.filename or "").replace("\\", "/")
                lower = name.lower()

                label_enum = _infer_label_from_path(name)
                if label_enum is None:
                    skipped += 1
                    continue

                if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png") or lower.endswith(".webp")):
                    skipped += 1
                    continue

                if e.file_size <= 0:
                    skipped += 1
                    continue

                if e.file_size > MAX_IMAGE_SIZE:
                    skipped += 1
                    if len(error_samples) < error_sample_limit:
                        error_samples.append(f"too_large:{name}")
                    continue

                try:
                    img_bytes = zf.read(e)
                    # Determinar content type por extensión
                    content_type = "image/jpeg"
                    if lower.endswith(".png"):
                        content_type = "image/png"
                    elif lower.endswith(".webp"):
                        content_type = "image/webp"

                    base = name.split("/")[-1] or "image"
                    up = upload_file(BytesIO(img_bytes), base, content_type)
                    sample = TrainingSample(
                        id=uuid.uuid4(),
                        user_id=current_user.id,
                        object_key=up["object_key"],
                        filename=base,
                        label=label_enum,
                        source="dataset_zip_auto",
                    )
                    db.add(sample)
                    if label_enum == FeedbackLabel.ai_generated:
                        uploaded_ai += 1
                    else:
                        uploaded_not += 1
                except Exception:
                    failed += 1
                    if len(error_samples) < error_sample_limit:
                        error_samples.append(f"failed:{name}")

            db.commit()

        return DatasetBulkAutoUploadResponse(
            source="dataset_zip_auto",
            total_entries=total_entries,
            uploaded_ai_generated=uploaded_ai,
            uploaded_not_ai_generated=uploaded_not,
            skipped=skipped,
            failed=failed,
            error_samples=error_samples,
        )

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="ZIP inválido o corrupto")
    finally:
        db.close()


@app.post("/v1/dataset/train", response_model=TrainModelResponse)
@limiter.limit(RATE_LIMITS["default"])
def train_dataset_model(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Encola entrenamiento supervisado (head de CLIP) usando dataset + feedback."""

    from .tasks.training import train_clip_head_global

    task = train_clip_head_global.delay()
    return TrainModelResponse(job_id=task.id, status="queued")


@app.post("/v1/analyze", response_model=JobResponse)
@limiter.limit(RATE_LIMITS["analyze_auth"])
def create_analysis(
    request: Request,
    req: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    """Encola análisis de un archivo ya subido. Requiere autenticación."""
    task = analyze_media.delay(
        req.media_type, 
        req.reference, 
        user_id=str(current_user.id)
    )
    return JobResponse(job_id=task.id, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
@limiter.limit(RATE_LIMITS["default"])
def get_job(
    request: Request,
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    """Consulta estado de un job. Requiere autenticación."""
    result = AsyncResult(job_id)
    status = result.status.lower()

    if status == "failure":
        raise HTTPException(status_code=500, detail="Job failed")

    return JobResponse(job_id=job_id, status=status, result=result.result if result.ready() else None)


@app.get("/v1/jobs/{job_id}/report")
@limiter.limit(RATE_LIMITS["report"])
def get_report(
    request: Request,
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Genera y devuelve un reporte PDF del análisis.
    El job debe estar completado (status=success).
    Requiere autenticación.
    """
    from fastapi.responses import Response
    from .reports import generate_pdf_report
    
    result = AsyncResult(job_id)
    
    if not result.ready():
        raise HTTPException(
            status_code=400, 
            detail="El análisis aún no ha terminado. Espera a que el status sea 'success'."
        )
    
    if result.status.lower() == "failure":
        raise HTTPException(status_code=500, detail="El análisis falló, no se puede generar reporte.")
    
    analysis_result = result.result
    if not analysis_result:
        raise HTTPException(status_code=404, detail="No se encontraron resultados del análisis.")
    
    # Generar PDF
    pdf_bytes = generate_pdf_report(
        analysis_id=job_id,
        result=analysis_result,
        filename=analysis_result.get("reference"),
        media_type=analysis_result.get("media_type", "image"),
    )
    
    # Devolver como descarga
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=analisis_{job_id[:8]}.pdf"
        }
    )


# === ENDPOINTS DE HISTORIAL DE ANÁLISIS ===

@app.get("/v1/history")
@limiter.limit(RATE_LIMITS["default"])
def get_analysis_history(
    request: Request,
    page: int = 1,
    page_size: int = 10,
    status: str = None,
    media_type: str = None,
    current_user: User = Depends(get_current_user),
):
    """
    Lista el historial de análisis del usuario actual.
    Soporta paginación y filtros por status y media_type.
    Requiere autenticación.
    """
    from sqlalchemy.orm import Session
    from .database import SessionLocal
    from .models import Analysis
    from .schemas import AnalysisListItem, AnalysisListResponse
    
    # Validar parámetros
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 10
    
    db: Session = SessionLocal()
    try:
        # Umbrales calibrados del usuario (si existen)
        from .models import UserCalibration
        calibration = db.query(UserCalibration).filter(UserCalibration.user_id == current_user.id).first()
        threshold_low = float(calibration.threshold_low) if calibration else 0.35
        threshold_high = float(calibration.threshold_high) if calibration else 0.65

        # Query base filtrada por usuario
        query = db.query(Analysis).filter(Analysis.user_id == current_user.id)
        
        # Aplicar filtros opcionales
        if status:
            query = query.filter(Analysis.status == status)
        if media_type:
            query = query.filter(Analysis.media_type == media_type)
        
        # Contar total
        total = query.count()
        
        # Calcular páginas
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        # Aplicar paginación
        analyses = query.order_by(Analysis.created_at.desc()) \
            .offset((page - 1) * page_size) \
            .limit(page_size) \
            .all()
        
        # Convertir a schema
        items = [
            AnalysisListItem(
                id=str(analysis.id),
                job_id=analysis.job_id,
                media_type=analysis.media_type,
                status=analysis.status.value if hasattr(analysis.status, 'value') else str(analysis.status),
                probability=analysis.probability,
                is_synthetic=analysis.is_synthetic,
                ai_decision=_compute_ai_decision(analysis.probability, threshold_low, threshold_high),
                confidence=analysis.confidence,
                created_at=analysis.created_at,
                completed_at=analysis.completed_at,
            )
            for analysis in analyses
        ]
        
        return AnalysisListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    finally:
        db.close()


@app.get("/v1/history/{analysis_id}")
@limiter.limit(RATE_LIMITS["default"])
def get_analysis_detail(
    request: Request,
    analysis_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Obtiene el detalle completo de un análisis específico.
    Solo el propietario del análisis puede acceder.
    Requiere autenticación.
    """
    from sqlalchemy.orm import Session
    from .database import SessionLocal
    from .models import Analysis
    from .schemas import AnalysisResponse
    import uuid
    
    db: Session = SessionLocal()
    try:
        # Umbrales calibrados del usuario (si existen)
        from .models import UserCalibration
        calibration = db.query(UserCalibration).filter(UserCalibration.user_id == current_user.id).first()
        threshold_low = float(calibration.threshold_low) if calibration else 0.35
        threshold_high = float(calibration.threshold_high) if calibration else 0.65

        # Buscar análisis
        try:
            analysis_uuid = uuid.UUID(analysis_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="ID de análisis inválido")
        
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_uuid,
            Analysis.user_id == current_user.id  # Solo el propietario
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
        
        return AnalysisResponse(
            id=str(analysis.id),
            job_id=analysis.job_id,
            object_key=analysis.object_key,
            media_type=analysis.media_type,
            status=analysis.status.value if hasattr(analysis.status, 'value') else str(analysis.status),
            probability=analysis.probability,
            is_synthetic=analysis.is_synthetic,
            ai_decision=_compute_ai_decision(analysis.probability, threshold_low, threshold_high),
            confidence=analysis.confidence,
            suspected_type=analysis.suspected_type,
            model_version=analysis.model_version,
            result_details=analysis.result_details,
            processing_time_seconds=analysis.processing_time_seconds,
            error_message=analysis.error_message,
            created_at=analysis.created_at,
            completed_at=analysis.completed_at,
        )
    finally:
        db.close()


@app.post("/v1/history/{analysis_id}/feedback", response_model=AnalysisFeedbackResponse)
@limiter.limit(RATE_LIMITS["default"])
def submit_analysis_feedback(
    request: Request,
    analysis_id: str,
    payload: AnalysisFeedbackCreate,
    current_user: User = Depends(get_current_user),
):
    """Permite al usuario indicar si el contenido era IA o no, para recalibrar umbrales."""
    from sqlalchemy.orm import Session
    from .database import SessionLocal
    from .models import Analysis, AnalysisFeedback, FeedbackLabel, UserCalibration
    import uuid
    from datetime import datetime

    def _percentile(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            raise ValueError("empty")
        if q <= 0:
            return sorted_vals[0]
        if q >= 100:
            return sorted_vals[-1]
        k = (len(sorted_vals) - 1) * (q / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_vals) - 1)
        if f == c:
            return sorted_vals[f]
        return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

    db: Session = SessionLocal()
    try:
        try:
            analysis_uuid = uuid.UUID(analysis_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="ID de análisis inválido")

        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_uuid,
            Analysis.user_id == current_user.id,
        ).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")

        # Cargar umbrales actuales (si existen)
        calibration = db.query(UserCalibration).filter(UserCalibration.user_id == current_user.id).first()
        threshold_low = float(calibration.threshold_low) if calibration else 0.35
        threshold_high = float(calibration.threshold_high) if calibration else 0.65

        probability = float(analysis.probability) if analysis.probability is not None else None
        model_decision = _compute_ai_decision(probability, threshold_low, threshold_high)

        label_enum = FeedbackLabel(payload.label)
        was_correct = None
        if model_decision in ("ai_generated", "not_ai_generated"):
            was_correct = (model_decision == payload.label)

        # Idempotencia: un usuario debe tener como máximo 1 feedback por análisis.
        # Si ya existe, lo actualizamos.
        feedback = db.query(AnalysisFeedback).filter(
            AnalysisFeedback.analysis_id == analysis.id,
            AnalysisFeedback.user_id == current_user.id,
        ).first()

        if feedback is None:
            feedback = AnalysisFeedback(
                id=uuid.uuid4(),
                analysis_id=analysis.id,
                user_id=current_user.id,
                label=label_enum,
                was_model_correct=was_correct,
                comment=payload.comment,
                probability_at_time=probability,
                ai_decision_at_time=model_decision,
                model_version_at_time=analysis.model_version,
                created_at=datetime.utcnow(),
            )
            db.add(feedback)
        else:
            feedback.label = label_enum
            feedback.was_model_correct = was_correct
            feedback.comment = payload.comment
            feedback.probability_at_time = probability
            feedback.ai_decision_at_time = model_decision
            feedback.model_version_at_time = analysis.model_version
            feedback.created_at = datetime.utcnow()

        db.commit()
        db.refresh(feedback)

        # Recalibrar umbrales del usuario a partir del feedback acumulado
        all_fb = db.query(AnalysisFeedback).filter(
            AnalysisFeedback.user_id == current_user.id,
            AnalysisFeedback.probability_at_time.isnot(None),
        ).all()

        pos = sorted([float(f.probability_at_time) for f in all_fb if f.label == FeedbackLabel.ai_generated])
        neg = sorted([float(f.probability_at_time) for f in all_fb if f.label == FeedbackLabel.not_ai_generated])

        new_low = 0.35
        new_high = 0.65

        # Solo recalibrar cuando hay suficientes ejemplos de ambas clases
        if len(pos) >= 5 and len(neg) >= 5:
            neg_high = _percentile(neg, 90)
            pos_low = _percentile(pos, 10)

            if neg_high < pos_low:
                new_low = max(0.05, min(0.95, neg_high))
                new_high = max(0.05, min(0.95, pos_low))
            else:
                mid = (neg_high + pos_low) / 2.0
                new_low = max(0.05, min(0.95, mid - 0.05))
                new_high = max(0.05, min(0.95, mid + 0.05))

            if new_low > new_high:
                new_low, new_high = new_high, new_low

            # Guardrails: evitar que el umbral de NO-IA suba demasiado (>
            # 0.5) o que el umbral de IA baje demasiado (<0.5), lo cual puede
            # invertir decisiones y degradar drásticamente la precisión.
            # Mantener un "punto neutro" alrededor de 0.5.
            new_low = min(new_low, 0.49)
            new_high = max(new_high, 0.51)

            # Asegurar un gap mínimo para conservar la zona inconclusiva.
            if (new_high - new_low) < 0.10:
                mid = (new_low + new_high) / 2.0
                new_low = max(0.05, min(0.49, mid - 0.05))
                new_high = max(0.51, min(0.95, mid + 0.05))

        if calibration is None:
            calibration = UserCalibration(
                id=uuid.uuid4(),
                user_id=current_user.id,
                threshold_low=new_low,
                threshold_high=new_high,
                samples_total=len(all_fb),
                samples_ai=len(pos),
                samples_not_ai=len(neg),
                updated_at=datetime.utcnow(),
            )
            db.add(calibration)
        else:
            calibration.threshold_low = new_low
            calibration.threshold_high = new_high
            calibration.samples_total = len(all_fb)
            calibration.samples_ai = len(pos)
            calibration.samples_not_ai = len(neg)
            calibration.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(calibration)

        # Entrenamiento real: cuando hay suficientes muestras y ambas clases,
        # encolamos un calibrador que aprende a mapear outputs del ensemble -> prob final.
        try:
            should_train = (len(pos) >= 10 and len(neg) >= 10)
            if should_train:
                from .tasks.training import train_ensemble_calibrator
                train_ensemble_calibrator.delay()
        except Exception as e:
            # No bloquear feedback por fallos de entrenamiento
            logger.warning(f"No se pudo encolar entrenamiento del calibrador: {e}")

        return AnalysisFeedbackResponse(
            id=str(feedback.id),
            analysis_id=str(feedback.analysis_id),
            label=feedback.label.value if hasattr(feedback.label, "value") else str(feedback.label),
            was_model_correct=feedback.was_model_correct,
            probability_at_time=feedback.probability_at_time,
            ai_decision_at_time=feedback.ai_decision_at_time,
            created_at=feedback.created_at,
            calibration=UserCalibrationResponse(
                threshold_low=float(calibration.threshold_low),
                threshold_high=float(calibration.threshold_high),
                samples_total=int(calibration.samples_total),
                samples_ai=int(calibration.samples_ai),
                samples_not_ai=int(calibration.samples_not_ai),
            ),
        )
    finally:
        db.close()


@app.delete("/v1/history/{analysis_id}")
@limiter.limit(RATE_LIMITS["default"])
def delete_analysis(
    request: Request,
    analysis_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Elimina un análisis del historial.
    Solo el propietario del análisis puede eliminarlo.
    Requiere autenticación.
    """
    from sqlalchemy.orm import Session
    from .database import SessionLocal
    from .models import Analysis
    import uuid
    
    db: Session = SessionLocal()
    try:
        # Buscar análisis
        try:
            analysis_uuid = uuid.UUID(analysis_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="ID de análisis inválido")
        
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_uuid,
            Analysis.user_id == current_user.id  # Solo el propietario
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
        
        db.delete(analysis)
        db.commit()
        
        return {"message": "Análisis eliminado exitosamente", "id": analysis_id}
    finally:
        db.close()
