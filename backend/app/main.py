from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult

from .config import settings
from .schemas import AnalyzeRequest, JobResponse, UploadResponse
from .tasks.analyze import analyze_media
from .storage import upload_file
from .database import engine, Base
from .routes.auth import router as auth_router
from .deps import get_current_user
from .models import User
from .rate_limiting import setup_rate_limiting, limiter, RATE_LIMITS

# Crear tablas en la base de datos
Base.metadata.create_all(bind=engine)

# Límites de tamaño
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB

# Tipos MIME permitidos
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}
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
