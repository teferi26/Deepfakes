from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
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
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/upload", response_model=UploadResponse)
async def upload_media(
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
    
    # Encolar análisis automáticamente
    task = analyze_media.delay(media_type, result["object_key"])
    
    return UploadResponse(
        object_key=result["object_key"],
        hash=result["hash"],
        size=result["size"],
        media_type=media_type,
        job_id=task.id,
        status="queued"
    )


@app.post("/v1/analyze", response_model=JobResponse)
def create_analysis(
    req: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    """Encola análisis de un archivo ya subido. Requiere autenticación."""
    task = analyze_media.delay(req.media_type, req.reference)
    return JobResponse(job_id=task.id, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    """Consulta estado de un job. Requiere autenticación."""
    result = AsyncResult(job_id)
    status = result.status.lower()

    if status == "failure":
        raise HTTPException(status_code=500, detail="Job failed")

    return JobResponse(job_id=job_id, status=status, result=result.result if result.ready() else None)
