from typing import Literal, Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    media_type: Literal["image", "video"] = Field(description="Tipo de medio a analizar")
    reference: str = Field(description="Referencia/ID/URL temporal del archivo almacenado")


class JobResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None


class UploadResponse(BaseModel):
    object_key: str = Field(description="Clave del objeto en S3/MinIO")
    hash: str = Field(description="SHA-256 del archivo")
    size: int = Field(description="Tamaño en bytes")
    media_type: Literal["image", "video"] = Field(description="Tipo de medio detectado")
    job_id: str = Field(description="ID del job de análisis encolado")
    status: str = Field(description="Estado inicial del job")
