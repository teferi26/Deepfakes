from typing import Literal, Optional
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime


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


# Auth schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, description="Mínimo 8 caracteres")


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Segundos hasta expiración")


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    role: str
    api_key: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyResponse(BaseModel):
    api_key: str


# Analysis History schemas
class AnalysisResponse(BaseModel):
    """Respuesta completa de un análisis"""
    id: str
    job_id: str
    object_key: str
    media_type: str
    status: str
    probability: Optional[float] = None
    is_synthetic: Optional[bool] = None
    confidence: Optional[str] = None
    suspected_type: Optional[str] = None
    model_version: Optional[str] = None
    result_details: Optional[dict] = None
    processing_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AnalysisListItem(BaseModel):
    """Item resumido para listado de análisis"""
    id: str
    job_id: str
    media_type: str
    status: str
    probability: Optional[float] = None
    is_synthetic: Optional[bool] = None
    confidence: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AnalysisListResponse(BaseModel):
    """Respuesta paginada de listado de análisis"""
    items: list[AnalysisListItem]
    total: int
    page: int
    page_size: int
    total_pages: int
