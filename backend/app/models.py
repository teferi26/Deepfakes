import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, Float, Boolean, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import enum

from .database import Base


class UserRole(str, enum.Enum):
    user = "user"
    admin = "admin"


class AnalysisStatus(str, enum.Enum):
    """Estado del análisis."""
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    api_key = Column(String(64), unique=True, index=True, nullable=True)
    role = Column(SQLEnum(UserRole), default=UserRole.user, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relación con análisis
    analyses = relationship("Analysis", back_populates="user")

    def __repr__(self):
        return f"<User {self.email}>"


class Analysis(Base):
    """Registro de análisis realizados para historial."""
    __tablename__ = "analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Info del archivo
    filename = Column(String(255), nullable=True)
    object_key = Column(String(512), nullable=False)
    media_type = Column(String(20), nullable=False)  # "image" o "video"
    file_hash = Column(String(64), nullable=True)
    file_size = Column(String(20), nullable=True)
    
    # Resultados
    probability = Column(Float, nullable=True)
    is_synthetic = Column(Boolean, nullable=True)
    confidence = Column(String(20), nullable=True)
    suspected_type = Column(Text, nullable=True)
    model_version = Column(String(50), default="ensemble-v1", nullable=True)
    
    # Detalles completos en JSON (incluye individual_results)
    result_details = Column(JSONB, nullable=True)
    
    # Estado
    status = Column(SQLEnum(AnalysisStatus), default=AnalysisStatus.pending, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Tiempos
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    
    # Relación con usuario
    user = relationship("User", back_populates="analyses")
    
    def __repr__(self):
        return f"<Analysis {self.job_id} - {self.status}>"
