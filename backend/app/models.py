import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, Float, Boolean, Text, ForeignKey, Integer, LargeBinary, UniqueConstraint
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

    # Feedback del usuario sobre este análisis
    feedback_items = relationship("AnalysisFeedback", back_populates="analysis", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Analysis {self.job_id} - {self.status}>"


class FeedbackLabel(str, enum.Enum):
    ai_generated = "ai_generated"
    not_ai_generated = "not_ai_generated"


class AnalysisFeedback(Base):
    """Feedback del usuario para un análisis.

    Guarda la etiqueta real (según el usuario) y una foto del estado del
    modelo en ese momento (probabilidad/decisión), para poder recalibrar.
    """

    __tablename__ = "analysis_feedback"

    __table_args__ = (
        UniqueConstraint("analysis_id", "user_id", name="uq_feedback_analysis_user"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    label = Column(SQLEnum(FeedbackLabel), nullable=False)
    was_model_correct = Column(Boolean, nullable=True)
    comment = Column(Text, nullable=True)

    # Snapshot del resultado del modelo en el momento del feedback
    probability_at_time = Column(Float, nullable=True)
    ai_decision_at_time = Column(String(32), nullable=True)
    model_version_at_time = Column(String(50), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis = relationship("Analysis", back_populates="feedback_items")
    user = relationship("User")


class ModelArtifact(Base):
    """Artefactos entrenados y persistidos en DB.

    Se usa para guardar el calibrador entrenado con feedback, de forma que el
    sistema realmente "aprenda" y no dependa de ficheros dentro del contenedor.
    """

    __tablename__ = "model_artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(128), index=True, nullable=False)
    version = Column(String(64), index=True, nullable=False)
    artifact = Column(LargeBinary, nullable=False)
    metrics = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class UserCalibration(Base):
    """Umbrales calibrados por usuario basados en feedback acumulado."""

    __tablename__ = "user_calibration"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True, index=True)

    threshold_low = Column(Float, nullable=False, default=0.35)
    threshold_high = Column(Float, nullable=False, default=0.65)

    samples_total = Column(Integer, nullable=False, default=0)
    samples_ai = Column(Integer, nullable=False, default=0)
    samples_not_ai = Column(Integer, nullable=False, default=0)

    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User")


class TrainingSample(Base):
    """Muestra etiquetada para entrenamiento supervisado.

    Se alimenta con:
    - Banco de imágenes (dataset) subido explícitamente.
    - (Opcional) reutilización de análisis con feedback.
    """

    __tablename__ = "training_samples"

    __table_args__ = (
        UniqueConstraint("object_key", "label", "source", name="uq_training_sample_key_label_source"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)

    object_key = Column(String(512), nullable=False, index=True)
    filename = Column(String(255), nullable=True)
    label = Column(SQLEnum(FeedbackLabel), nullable=False)
    source = Column(String(32), nullable=False, default="dataset")  # dataset|feedback

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User")
