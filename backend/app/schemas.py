from typing import Literal, Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    media_type: Literal["image", "video"] = Field(description="Tipo de medio a analizar")
    reference: str = Field(description="Referencia/ID/URL temporal del archivo almacenado")


class JobResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
