from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult

from .config import settings
from .schemas import AnalyzeRequest, JobResponse
from .tasks.analyze import analyze_media

app = FastAPI(title="Fraud Detector API", version="0.1.0")

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


@app.post("/v1/analyze", response_model=JobResponse)
def create_analysis(req: AnalyzeRequest) -> JobResponse:
    task = analyze_media.delay(req.media_type, req.reference)
    return JobResponse(job_id=task.id, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    result = AsyncResult(job_id)
    status = result.status.lower()

    if status == "failure":
        raise HTTPException(status_code=500, detail="Job failed")

    return JobResponse(job_id=job_id, status=status, result=result.result if result.ready() else None)
