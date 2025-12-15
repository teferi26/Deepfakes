from celery import Celery
from .config import settings

celery = Celery(
    "fraud-detector",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.analyze", "app.tasks.training"],
)

celery.conf.update(task_track_started=True, result_extended=True)
