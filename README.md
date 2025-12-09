# Plataforma IA de Detección de Fraude Audiovisual

MVP con FastAPI + Celery + Redis + PostgreSQL + MinIO (S3) + Next.js.

## Estructura
- `backend/`: API FastAPI y workers Celery.
- `frontend/`: Next.js (UI bilingüe básica).
- `docs/`: documentación funcional y técnica.
- `docker-compose.yml`: entorno local con Postgres, Redis, MinIO, API, worker y frontend.

## Requisitos
- Docker y Docker Compose.
- Puertos libres: 3000 (frontend), 8000 (API), 5432 (Postgres), 6379 (Redis), 9000/9001 (MinIO).

## Uso rápido
```powershell
# 1) Copia y ajusta variables
copy .env.example .env

# 2) Levanta el entorno
docker compose up --build

# 3) API en http://localhost:8000/docs
#    Frontend en http://localhost:3000
```

## Servicios (compose)
- `api`: FastAPI con endpoints de salud y creación de jobs de análisis (simulado).
- `worker`: Celery worker para procesar jobs.
- `redis`: broker/back-end de Celery.
- `db`: PostgreSQL para metadatos.
- `minio`: almacenamiento S3-compatible.
- `frontend`: Next.js con página básica de upload (placeholder).

## Variables de entorno
Ver `.env.example` para valores iniciales.

## Estado del MVP
- Endpoints de salud y creación/consulta de jobs (detección simulada).
- UI mínima de bienvenida.
- Pendiente: integración de detector real, uploads binarios, seguridad avanzada, métricas, panel, pagos.

## Licencia
Pendiente de definir.
