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

## ¿Qué mira el detector para decidir “IA / NO IA”?

Este MVP usa un **ensemble** (varias señales) y combina sus resultados:

- **CLIP (UniversalFakeDetect)**: extrae *features semánticas* globales con CLIP (ViT-L/14) y aplica un clasificador lineal entrenado para separar **real vs sintético**. No “lee prompts”; aprende patrones estadísticos y semánticos presentes en generadores.
- **CNNDetect (ResNet50)**: busca *artefactos de generación/upsampling* y texturas locales típicas de ciertos pipelines (especialmente GANs), además de inconsistencias sutiles.
- **Frecuencia (FFT/DCT heurístico)**: analiza el *espectro de frecuencias* (ruido/altas frecuencias/picos) que a veces delata síntesis o postprocesado agresivo.
- **Metadatos/estructura (EXIF/JPEG)**: señales forenses como metadatos inconsistentes/ausentes o patrones de compresión/estructura. Es una señal auxiliar (no prueba IA por sí sola).

La API devuelve:

- `probability`: probabilidad continua $[0,1]$.
- `ai_decision`: veredicto:
	- `ai_generated` si `probability >= 0.65`
	- `not_ai_generated` si `probability <= 0.35`
	- `inconclusive` en el rango intermedio

## ¿Cómo aprende de los errores?

El sistema aprende de forma incremental con el feedback del usuario (sin “reentrenar CLIP” en producción):

- Cada vez que el usuario pulsa **“Es IA”** o **“No es IA”**, se guarda un registro en la tabla `analysis_feedback`.
- El guardado es **idempotente**: 1 feedback por usuario+análisis (si se reenvía, se actualiza).
- Con suficientes ejemplos de ambas clases, se encola un entrenamiento automático (`train.ensemble_calibrator`) que aprende un **calibrador** ligero (logistic regression) con las señales del ensemble.
- Ese calibrador se guarda en Postgres (`model_artifacts`) y el detector lo carga automáticamente para refinar la probabilidad final. Si aún no hay calibrador, el sistema usa el promedio ponderado del ensemble.

## Estado del MVP
- Endpoints de salud y creación/consulta de jobs (detección simulada).
- UI mínima de bienvenida.
- Pendiente: integración de detector real, uploads binarios, seguridad avanzada, métricas, panel, pagos.

## Licencia
Pendiente de definir.
