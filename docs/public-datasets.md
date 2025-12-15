# Bancos públicos (sugeridos) + cómo usarlos

No voy a scrapear Internet “a lo loco” porque suele violar licencias/ToS y te puede contaminar legalmente el proyecto.
Lo que sí dejo listo es:
- un pipeline para **ingestar** cualquier banco local que tengas (carpetas o datasets descargados)
- un entrenamiento que **solo persiste** el modelo si alcanza un umbral mínimo de calidad en validación

## Formato recomendado del banco

Estructura de carpetas:

- `dataset/ai_generated/*.jpg|png|webp`
- `dataset/not_ai_generated/*.jpg|png|webp`

## Scripts incluidos

- `scripts/ingest-folder.ps1`
  - Sube una carpeta completa con una etiqueta (`ai_generated` o `not_ai_generated`) al endpoint `/v1/dataset/upload`.
- `scripts/ingest-zip.ps1`
  - Sube un ZIP con muchas imágenes en una sola llamada al endpoint `/v1/dataset/upload-zip`.
- `scripts/ingest-zip-auto.ps1`
  - Sube un ZIP con estructura `ai_generated/` y `not_ai_generated/` al endpoint `/v1/dataset/upload-zip-auto`.
- `scripts/train-until-target.ps1`
  - Lanza `/v1/dataset/train` y espera a que termine.

## Datasets públicos típicos (para descargar tú)

Busca datasets que ya separen **real vs synthetic/AI** y que tengan licencia clara.
Ejemplos de familias de datasets que suelen servir para este problema:
- Datasets “AI vs Real” curados por la comunidad (muchos están en Hugging Face Datasets o Kaggle).
- Datasets multi-modelo (incluyen imágenes generadas por varios generadores), mejores para generalización.

Recomendación práctica:
- Empieza con un dataset mediano (decenas de miles) para validar pipeline.
- Luego incorpora uno grande y diverso para reducir overfitting.

## Cómo cargar un banco local

1) Arranca el stack

`docker compose up -d --build`

2) Obtén una API key (o usa JWT). Puedes verla en `/auth/me`.

3) Ingresa las carpetas

Ejemplo:

`powershell -File scripts/ingest-folder.ps1 -BaseUrl http://localhost:8000 -ApiKey <TU_API_KEY> -Label ai_generated -Folder C:\dataset\ai_generated`

`powershell -File scripts/ingest-folder.ps1 -BaseUrl http://localhost:8000 -ApiKey <TU_API_KEY> -Label not_ai_generated -Folder C:\dataset\not_ai_generated`

4) Entrena hasta el umbral

`powershell -File scripts/train-until-target.ps1 -BaseUrl http://localhost:8000 -ApiKey <TU_API_KEY>`

## Cómo cargar un ZIP (más rápido)

`powershell -File scripts/ingest-zip.ps1 -BaseUrl http://localhost:8000 -ApiKey <TU_API_KEY> -Label ai_generated -ZipPath C:\dataset\ai_generated.zip`

`powershell -File scripts/ingest-zip.ps1 -BaseUrl http://localhost:8000 -ApiKey <TU_API_KEY> -Label not_ai_generated -ZipPath C:\dataset\not_ai_generated.zip`

Notas:
- Hay límites por seguridad: `DATASET_ZIP_MAX_BYTES` (default 200MB) y `DATASET_ZIP_MAX_FILES` (default 5000).

## Cómo cargar un ZIP auto-etiquetado (un ZIP para ambas clases)

Estructura del ZIP:
- `ai_generated/.../*.jpg|png|webp`
- `not_ai_generated/.../*.jpg|png|webp`

Comando:

`powershell -File scripts/ingest-zip-auto.ps1 -BaseUrl http://localhost:8000 -ApiKey <TU_API_KEY> -ZipPath C:\dataset\ai_vs_real.zip`

## Umbral 0.85 (config)

El entrenamiento del head CLIP usa estos env vars:
- `CLIP_HEAD_TARGET_VAL_ACC` (default `0.85`)
- `CLIP_HEAD_MAX_ATTEMPTS` (default `3`)
- `CLIP_HEAD_MAX_STEPS` (default `1200`)

Puedes ponerlos en `.env` y reconstruir contenedores.
