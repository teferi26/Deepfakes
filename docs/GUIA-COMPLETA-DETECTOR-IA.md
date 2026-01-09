# 🔍 Guía Completa: Detector de Imágenes Generadas por IA

## Índice
1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Cómo Funciona la Detección](#3-cómo-funciona-la-detección)
4. [Estado Actual del Proyecto](#4-estado-actual-del-proyecto)
5. [Cómo Lanzar el Sistema](#5-cómo-lanzar-el-sistema)
6. [Cómo Entrenar el Modelo](#6-cómo-entrenar-el-modelo)
7. [Cómo Verificar la Precisión](#7-cómo-verificar-la-precisión)
8. [Próximos Pasos](#8-próximos-pasos)
9. [Datasets Recomendados](#9-datasets-recomendados)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Resumen Ejecutivo

### ¿Qué es este sistema?
Un detector de imágenes generadas por IA que utiliza **CLIP ViT-L/14** (modelo de OpenAI) combinado con un clasificador lineal entrenado para distinguir imágenes reales de sintéticas.

### Resultados Actuales
| Tipo de Imagen | Accuracy |
|----------------|----------|
| Stable Diffusion | **100%** |
| Imágenes Reales | **100%** |
| GANs (StyleGAN, ProGAN) | ~50% (necesita entrenamiento) |
| DALL-E / MidJourney | ~50% (necesita entrenamiento) |

### Problema Principal Identificado
El modelo actual está **sobreentrenado con Stable Diffusion**. Funciona perfecto para ese generador pero no generaliza a otros (GANs, DALL-E, MidJourney).

---

## 2. Arquitectura del Sistema

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                        │
│                         Puerto: 3000                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API (FastAPI)                             │
│                         Puerto: 8000                             │
│  - Recibe imágenes                                               │
│  - Crea tareas de análisis                                       │
│  - Devuelve resultados                                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        WORKER (Celery)                           │
│                         Con GPU CUDA                             │
│  - Ejecuta análisis de imágenes                                  │
│  - Entrena modelos                                               │
│  - Usa CLIP + Clasificador                                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Redis   │   │ Postgres │   │  MinIO   │
        │  Cola    │   │    DB    │   │  Storage │
        └──────────┘   └──────────┘   └──────────┘
```

### Detectores Disponibles

El sistema usa un **ensemble** de 4 detectores:

| Detector | Peso | Descripción |
|----------|------|-------------|
| `clip_universal` | 70% | CLIP ViT-L/14 + Clasificador lineal entrenado |
| `cnn_detect` | 15% | Red CNN para detección de artefactos |
| `frequency_analysis` | 10% | Análisis de frecuencias (FFT) |
| `metadata_analysis` | 5% | Análisis de metadatos EXIF |

**El detector principal es `clip_universal`** - es el que se entrena y el más preciso.

---

## 3. Cómo Funciona la Detección

### Flujo de Análisis de una Imagen

```
1. IMAGEN DE ENTRADA (JPEG/PNG/WebP)
          │
          ▼
2. PREPROCESAMIENTO
   - Convertir a RGB
   - Redimensionar a 224x224
   - Normalizar valores de píxeles
          │
          ▼
3. EXTRACCIÓN DE FEATURES (CLIP ViT-L/14)
   - CLIP extrae un vector de 768 dimensiones
   - Este vector captura la "esencia visual" de la imagen
   - CLIP está CONGELADO (no se entrena)
          │
          ▼
4. CLASIFICACIÓN (Head Entrenado)
   - Un clasificador lineal (768 → 1)
   - Toma el vector de CLIP
   - Devuelve probabilidad [0, 1]
          │
          ▼
5. DECISIÓN
   - probability > 0.5 → Imagen AI
   - probability < 0.5 → Imagen Real
```

### ¿Por Qué Funciona?

CLIP fue entrenado con 400 millones de pares imagen-texto. Esto le da una comprensión profunda de:
- Texturas naturales vs artificiales
- Patrones de iluminación realistas
- Coherencia semántica
- Artefactos típicos de generadores AI

El clasificador lineal aprende a distinguir los patrones en las features de CLIP que son característicos de imágenes generadas.

### Código Clave

```python
# backend/app/detectors/clip_detector.py

# 1. Extraer features con CLIP
features = self.clip_model.encode_image(input_tensor)
features = F.normalize(features, dim=-1)

# 2. Clasificar
logits = self.classifier(features)
probability = torch.sigmoid(logits).item()

# 3. Decisión
is_synthetic = probability > 0.5
```

---

## 4. Estado Actual del Proyecto

### ✅ Lo que Funciona

1. **API REST completa** en FastAPI
2. **Frontend funcional** en Next.js
3. **Sistema de colas** con Celery + Redis
4. **Detección de Stable Diffusion** con 100% accuracy
5. **Hot-reload del modelo** desde base de datos
6. **Sistema de calibración** del ensemble

### ⚠️ Lo que Necesita Mejora

1. **Generalización a otros generadores AI**
   - El head actual solo detecta bien Stable Diffusion
   - GANs (StyleGAN, ProGAN) no se detectan
   - DALL-E, MidJourney no se detectan

2. **Entrenamiento con múltiples datasets**
   - Script `train_universal_detector.py` creado pero no ejecutado completamente

### 📊 Métricas del Head Actual

```
Head version: 20251218111059
Head source: db
Accuracy validación: 98.43%
Dataset entrenamiento: DiffusionDB (Stable Diffusion) + Hemg (Real)
Samples: 40,000 (20K AI + 20K Real)
```

---

## 5. Cómo Lanzar el Sistema

### Prerrequisitos
- Docker Desktop con WSL2
- GPU NVIDIA con drivers CUDA
- ~20GB espacio en disco

### Paso 1: Levantar todos los servicios

```powershell
# Desde la raíz del proyecto
cd "C:\Users\tefer\OneDrive\Documentos\SPRINGMARKET\CLIENTES\PAGINAS WEB\PERITCAIONES.IO"

# Levantar todo
docker-compose up -d

# Verificar que todo está corriendo
docker-compose ps
```

Deberías ver:
```
NAME                      STATUS
peritcaionesio-api-1      Up
peritcaionesio-worker-1   Up
peritcaionesio-redis-1    Up
peritcaionesio-db-1       Up
peritcaionesio-minio-1    Up
peritcaionesio-frontend-1 Up
```

### Paso 2: Verificar que el detector funciona

```powershell
# Test rápido
docker-compose exec api python test_ensemble.py
```

### Paso 3: Acceder a la aplicación

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001

### Paso 4: Analizar una imagen vía API

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@imagen.jpg"
```

---

## 6. Cómo Entrenar el Modelo

### Opción A: Entrenamiento Rápido (Solo Stable Diffusion)

```powershell
# Copiar script al contenedor
docker cp backend\scripts\train_diffusion_dataset.py peritcaionesio-worker-1:/app/scripts/

# Ejecutar entrenamiento
docker-compose exec worker python scripts/train_diffusion_dataset.py
```

**Tiempo estimado**: 15-30 minutos
**Resultado**: ~98% accuracy en Stable Diffusion

### Opción B: Entrenamiento Universal (Múltiples Generadores)

```powershell
# Copiar script al contenedor
docker cp backend\scripts\train_universal_detector.py peritcaionesio-worker-1:/app/scripts/

# Ejecutar entrenamiento (tarda más porque descarga más datasets)
docker-compose exec worker python scripts/train_universal_detector.py
```

**Tiempo estimado**: 1-2 horas (incluye descarga de datasets)
**Resultado esperado**: >90% accuracy en múltiples generadores

### Qué hace el entrenamiento

1. **Carga datasets** de HuggingFace (DiffusionDB, CIFAKE, Hemg, etc.)
2. **Extrae features** con CLIP para todas las imágenes
3. **Entrena clasificador lineal** con SGD
4. **Guarda el modelo** en la base de datos como `clip_head_global`
5. El worker **recarga automáticamente** el nuevo head (hot-reload cada 30s)

### Estructura del Head Guardado

```python
{
    'state_dict': {
        'fc.weight': tensor([...]),  # 768 valores
        'fc.bias': tensor([...])     # 1 valor
    },
    'architecture': 'linear',
    'in_dim': 768
}
```

---

## 7. Cómo Verificar la Precisión

### Test con DiffusionDB (Stable Diffusion)

```powershell
# Copiar script de test
docker cp backend\scripts\test_accuracy_diffusiondb.py peritcaionesio-api-1:/app/scripts/

# Ejecutar test
docker-compose exec api python scripts/test_accuracy_diffusiondb.py
```

### Test con Dataset Hemg (Múltiples Generadores)

```powershell
# Copiar script de test
docker cp backend\scripts\test_accuracy_200.py peritcaionesio-api-1:/app/scripts/

# Ejecutar test
docker-compose exec api python scripts/test_accuracy_200.py
```

### Verificar versión del head actual

```powershell
docker-compose exec api python -c "
from app.detectors.clip_detector import CLIPDetector
d = CLIPDetector()
print(f'Head version: {d._head_version}')
print(f'Head source: {d._head_source}')
"
```

### Ver todos los heads entrenados

```powershell
docker-compose exec api python -c "
from app.database import SessionLocal
from app.models import ModelArtifact

db = SessionLocal()
heads = db.query(ModelArtifact).filter(ModelArtifact.name == 'clip_head_global').order_by(ModelArtifact.created_at.desc()).all()
for h in heads:
    print(f'v{h.version} - Accuracy: {h.metrics.get(\"val_accuracy\", \"N/A\")}')
"
```

---

## 8. Próximos Pasos

### Prioridad Alta 🔴

1. **Ejecutar entrenamiento universal completo**
   ```powershell
   docker-compose exec worker python scripts/train_universal_detector.py
   ```
   - Incluye: Stable Diffusion, GANs (StyleGAN, ProGAN), CIFAKE
   - Target: >90% accuracy en todos los generadores

2. **Agregar más datasets de GANs modernos**
   - DALL-E 3
   - MidJourney v5/v6
   - Flux
   - Adobe Firefly

### Prioridad Media 🟡

3. **Implementar data augmentation** durante entrenamiento
   - JPEG compression
   - Resize
   - Crop
   - Esto mejora robustez del modelo

4. **Cambiar arquitectura del clasificador**
   - Probar MLP de 2 capas en lugar de lineal
   - Ya está implementado como `CLIPMLPClassifier`

### Prioridad Baja 🟢

5. **Fine-tuning parcial de CLIP**
   - Descongelar últimas capas de CLIP
   - Requiere más VRAM y tiempo

6. **Ensemble con otros modelos**
   - Agregar DIRE (Diffusion Reconstruction Error)
   - Agregar CNNDetect

---

## 9. Datasets Recomendados

### Para Imágenes AI

| Dataset | Generadores | Samples | Link |
|---------|-------------|---------|------|
| **DiffusionDB** | Stable Diffusion | 14M | https://huggingface.co/datasets/poloclub/diffusiondb |
| **CIFAKE** | StyleGAN, ProGAN | 120K | https://huggingface.co/datasets/CIKM/CIFAKE-image-classification |
| **Hemg AI** | Varios GANs | ~75K | https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets |
| **AI Art Gallery** | DALL-E, MidJourney, SD | 50K+ | https://huggingface.co/datasets/alfredplpl/artstation-stable-diffusion-dataset |
| **GenImage** | 8 generadores | 1.35M | https://github.com/GenImage-Dataset/GenImage |
| **DRCT-2M** | DALL-E 3, MJ, SD3 | 2M | https://huggingface.co/datasets/whluo/DRCT-2M |

### Para Imágenes Reales

| Dataset | Tipo | Samples | Link |
|---------|------|---------|------|
| **Hemg Real** | Fotos naturales | ~75K | https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets |
| **CIFAKE Real** | CIFAR real | 60K | https://huggingface.co/datasets/CIKM/CIFAKE-image-classification |
| **LAION** | Web images | Millones | https://laion.ai/blog/laion-5b/ |
| **Unsplash** | Fotos profesionales | 25K | https://unsplash.com/data |

### Cómo Usar un Nuevo Dataset

```python
from datasets import load_dataset

# Cargar dataset
ds = load_dataset('nombre/dataset', split='train')

# Ver estructura
print(ds.features)

# Iterar sobre imágenes
for item in ds:
    image = item['image']  # PIL Image
    label = item['label']  # 0=AI, 1=Real (varía por dataset)
```

---

## 10. Troubleshooting

### Docker no conecta

```powershell
# Reiniciar Docker Desktop
# O ejecutar:
wsl --shutdown
# Luego abrir Docker Desktop de nuevo
```

### Worker sin GPU

```powershell
# Verificar que CUDA está disponible
docker-compose exec worker python -c "import torch; print(torch.cuda.is_available())"
```

Debe imprimir `True`. Si no:
- Verificar drivers NVIDIA
- Verificar que Docker tiene acceso a GPU en settings

### El modelo no se actualiza después de entrenar

```powershell
# El worker recarga el head cada 30 segundos automáticamente
# Para forzar recarga, reiniciar el worker:
docker-compose restart worker
```

### Error "Out of Memory" durante entrenamiento

Reducir batch size en el script de entrenamiento:
```python
BATCH_SIZE = 32  # En lugar de 64
```

### Ver logs del sistema

```powershell
# Logs del API
docker-compose logs -f api

# Logs del worker
docker-compose logs -f worker

# Todos los logs
docker-compose logs -f
```

---

## Apéndice: Archivos Importantes

```
backend/
├── app/
│   ├── detectors/
│   │   ├── clip_detector.py      # ⭐ Detector principal CLIP
│   │   ├── image_detector.py     # Detector base (NO usar directamente)
│   │   ├── ensemble_detector.py  # Ensemble de todos los detectores
│   │   └── weights/
│   │       └── fc_weights.pth    # Pesos originales UniversalFakeDetect
│   ├── tasks/
│   │   ├── analyze.py            # Tarea Celery de análisis
│   │   └── training.py           # Tarea Celery de entrenamiento
│   └── models.py                 # Modelos de DB (ModelArtifact)
├── scripts/
│   ├── train_diffusion_dataset.py    # Entrenamiento con SD
│   ├── train_universal_detector.py   # Entrenamiento universal
│   ├── test_accuracy_200.py          # Test con Hemg
│   └── test_accuracy_diffusiondb.py  # Test con DiffusionDB
└── test_ensemble.py                  # Test rápido del ensemble
```

---

## Contacto y Soporte

Para dudas sobre este proyecto, revisar:
1. Este documento
2. Los comentarios en el código
3. La documentación de los datasets en HuggingFace

---

*Documento generado el 9 de Enero de 2026*
*Versión del sistema: 1.0.0*
*Head actual: v20251218111059*
