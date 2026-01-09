# 🔍 Peritaciones.io - Detector de Imágenes IA

Sistema de detección de imágenes generadas por IA usando CLIP + clasificador lineal entrenado.

---

## 📋 Tabla de Contenidos

1. [Resumen del Proyecto](#-resumen-del-proyecto)
2. [Arquitectura del Sistema](#-arquitectura-del-sistema)
3. [Cómo Funciona la Detección](#-cómo-funciona-la-detección)
4. [Estado Actual del Proyecto](#-estado-actual-del-proyecto)
5. [Instalación y Configuración](#-instalación-y-configuración)
6. [Cómo Lanzar el Sistema](#-cómo-lanzar-el-sistema)
7. [Cómo Entrenar el Modelo](#-cómo-entrenar-el-modelo)
8. [Cómo Testear la Precisión](#-cómo-testear-la-precisión)
9. [Datasets Recomendados](#-datasets-recomendados)
10. [Próximos Pasos](#-próximos-pasos)
11. [Troubleshooting](#-troubleshooting)

---

## 🎯 Resumen del Proyecto

### ¿Qué hace?
Detecta si una imagen ha sido generada por IA (Stable Diffusion, DALL-E, MidJourney, GANs, etc.) con alta precisión.

### Tecnologías Principales
- **Backend**: FastAPI + Celery + Redis + PostgreSQL + MinIO
- **Frontend**: Next.js (React)
- **ML**: CLIP ViT-L/14 + Clasificador Lineal Entrenado
- **Contenedores**: Docker Compose

### Precisión Actual
| Tipo de Imagen | Precisión |
|---------------|-----------|
| Stable Diffusion | **100%** ✅ |
| GANs (StyleGAN, etc.) | ~50% ⚠️ |
| DALL-E / MidJourney | No probado |

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│                      http://localhost:3000                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API (FastAPI)                             │
│                      http://localhost:8000                       │
│   • POST /analyze - Sube imagen y crea job                      │
│   • GET /jobs/{id} - Consulta resultado                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ Celery Task
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Worker (Celery)                           │
│   • CLIPDetector - Detecta IA usando CLIP + cabeza entrenada    │
│   • Cada 30s recarga la cabeza desde PostgreSQL                 │
│   • GPU: CUDA si disponible                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
│   MinIO     │    │   PostgreSQL    │    │    Redis    │
│  (Imágenes) │    │ (Cabeza modelo) │    │   (Cola)    │
│  :9000      │    │    :5432        │    │   :6379     │
└─────────────┘    └─────────────────┘    └─────────────┘
```

### Servicios Docker
| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| `api` | 8000 | API REST FastAPI |
| `worker` | - | Celery Worker con GPU |
| `frontend` | 3000 | Next.js UI |
| `db` | 5432 | PostgreSQL |
| `redis` | 6379 | Broker Celery |
| `minio` | 9000/9001 | Storage S3 |

---

## 🧠 Cómo Funciona la Detección

### Flujo de Análisis

```
1. IMAGEN (RGB)
      │
      ▼
2. CLIP ViT-L/14 (OpenAI, Frozen)
      │ Extrae vector de 768 dimensiones
      ▼
3. CLASIFICADOR LINEAL (Entrenado)
      │ 768 → 1 (sigmoid)
      ▼
4. PROBABILIDAD [0.0 - 1.0]
      │
      ▼
5. DECISIÓN: prob > 0.5 → "IA Detectada"
```

### Componentes Clave

#### 1. CLIP (Congelado)
- **Modelo**: ViT-L/14 de OpenAI
- **Función**: Extractor de características visuales
- **NO se modifica** durante el entrenamiento

#### 2. Cabeza Clasificadora (Entrenada)
- **Arquitectura**: `Linear(768 → 1) + Sigmoid`
- **Almacenamiento**: Tabla `clip_head_global` en PostgreSQL
- **Hot-Reload**: El worker recarga cada 30 segundos

### Archivos Importantes

```
backend/app/detectors/
├── clip_detector.py      # ⭐ DETECTOR PRINCIPAL - Usa cabeza entrenada
├── image_detector.py     # ⚠️ NO USAR - Usa pesos genéricos
├── base_detector.py      # Clase base abstracta
└── ensemble_detector.py  # Combina múltiples detectores
```

#### ⚠️ ERROR COMÚN
```python
# ❌ INCORRECTO - Usa pesos genéricos, NO la cabeza entrenada
from app.detectors.image_detector import ImageDetector
detector = ImageDetector()

# ✅ CORRECTO - Usa la cabeza entrenada de la BD
from app.detectors.clip_detector import CLIPDetector
detector = CLIPDetector()
```

---

## 📊 Estado Actual del Proyecto

### ✅ Completado
- [x] Arquitectura Docker completa
- [x] API REST funcional
- [x] Sistema de entrenamiento
- [x] Hot-reload de cabeza desde BD
- [x] 100% precisión en Stable Diffusion
- [x] Scripts de testing

### ⚠️ Pendiente
- [ ] **Entrenar con más generadores** (GANs, DALL-E, MidJourney)
- [ ] Verificar precisión multi-generador
- [ ] Frontend completo
- [ ] Sistema de feedback/calibración

### Cabeza Actual
- **Versión**: `v20251218111059`
- **Precisión Validación**: 98.43%
- **Dataset Entrenamiento**: DiffusionDB (SD) + Hemg (Real)
- **Limitación**: Solo funciona bien con Stable Diffusion

---

## 🚀 Instalación y Configuración

### Requisitos Previos
- Docker Desktop
- Git
- GPU NVIDIA (recomendado) con drivers CUDA

### Clonar Repositorio
```powershell
git clone https://github.com/teferi26/Deepfakes.git
cd Deepfakes
```

### Configurar Variables
```powershell
copy .env.example .env
# Editar .env si es necesario
```

### Puertos Requeridos
- 3000: Frontend
- 8000: API
- 5432: PostgreSQL
- 6379: Redis
- 9000/9001: MinIO

---

## 🎮 Cómo Lanzar el Sistema

### 1. Iniciar Todo
```powershell
docker-compose up --build -d
```

### 2. Verificar que todo esté corriendo
```powershell
docker-compose ps
```

Deberías ver:
```
NAME                    STATUS
peritaciones-api-1      Up
peritaciones-worker-1   Up
peritaciones-redis-1    Up
peritaciones-db-1       Up
peritaciones-minio-1    Up
peritaciones-frontend-1 Up
```

### 3. Acceder a los Servicios
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000
- **MinIO Console**: http://localhost:9001

### 4. Probar la API
```powershell
# Subir imagen para análisis
curl -X POST http://localhost:8000/analyze `
  -F "file=@imagen.jpg" `
  -H "Content-Type: multipart/form-data"
```

### 5. Ver Logs
```powershell
# Todos los servicios
docker-compose logs -f

# Solo el worker (donde ocurre la detección)
docker-compose logs -f worker
```

### 6. Detener Todo
```powershell
docker-compose down
```

---

## 🎓 Cómo Entrenar el Modelo

### Entrenamiento Básico (Solo Stable Diffusion)
```powershell
docker-compose exec worker python scripts/train_diffusion_dataset.py
```

### Entrenamiento Universal (Múltiples Generadores)
```powershell
# Script ya creado pero NO ejecutado aún
docker-compose exec worker python scripts/train_universal_detector.py
```

Este script entrena con:
- **DiffusionDB**: Stable Diffusion (50K imágenes)
- **CIFAKE**: CIFAR sintético (100K imágenes)
- **Hemg AI**: Variedad de generadores

### Parámetros Importantes
Editar en el script de entrenamiento:
```python
EPOCHS = 10          # Más épocas = más precisión (hasta cierto punto)
BATCH_SIZE = 32      # Reducir si hay errores de memoria GPU
LEARNING_RATE = 1e-4 # Ajustar si el loss no baja
```

### ¿Cómo Saber si Está Funcionando?
```
[Epoch 1/10] Batch 50/1000 - Loss: 0.693  ← Debe empezar cerca de 0.693
[Epoch 1/10] Batch 100/1000 - Loss: 0.521 ← Debe ir bajando
...
[Epoch 10/10] Val Accuracy: 95.2%  ← Objetivo: >90%
```

### Después del Entrenamiento
La nueva cabeza se guarda automáticamente en PostgreSQL.
El worker la recarga en los próximos 30 segundos.

---

## 🧪 Cómo Testear la Precisión

### Test Rápido (100 IA + 100 Real)
```powershell
docker-compose exec worker python scripts/test_accuracy_diffusiondb.py
```

Salida esperada:
```
=== RESULTADOS FINALES ===
Imágenes IA correctas: 100/100 (100.0%)
Imágenes REALES correctas: 100/100 (100.0%)
PRECISIÓN TOTAL: 200/200 (100.0%)
```

### Test Manual con Imagen Específica
```powershell
docker-compose exec worker python -c "
from app.detectors.clip_detector import CLIPDetector
from PIL import Image

detector = CLIPDetector()
result = detector.analyze('test.jpg')
prob = result['probability']
print(f'Probabilidad IA: {prob:.2%}')
print(f'Decisión: {'IA' if prob > 0.5 else 'Real'}')
"
```

---

## 📚 Datasets Recomendados

### Para Entrenar

| Dataset | Contenido | Tamaño | Link |
|---------|-----------|--------|------|
| DiffusionDB | Stable Diffusion | 14M+ | [HuggingFace](https://huggingface.co/datasets/poloclub/diffusiondb) |
| CIFAKE | CIFAR sintético | 120K | [HuggingFace](https://huggingface.co/datasets/CIKM/CIFAKE-image-classification) |
| GenImage | Multi-generador | 1.3M | [GitHub](https://github.com/GenImage-Dataset/GenImage) |
| DRCT-2M | DALL-E, SD, MJ | 2M | [HuggingFace](https://huggingface.co/datasets/whluo/DRCT-2M) |

### Para Testear

| Dataset | Contenido | Link |
|---------|-----------|------|
| Hemg | Real + Fake | [HuggingFace](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets) |
| ArtiFact | Multi-generador | [HuggingFace](https://huggingface.co/datasets/awsaf49/artifact-dataset) |

---

## 📋 Próximos Pasos

### Prioridad 1: Entrenamiento Universal
```powershell
# 1. Asegurar que Docker esté estable
docker-compose up -d

# 2. Ejecutar entrenamiento universal
docker-compose exec worker python scripts/train_universal_detector.py
```

**Tiempo estimado**: 1-2 horas (descarga datasets + entrenamiento)

### Prioridad 2: Verificar Mejora
```powershell
# Después del entrenamiento, testear con diferentes tipos
docker-compose exec worker python scripts/test_accuracy_200.py
```

### Prioridad 3: Frontend
- Completar UI de subida de imágenes
- Mostrar resultados con explicación
- Dashboard de historial

---

## 🔧 Troubleshooting

### El worker no arranca
```powershell
# Ver logs del worker
docker-compose logs worker

# Reiniciar solo el worker
docker-compose restart worker
```

### Error de memoria GPU
```python
# Reducir batch size en el script de entrenamiento
BATCH_SIZE = 16  # En lugar de 32
```

### La cabeza no se recarga
```powershell
# Forzar recarga reiniciando worker
docker-compose restart worker
```

### Precisión del 50% (aleatorio)
1. Verificar que usas `CLIPDetector`, NO `ImageDetector`
2. Verificar que hay una cabeza en la BD:
```sql
SELECT version, val_accuracy FROM clip_head_global ORDER BY created_at DESC LIMIT 1;
```

### Docker tarda mucho en arrancar
```powershell
# Rebuild solo lo necesario
docker-compose build --no-cache worker
docker-compose up -d
```

---

## 📁 Estructura del Proyecto

```
├── backend/
│   ├── app/
│   │   ├── detectors/
│   │   │   ├── clip_detector.py    # ⭐ Detector principal
│   │   │   └── image_detector.py   # ⚠️ No usar
│   │   ├── main.py                 # API FastAPI
│   │   └── models.py               # Modelos SQLAlchemy
│   └── scripts/
│       ├── train_diffusion_dataset.py      # Entrenamiento SD
│       ├── train_universal_detector.py     # Entrenamiento multi
│       └── test_accuracy_diffusiondb.py    # Testing
├── frontend/                       # Next.js
├── docs/
│   └── GUIA-COMPLETA-DETECTOR-IA.md  # Documentación detallada
└── docker-compose.yml
```

---

## 📄 Documentación Adicional

Para documentación técnica más detallada, ver:
- [docs/GUIA-COMPLETA-DETECTOR-IA.md](docs/GUIA-COMPLETA-DETECTOR-IA.md)
- [docs/alcance-mvp.md](docs/alcance-mvp.md)
- [docs/public-datasets.md](docs/public-datasets.md)

---

## 🔒 Licencia

Pendiente de definir.
