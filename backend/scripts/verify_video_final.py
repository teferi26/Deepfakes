#!/usr/bin/env python3
"""
Verificación final del modelo de video IA con datos reales de HuggingFace.
Usa DiffusionDB para IA y CIFAR-10 para reales.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import psycopg2
import pickle
import clip
from PIL import Image

# ============================================================
# CONFIGURACIÓN
# ============================================================
TEST_SIZE_PER_CLASS = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# MODELO
# ============================================================

class VideoClassifier(nn.Module):
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# CARGAR MODELO Y CLIP
# ============================================================

print("\n📦 Cargando CLIP...")
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()
print("   ✓ CLIP listo")

print("\n📥 Cargando modelo desde PostgreSQL...")
conn = psycopg2.connect(
    host=os.environ.get("POSTGRES_HOST", "db"),
    database=os.environ.get("POSTGRES_DB", "peritaciones"),
    user=os.environ.get("POSTGRES_USER", "postgres"),
    password=os.environ.get("POSTGRES_PASSWORD", "postgres")
)
cur = conn.cursor()

cur.execute("""
    SELECT model_data, accuracy, source, created_at 
    FROM video_heads 
    ORDER BY accuracy DESC, created_at DESC 
    LIMIT 1
""")
result = cur.fetchone()
cur.close()
conn.close()

if not result:
    print("   ❌ No se encontró modelo en video_heads")
    sys.exit(1)

model_info = pickle.loads(result[0])
print(f"   ✓ Modelo encontrado:")
print(f"     - Accuracy: {result[1]}%")
print(f"     - Fuente: {result[2]}")
print(f"     - Creado: {result[3]}")

# Crear y cargar modelo
classifier = VideoClassifier().to(device)
classifier.load_state_dict(model_info["state_dict"])
classifier.eval()


# ============================================================
# DESCARGAR DATOS DE TEST
# ============================================================

print(f"\n📥 Descargando {TEST_SIZE_PER_CLASS} samples de cada clase...")

from datasets import load_dataset

# Cargar datasets
print("   Cargando DiffusionDB (IA)...")
ds_ai = load_dataset("poloclub/diffusiondb", "2m_random_1k", split="train")

print("   Cargando CIFAR-10 (real)...")
ds_real = load_dataset("cifar10", split="test")

# Tomar muestras (diferente split que el entrenamiento)
ai_images = [s["image"].convert("RGB") for s in list(ds_ai)[500:500+TEST_SIZE_PER_CLASS]]
real_images = [s["img"].convert("RGB").resize((224, 224)) for s in list(ds_real)[:TEST_SIZE_PER_CLASS]]

print(f"   ✓ IA: {len(ai_images)} | Real: {len(real_images)}")


# ============================================================
# EVALUAR
# ============================================================

def evaluate_images(images: list, is_ai: bool):
    """Evalúa una lista de imágenes."""
    correct = 0
    total = 0
    
    label = "IA" if is_ai else "Real"
    
    with torch.no_grad():
        for img in tqdm(images, desc=f"   {label}"):
            try:
                # Preprocesar imagen
                img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
                
                # Extraer features CLIP
                features = clip_model.encode_image(img_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.float()
                
                # Clasificar
                output = classifier(features).squeeze()
                prob = torch.sigmoid(output).item()
                
                # Predicción: >0.5 = IA, <0.5 = Real
                predicted_ai = prob > 0.5
                
                # Verificar
                if is_ai and predicted_ai:
                    correct += 1
                elif not is_ai and not predicted_ai:
                    correct += 1
                
                total += 1
                
            except Exception as e:
                continue
    
    return correct, total


print("\n🔍 Evaluando imágenes...")

# Evaluar reales
real_correct, real_total = evaluate_images(real_images, is_ai=False)
real_acc = real_correct / real_total * 100 if real_total > 0 else 0

# Evaluar IA
ai_correct, ai_total = evaluate_images(ai_images, is_ai=True)
ai_acc = ai_correct / ai_total * 100 if ai_total > 0 else 0

# Resultados
print("\n" + "="*70)
print("📊 RESULTADOS DE VERIFICACIÓN")
print("="*70)

print(f"\n📹 Imágenes REALES ({real_total} muestras):")
print(f"   ✅ Correctos: {real_correct}")
print(f"   ❌ Errores: {real_total - real_correct}")
print(f"   Accuracy: {real_acc:.1f}%")

print(f"\n🤖 Imágenes IA ({ai_total} muestras):")
print(f"   ✅ Correctos: {ai_correct}")
print(f"   ❌ Errores: {ai_total - ai_correct}")
print(f"   Accuracy: {ai_acc:.1f}%")

total_correct = real_correct + ai_correct
total_samples = real_total + ai_total
global_acc = total_correct / total_samples * 100 if total_samples > 0 else 0

print(f"\n🎯 ACCURACY GLOBAL: {global_acc:.1f}%")
print(f"   ({total_correct}/{total_samples} correctos)")

if global_acc >= 99.0:
    print(f"\n🏆 ¡VERIFICACIÓN EXITOSA!")
else:
    print(f"\n⚠️ Accuracy por debajo del objetivo")
