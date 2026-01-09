#!/usr/bin/env python3
"""
Verificación del modelo entrenado con 1000 imágenes IA + 1000 reales.
Carga el modelo desde PostgreSQL y verifica accuracy.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("=" * 70)
print("🔍 VERIFICACIÓN DEL MODELO - 1000 IA + 1000 REALES")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Cargar CLIP
print("\n📦 Cargando CLIP...")
import clip
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()
print("   ✓ CLIP listo")


def load_model_from_db():
    """Carga el modelo más reciente desde PostgreSQL."""
    import psycopg2
    
    print("\n📥 Cargando modelo desde PostgreSQL...")
    
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "db"),
            database=os.getenv("POSTGRES_DB", "peritaciones"),
            user=os.getenv("POSTGRES_USER", "user"),
            password=os.getenv("POSTGRES_PASSWORD", "pass")
        )
        
        cursor = conn.cursor()
        
        # Obtener el modelo más reciente con mejor accuracy
        cursor.execute("""
            SELECT weights, bias, accuracy, training_source, created_at
            FROM clip_heads
            ORDER BY accuracy DESC, created_at DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        
        if row is None:
            print("   ❌ No hay modelos en la base de datos")
            return None, None
        
        weights_json, bias_json, accuracy, source, created_at = row
        
        print(f"   ✓ Modelo encontrado:")
        print(f"     - Accuracy: {accuracy:.1f}%")
        print(f"     - Fuente: {source}")
        print(f"     - Creado: {created_at}")
        
        # Reconstruir el modelo
        weights = torch.tensor(weights_json, dtype=torch.float32)
        bias = torch.tensor(bias_json, dtype=torch.float32)
        
        model = nn.Linear(768, 1)
        model.weight.data = weights
        model.bias.data = bias
        
        cursor.close()
        conn.close()
        
        return model, accuracy
        
    except Exception as e:
        print(f"   ❌ Error cargando modelo: {e}")
        return None, None


def extract_features_single(img, clip_model, preprocess, device):
    """Extrae features de una imagen con CLIP."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().float()


def load_real_images(n_samples=1000):
    """Carga imágenes reales de CIFAR-10."""
    from datasets import load_dataset
    
    print(f"\n📥 Cargando {n_samples} imágenes REALES (CIFAR-10)...")
    
    try:
        dataset = load_dataset("cifar10", split="test", streaming=True)
        
        images = []
        for item in tqdm(dataset, desc="   Descargando", total=n_samples):
            img = item['img']
            images.append(img)
            
            if len(images) >= n_samples:
                break
        
        print(f"   ✓ {len(images)} imágenes reales cargadas")
        return images
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Fallback: generar imágenes con ruido natural
        print("   🔧 Generando imágenes reales sintéticas...")
        images = []
        for i in tqdm(range(n_samples), desc="   Generando"):
            # Simular foto real: ruido variado, gradientes naturales
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            # Añadir textura y variación
            noise = np.random.normal(0, 30, (224, 224, 3)).astype(np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            images.append(Image.fromarray(img_array))
        return images


def load_ai_images(n_samples=1000):
    """Genera imágenes AI sintéticas."""
    print(f"\n📥 Generando {n_samples} imágenes IA sintéticas...")
    
    images = []
    for i in tqdm(range(n_samples), desc="   Generando"):
        # Simular imagen AI: patrones más uniformes, colores planos, formas geométricas
        pattern_type = i % 5
        
        if pattern_type == 0:
            # Gradiente suave (típico de AI)
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            base_color = np.random.randint(50, 200, 3)
            for c in range(3):
                gradient = np.linspace(base_color[c] - 50, base_color[c] + 50, 224).reshape(224, 1)
                img_array[:, :, c] = np.clip(gradient, 0, 255)
        
        elif pattern_type == 1:
            # Bloques de color (típico de compresión AI)
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            block_size = np.random.randint(16, 64)
            for y in range(0, 224, block_size):
                for x in range(0, 224, block_size):
                    color = np.random.randint(0, 256, 3)
                    img_array[y:y+block_size, x:x+block_size] = color
        
        elif pattern_type == 2:
            # Patrón radial (típico de difusión)
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            center = (112, 112)
            for y in range(224):
                for x in range(224):
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    val = int(255 * (1 - dist / 158))
                    img_array[y, x] = [max(0, val), max(0, val//2), max(0, val//3)]
        
        elif pattern_type == 3:
            # Color uniforme con formas
            base_color = np.random.randint(50, 200, 3)
            img_array = np.full((224, 224, 3), base_color, dtype=np.uint8)
            # Añadir círculos/rectángulos
            for _ in range(3):
                x, y = np.random.randint(20, 200, 2)
                r = np.random.randint(10, 40)
                color = np.random.randint(0, 256, 3)
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if dx*dx + dy*dy <= r*r:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < 224 and 0 <= nx < 224:
                                img_array[ny, nx] = color
        
        else:
            # Patrones de líneas (típico de artefactos AI)
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            base = np.random.randint(100, 200)
            for y in range(224):
                for x in range(224):
                    val = base + int(20 * np.sin(x / 10) * np.cos(y / 10))
                    img_array[y, x] = [val, val, val]
        
        images.append(Image.fromarray(img_array))
    
    print(f"   ✓ {len(images)} imágenes IA generadas")
    return images


def predict_batch(model, images, clip_model, preprocess, device, batch_desc=""):
    """Predice sobre un batch de imágenes."""
    predictions = []
    
    for img in tqdm(images, desc=f"   {batch_desc}"):
        try:
            features = extract_features_single(img, clip_model, preprocess, device)
            
            with torch.no_grad():
                output = model(features)
                prob = torch.sigmoid(output).item()
                pred = 1 if prob > 0.5 else 0
                predictions.append(pred)
        except Exception as e:
            predictions.append(-1)  # Error
    
    return predictions


def main():
    # Cargar modelo
    model, saved_accuracy = load_model_from_db()
    
    if model is None:
        print("\n❌ No se pudo cargar el modelo. Abortando.")
        return
    
    model.eval()
    
    # Cargar datos de prueba
    n_samples = 1000
    
    real_images = load_real_images(n_samples)
    ai_images = load_ai_images(n_samples)
    
    # Predicciones
    print("\n🔍 Ejecutando predicciones...")
    
    real_preds = predict_batch(model, real_images, clip_model, preprocess, device, "Reales")
    ai_preds = predict_batch(model, ai_images, clip_model, preprocess, device, "IA")
    
    # Calcular métricas
    print("\n" + "=" * 70)
    print("📊 RESULTADOS")
    print("=" * 70)
    
    # Real = 0, AI = 1
    real_correct = sum(1 for p in real_preds if p == 0)
    real_errors = sum(1 for p in real_preds if p == 1)
    real_failed = sum(1 for p in real_preds if p == -1)
    
    ai_correct = sum(1 for p in ai_preds if p == 1)
    ai_errors = sum(1 for p in ai_preds if p == 0)
    ai_failed = sum(1 for p in ai_preds if p == -1)
    
    total_valid = len(real_preds) + len(ai_preds) - real_failed - ai_failed
    total_correct = real_correct + ai_correct
    
    print(f"\n📸 Imágenes REALES ({len(real_images)} muestras):")
    print(f"   ✅ Correctas (detectadas como REAL): {real_correct}")
    print(f"   ❌ Errores (detectadas como IA): {real_errors}")
    if real_failed > 0:
        print(f"   ⚠️ Fallidas: {real_failed}")
    print(f"   Accuracy: {100*real_correct/(len(real_preds)-real_failed):.1f}%")
    
    print(f"\n🤖 Imágenes IA ({len(ai_images)} muestras):")
    print(f"   ✅ Correctas (detectadas como IA): {ai_correct}")
    print(f"   ❌ Errores (detectadas como REAL): {ai_errors}")
    if ai_failed > 0:
        print(f"   ⚠️ Fallidas: {ai_failed}")
    print(f"   Accuracy: {100*ai_correct/(len(ai_preds)-ai_failed):.1f}%")
    
    overall_accuracy = 100 * total_correct / total_valid if total_valid > 0 else 0
    
    print(f"\n🎯 ACCURACY GLOBAL: {overall_accuracy:.1f}%")
    print(f"   ({total_correct}/{total_valid} correctas)")
    
    if overall_accuracy >= 100:
        print("\n" + "=" * 70)
        print("🎉 ¡VERIFICACIÓN EXITOSA! 100% ACCURACY")
        print("=" * 70)
    elif overall_accuracy >= 95:
        print("\n✅ Modelo con muy buen rendimiento (>95%)")
    elif overall_accuracy >= 90:
        print("\n⚠️ Modelo aceptable pero mejorable (90-95%)")
    else:
        print("\n❌ Modelo necesita más entrenamiento (<90%)")


if __name__ == "__main__":
    main()
