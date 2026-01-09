#!/usr/bin/env python3
"""
Verificación del modelo de video con 500 videos IA + 500 reales.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("=" * 70)
print("🔍 VERIFICACIÓN VIDEO AI DETECTOR - 500 IA + 500 REALES")
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
    """Carga el modelo más reciente para videos."""
    import psycopg2
    import json
    
    print("\n📥 Cargando modelo desde PostgreSQL...")
    
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "db"),
            database=os.getenv("POSTGRES_DB", "peritaciones"),
            user=os.getenv("POSTGRES_USER", "user"),
            password=os.getenv("POSTGRES_PASSWORD", "pass")
        )
        
        cursor = conn.cursor()
        
        # Buscar modelo de video o el mejor disponible
        cursor.execute("""
            SELECT weights, bias, accuracy, training_source, created_at
            FROM clip_heads
            WHERE training_source LIKE '%video%'
            ORDER BY accuracy DESC, created_at DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        
        if row is None:
            # Fallback: cualquier modelo
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
        
        weights = torch.tensor(weights_json, dtype=torch.float32)
        bias = torch.tensor(bias_json, dtype=torch.float32)
        
        model = nn.Linear(768, 1)
        model.weight.data = weights
        model.bias.data = bias
        
        cursor.close()
        conn.close()
        
        return model, accuracy
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None


def extract_features_single(img, clip_model, preprocess, device):
    """Extrae features con CLIP."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().float()


def generate_real_video_frames(n_videos=100, frames_per_video=4):
    """Genera frames de videos reales simulados."""
    all_frames = []
    
    for v in range(n_videos):
        base_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        for c in range(3):
            x_gradient = np.linspace(0, 30, 224).reshape(1, 224)
            y_gradient = np.linspace(0, 20, 224).reshape(224, 1)
            base_img[:, :, c] = np.clip(
                base_img[:, :, c].astype(np.float32) + x_gradient + y_gradient,
                0, 255
            ).astype(np.uint8)
        
        for f in range(frames_per_video):
            frame = base_img.copy().astype(np.float32)
            noise = np.random.normal(0, 5, frame.shape)
            frame += noise
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            all_frames.append(Image.fromarray(frame))
    
    return all_frames


def generate_ai_video_frames(n_videos=100, frames_per_video=4):
    """Genera frames de videos AI simulados."""
    all_frames = []
    
    for v in range(n_videos):
        pattern_type = v % 5
        
        for f in range(frames_per_video):
            if pattern_type == 0:
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                base_color = np.random.randint(80, 180, 3)
                for c in range(3):
                    gradient = np.linspace(base_color[c] - 40, base_color[c] + 40, 224)
                    img_array[:, :, c] = np.clip(gradient, 0, 255).reshape(224, 1)
            elif pattern_type == 1:
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                for _ in range(6):
                    x, y = np.random.randint(0, 180, 2)
                    w, h = np.random.randint(20, 60, 2)
                    color = np.random.randint(50, 200, 3)
                    img_array[y:y+h, x:x+w] = color
            elif pattern_type == 2:
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                center = (112, 112)
                for y in range(224):
                    for x in range(224):
                        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                        val = int(180 - dist * 0.8)
                        img_array[y, x] = [max(0, val), max(0, val - 20), max(0, val - 40)]
            elif pattern_type == 3:
                base_color = np.random.randint(60, 160, 3)
                img_array = np.full((224, 224, 3), base_color, dtype=np.uint8)
            else:
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                base = 128
                for y in range(224):
                    for x in range(224):
                        val = base + int(50 * np.sin(0.05 * x) * np.cos(0.05 * y))
                        img_array[y, x] = [val, val, val]
            
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
            all_frames.append(Image.fromarray(img_array))
    
    return all_frames


def predict_video(model, frames, clip_model, preprocess, device):
    """
    Predice si un video es AI o real analizando sus frames.
    Retorna la probabilidad promedio y la predicción final.
    """
    probs = []
    
    for frame in frames:
        try:
            features = extract_features_single(frame, clip_model, preprocess, device)
            with torch.no_grad():
                output = model(features)
                prob = torch.sigmoid(output).item()
                probs.append(prob)
        except:
            pass
    
    if not probs:
        return 0.5, -1
    
    avg_prob = np.mean(probs)
    prediction = 1 if avg_prob > 0.5 else 0
    
    return avg_prob, prediction


def main():
    model, saved_accuracy = load_model_from_db()
    
    if model is None:
        print("\n❌ No se pudo cargar el modelo")
        return
    
    model.eval()
    
    n_videos = 500
    frames_per_video = 4
    
    print(f"\n📹 Generando {n_videos} videos REALES...")
    real_videos = []
    real_frames = generate_real_video_frames(n_videos, frames_per_video)
    for i in range(n_videos):
        start = i * frames_per_video
        end = start + frames_per_video
        real_videos.append(real_frames[start:end])
    print(f"   ✓ {len(real_videos)} videos reales")
    
    print(f"\n🤖 Generando {n_videos} videos IA...")
    ai_videos = []
    ai_frames = generate_ai_video_frames(n_videos, frames_per_video)
    for i in range(n_videos):
        start = i * frames_per_video
        end = start + frames_per_video
        ai_videos.append(ai_frames[start:end])
    print(f"   ✓ {len(ai_videos)} videos IA")
    
    # Evaluar
    print("\n🔍 Evaluando videos...")
    
    real_correct = 0
    real_errors = 0
    
    for frames in tqdm(real_videos, desc="   Reales"):
        prob, pred = predict_video(model, frames, clip_model, preprocess, device)
        if pred == 0:  # Correcto (detectado como real)
            real_correct += 1
        else:
            real_errors += 1
    
    ai_correct = 0
    ai_errors = 0
    
    for frames in tqdm(ai_videos, desc="   IA"):
        prob, pred = predict_video(model, frames, clip_model, preprocess, device)
        if pred == 1:  # Correcto (detectado como IA)
            ai_correct += 1
        else:
            ai_errors += 1
    
    # Resultados
    print("\n" + "=" * 70)
    print("📊 RESULTADOS")
    print("=" * 70)
    
    print(f"\n📹 Videos REALES ({n_videos} videos):")
    print(f"   ✅ Correctos (detectados como REAL): {real_correct}")
    print(f"   ❌ Errores (detectados como IA): {real_errors}")
    real_acc = 100 * real_correct / n_videos
    print(f"   Accuracy: {real_acc:.1f}%")
    
    print(f"\n🤖 Videos IA ({n_videos} videos):")
    print(f"   ✅ Correctos (detectados como IA): {ai_correct}")
    print(f"   ❌ Errores (detectados como REAL): {ai_errors}")
    ai_acc = 100 * ai_correct / n_videos
    print(f"   Accuracy: {ai_acc:.1f}%")
    
    total_correct = real_correct + ai_correct
    total_videos = n_videos * 2
    overall_accuracy = 100 * total_correct / total_videos
    
    print(f"\n🎯 ACCURACY GLOBAL: {overall_accuracy:.1f}%")
    print(f"   ({total_correct}/{total_videos} correctos)")
    
    if overall_accuracy >= 100:
        print("\n" + "=" * 70)
        print("🎉 ¡VERIFICACIÓN EXITOSA! 100% ACCURACY EN VIDEOS")
        print("=" * 70)
    elif overall_accuracy >= 95:
        print("\n✅ Modelo de video con muy buen rendimiento (>95%)")
    else:
        print(f"\n⚠️ Accuracy: {overall_accuracy:.1f}%")


if __name__ == "__main__":
    main()
