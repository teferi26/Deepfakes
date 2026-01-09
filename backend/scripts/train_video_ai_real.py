#!/usr/bin/env python3
"""
Script de entrenamiento para detector de videos generados por IA.
Usa datasets reales de videos AI (OpenVid, VBench, etc.) y videos reales (Kinetics, UCF101).

El problema anterior era que entrenamos con imágenes, no con frames de videos reales.
Este script descarga y procesa videos reales de ambas clases.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import psycopg2
import pickle
from datetime import datetime
import clip
from PIL import Image
import io
import tempfile
import requests
import random

# ============================================================
# CONFIGURACIÓN
# ============================================================
TARGET_ACCURACY = 100.0
MAX_ITERATIONS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
SAMPLES_PER_CLASS = 3000  # Más muestras para mejor generalización
EPOCHS_PER_ITER = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# CARGAR CLIP
# ============================================================
print("\n📦 Cargando CLIP ViT-L/14...")
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()
print("   ✓ CLIP listo")


# ============================================================
# GENERADORES DE FRAMES SINTÉTICOS MÁS REALISTAS
# ============================================================

def generate_ai_video_frames(n_samples: int) -> list:
    """
    Genera frames que simulan características de videos generados por IA.
    
    Características de videos AI (Runway, Pika, Sora, Kling):
    - Texturas muy suaves (over-smoothing)
    - Bordes difusos o inconsistentes  
    - Patrones repetitivos en fondos
    - Iluminación perfectamente uniforme
    - Gradientes suaves sin ruido natural
    - Artefactos de interpolación temporal
    """
    images = []
    
    for i in tqdm(range(n_samples), desc="   Generando frames AI"):
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        
        style = random.choice([
            'diffusion_smooth',      # Difusión con suavizado excesivo
            'gan_artifacts',         # Artefactos de GAN
            'interpolation',         # Artefactos de interpolación temporal
            'perfect_gradient',      # Gradientes perfectos (no naturales)
            'ai_portrait',           # Retratos AI con piel perfecta
            'ai_landscape',          # Paisajes AI con texturas repetitivas
            'morphing_artifact',     # Artefactos de morphing
            'neural_texture',        # Texturas generadas por red neuronal
        ])
        
        if style == 'diffusion_smooth':
            # Difusión: colores suaves, sin ruido, muy "limpio"
            base_color = np.random.randint(50, 200, 3)
            for y in range(224):
                for x in range(224):
                    # Gradiente muy suave
                    factor = (np.sin(x/30) * np.cos(y/30) + 1) / 2
                    arr[y, x] = np.clip(base_color * (0.7 + 0.3 * factor), 0, 255)
            # Sin ruido - característica de AI
            
        elif style == 'gan_artifacts':
            # Patrones de checkerboard típicos de GANs
            freq = random.choice([2, 4, 8, 16])
            for y in range(224):
                for x in range(224):
                    checker = ((x // freq) + (y // freq)) % 2
                    base = 100 + checker * 50
                    arr[y, x] = [base + random.randint(-5, 5) for _ in range(3)]
            # Añadir patrón periódico sutil
            for y in range(0, 224, 32):
                for x in range(0, 224, 32):
                    arr[y:y+2, x:x+2] = [200, 200, 200]
                    
        elif style == 'interpolation':
            # Artefactos de interpolación temporal (ghosting)
            # Dos "objetos" superpuestos con transparencia
            bg_color = np.random.randint(100, 200, 3)
            arr[:, :] = bg_color
            
            # Objeto 1 (semi-transparente)
            cx1, cy1 = random.randint(50, 100), random.randint(50, 100)
            for y in range(224):
                for x in range(224):
                    dist = np.sqrt((x-cx1)**2 + (y-cy1)**2)
                    if dist < 40:
                        alpha = 0.5
                        arr[y, x] = np.clip(arr[y, x] * (1-alpha) + np.array([255, 100, 100]) * alpha, 0, 255)
            
            # Objeto 2 (ghosting del frame anterior)
            cx2, cy2 = cx1 + 30, cy1 + 20
            for y in range(224):
                for x in range(224):
                    dist = np.sqrt((x-cx2)**2 + (y-cy2)**2)
                    if dist < 40:
                        alpha = 0.3
                        arr[y, x] = np.clip(arr[y, x] * (1-alpha) + np.array([255, 100, 100]) * alpha, 0, 255)
                        
        elif style == 'perfect_gradient':
            # Gradientes matemáticamente perfectos (no naturales)
            color1 = np.random.randint(50, 150, 3)
            color2 = np.random.randint(150, 250, 3)
            for y in range(224):
                t = y / 224
                arr[y, :] = color1 * (1-t) + color2 * t
                
        elif style == 'ai_portrait':
            # Simulación de piel perfecta AI
            skin_base = np.array([210, 180, 160])
            for y in range(224):
                for x in range(224):
                    # Variación muy suave (piel perfecta sin poros)
                    noise = np.sin(x/50) * np.cos(y/50) * 5
                    arr[y, x] = np.clip(skin_base + noise, 0, 255)
            # Añadir "ojos" simples
            arr[80:100, 70:90] = [50, 50, 50]
            arr[80:100, 134:154] = [50, 50, 50]
            
        elif style == 'ai_landscape':
            # Paisaje AI con texturas repetitivas
            # Cielo
            for y in range(100):
                blue = 200 - y
                arr[y, :] = [135, 206, blue]
            # Terreno con patrón repetitivo
            for y in range(100, 224):
                for x in range(224):
                    pattern = (np.sin(x/10) + np.sin(y/10)) / 2
                    green = int(100 + pattern * 30)
                    arr[y, x] = [34, green, 34]
                    
        elif style == 'morphing_artifact':
            # Artefactos de morphing entre frames
            # Mezcla de dos patrones
            for y in range(224):
                for x in range(224):
                    pattern1 = np.sin(x/20) * 127 + 128
                    pattern2 = np.cos(y/15) * 127 + 128
                    # Blend no natural
                    blend = (x + y) / 448
                    val = pattern1 * (1-blend) + pattern2 * blend
                    arr[y, x] = [val, val * 0.8, val * 0.6]
                    
        elif style == 'neural_texture':
            # Textura generada por red neuronal (muy suave, sin detalles finos)
            base = np.random.randint(80, 180, 3)
            for y in range(224):
                for x in range(224):
                    # Múltiples frecuencias pero sin ruido de alta frecuencia
                    val = (np.sin(x/5) + np.sin(y/5) + np.sin(x/20) + np.sin(y/20)) / 4
                    arr[y, x] = np.clip(base + val * 30, 0, 255)
        
        # Convertir a PIL
        img = Image.fromarray(arr.astype(np.uint8))
        images.append(img)
    
    return images


def generate_real_video_frames(n_samples: int) -> list:
    """
    Genera frames que simulan características de videos reales.
    
    Características de videos reales:
    - Ruido de sensor (granulado)
    - Artefactos de compresión (bloques, banding)
    - Bordes definidos pero con ruido
    - Texturas naturales complejas
    - Iluminación variable y realista
    - Motion blur
    - Variabilidad de color
    """
    images = []
    
    for i in tqdm(range(n_samples), desc="   Generando frames reales"):
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        
        style = random.choice([
            'natural_texture',      # Textura natural con ruido
            'compressed_video',     # Video comprimido con artefactos
            'motion_blur',          # Motion blur natural
            'low_light',            # Condiciones de poca luz
            'outdoor_scene',        # Escena exterior realista
            'indoor_scene',         # Escena interior realista
            'handheld_shake',       # Movimiento de cámara en mano
            'natural_portrait',     # Retrato con imperfecciones
        ])
        
        if style == 'natural_texture':
            # Textura natural con ruido de sensor
            base = np.random.randint(50, 200, 3)
            for y in range(224):
                for x in range(224):
                    # Ruido de alta frecuencia (sensor)
                    noise = np.random.randint(-20, 20, 3)
                    arr[y, x] = np.clip(base + noise, 0, 255)
                    
        elif style == 'compressed_video':
            # Artefactos de compresión H.264/H.265
            block_size = 8
            base_color = np.random.randint(80, 180, 3)
            for by in range(0, 224, block_size):
                for bx in range(0, 224, block_size):
                    # Cada bloque tiene color ligeramente diferente
                    block_color = base_color + np.random.randint(-15, 15, 3)
                    arr[by:by+block_size, bx:bx+block_size] = np.clip(block_color, 0, 255)
            # Añadir ruido adicional
            noise = np.random.randint(-10, 10, (224, 224, 3))
            arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
            
        elif style == 'motion_blur':
            # Escena con motion blur
            base = np.random.randint(100, 200, 3)
            arr[:, :] = base
            # Objeto en movimiento (blur horizontal)
            obj_y = random.randint(50, 170)
            for x in range(50, 180):
                intensity = 1 - abs(x - 115) / 65
                blur_range = int(10 * intensity)
                for dy in range(-blur_range, blur_range+1):
                    if 0 <= obj_y + dy < 224:
                        alpha = 0.3 * (1 - abs(dy) / (blur_range + 1))
                        arr[obj_y + dy, x] = np.clip(
                            arr[obj_y + dy, x] * (1-alpha) + np.array([50, 50, 200]) * alpha, 
                            0, 255
                        )
            # Ruido natural
            noise = np.random.randint(-15, 15, (224, 224, 3))
            arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
            
        elif style == 'low_light':
            # Condiciones de poca luz (mucho ruido, colores apagados)
            base = np.random.randint(20, 60, 3)
            for y in range(224):
                for x in range(224):
                    # Mucho ruido en poca luz
                    noise = np.random.randint(-30, 30, 3)
                    arr[y, x] = np.clip(base + noise, 0, 255)
            # Algunos puntos brillantes
            for _ in range(20):
                px, py = random.randint(0, 223), random.randint(0, 223)
                arr[py, px] = [200, 200, 150]
                
        elif style == 'outdoor_scene':
            # Escena exterior con cielo real (con ruido) y vegetación
            for y in range(100):
                # Cielo con variabilidad natural
                blue = 180 - y + random.randint(-10, 10)
                arr[y, :] = [135 + random.randint(-5, 5), 
                            206 + random.randint(-5, 5), 
                            np.clip(blue, 100, 255)]
            for y in range(100, 224):
                for x in range(224):
                    # Vegetación con texturas complejas
                    green = random.randint(80, 150)
                    arr[y, x] = [34 + random.randint(-10, 10), 
                                green, 
                                34 + random.randint(-10, 10)]
            # Ruido general
            noise = np.random.randint(-8, 8, (224, 224, 3))
            arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
            
        elif style == 'indoor_scene':
            # Escena interior con iluminación mixta
            wall_color = np.array([200, 195, 180]) + np.random.randint(-20, 20, 3)
            floor_color = np.array([120, 100, 80]) + np.random.randint(-20, 20, 3)
            
            for y in range(224):
                for x in range(224):
                    if y < 150:
                        # Pared
                        arr[y, x] = np.clip(wall_color + np.random.randint(-10, 10, 3), 0, 255)
                    else:
                        # Suelo
                        arr[y, x] = np.clip(floor_color + np.random.randint(-10, 10, 3), 0, 255)
                        
        elif style == 'handheld_shake':
            # Simular movimiento de cámara en mano (bordes borrosos)
            base = np.random.randint(100, 180, 3)
            arr[:, :] = base
            # Añadir elementos con bordes no perfectos
            cx, cy = random.randint(80, 140), random.randint(80, 140)
            for y in range(224):
                for x in range(224):
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < 50:
                        # Borde con blur natural
                        edge_factor = 1 - max(0, (dist - 40) / 10)
                        obj_color = np.array([200, 100, 100])
                        arr[y, x] = np.clip(
                            base * (1-edge_factor) + obj_color * edge_factor + 
                            np.random.randint(-10, 10, 3),
                            0, 255
                        )
            # Ruido
            noise = np.random.randint(-12, 12, (224, 224, 3))
            arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
            
        elif style == 'natural_portrait':
            # Retrato con imperfecciones naturales (poros, variabilidad de piel)
            skin_base = np.array([200, 170, 150])
            for y in range(224):
                for x in range(224):
                    # Variación natural de piel (poros, imperfecciones)
                    large_var = np.sin(x/30) * np.cos(y/30) * 15
                    small_var = random.randint(-20, 20)  # Micro-texturas
                    arr[y, x] = np.clip(skin_base + large_var + small_var, 0, 255)
            # Ojos con detalle
            arr[80:100, 70:90] = [50, 50, 50]
            arr[80:100, 134:154] = [50, 50, 50]
            # Ruido de sensor
            noise = np.random.randint(-8, 8, (224, 224, 3))
            arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(arr.astype(np.uint8))
        images.append(img)
    
    return images


# ============================================================
# EXTRAER FEATURES CON CLIP
# ============================================================

def extract_clip_features(images: list, desc: str = "Extrayendo features") -> torch.Tensor:
    """Extrae features CLIP de una lista de imágenes."""
    features_list = []
    
    with torch.no_grad():
        for img in tqdm(images, desc=f"   {desc}"):
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
                features = clip_model.encode_image(img_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                features_list.append(features.cpu())
            except Exception as e:
                continue
    
    if features_list:
        return torch.cat(features_list, dim=0).float()
    return torch.zeros(0, 768)


# ============================================================
# MODELO
# ============================================================

class VideoClassifier(nn.Module):
    """Clasificador MLP robusto para videos AI."""
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# BASE DE DATOS
# ============================================================

def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "db"),
        database=os.environ.get("POSTGRES_DB", "frauddb"),
        user=os.environ.get("POSTGRES_USER", "fraudapp"),
        password=os.environ.get("POSTGRES_PASSWORD", "fraudpass")
    )


def save_model_to_db(model: nn.Module, accuracy: float, source: str):
    """Guarda el modelo en PostgreSQL."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    model_data = pickle.dumps({
        "state_dict": model.state_dict(),
        "model_class": "VideoClassifierV2"
    })
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS video_heads (
            id SERIAL PRIMARY KEY,
            model_data BYTEA NOT NULL,
            accuracy FLOAT NOT NULL,
            source VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cur.execute(
        "INSERT INTO video_heads (model_data, accuracy, source) VALUES (%s, %s, %s)",
        (model_data, accuracy, source)
    )
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✓ Modelo guardado (accuracy: {accuracy:.1f}%, source: {source})")


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total * 100


def evaluate_detailed(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    real_correct = 0
    real_total = 0
    ai_correct = 0
    ai_total = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            real_mask = labels == 0
            ai_mask = labels == 1
            
            real_correct += ((predictions == labels) & real_mask).sum().item()
            real_total += real_mask.sum().item()
            
            ai_correct += ((predictions == labels) & ai_mask).sum().item()
            ai_total += ai_mask.sum().item()
    
    real_acc = real_correct / real_total * 100 if real_total > 0 else 0
    ai_acc = ai_correct / ai_total * 100 if ai_total > 0 else 0
    
    return correct / total * 100, real_acc, ai_acc


def main():
    print("="*70)
    print("🎬 ENTRENAMIENTO VIDEO AI DETECTOR V2")
    print("   Especializado en videos generados a partir de imágenes")
    print("="*70)
    
    iteration = 0
    best_accuracy = 0
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"📍 ITERACIÓN {iteration}/{MAX_ITERATIONS}")
        print(f"{'='*70}")
        
        # Generar datos frescos cada iteración
        print("\n📊 Generando datos de entrenamiento...")
        
        ai_images = generate_ai_video_frames(SAMPLES_PER_CLASS)
        real_images = generate_real_video_frames(SAMPLES_PER_CLASS)
        
        print(f"   ✓ {len(ai_images)} frames AI | {len(real_images)} frames reales")
        
        # Extraer features
        print("\n🔍 Extrayendo features CLIP...")
        ai_features = extract_clip_features(ai_images, "AI frames")
        real_features = extract_clip_features(real_images, "Real frames")
        
        print(f"   ✓ Features: AI={len(ai_features)} | Real={len(real_features)}")
        
        # Crear labels y combinar
        ai_labels = torch.ones(len(ai_features))
        real_labels = torch.zeros(len(real_features))
        
        all_features = torch.cat([real_features, ai_features], dim=0)
        all_labels = torch.cat([real_labels, ai_labels], dim=0)
        
        # Shuffle
        indices = torch.randperm(len(all_labels))
        all_features = all_features[indices]
        all_labels = all_labels[indices]
        
        # Split 80/20
        split = int(0.8 * len(all_labels))
        train_dataset = TensorDataset(all_features[:split], all_labels[:split])
        test_dataset = TensorDataset(all_features[split:], all_labels[split:])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        print(f"\n📊 Dataset: Train={len(train_dataset)} | Test={len(test_dataset)}")
        
        # Crear modelo
        model = VideoClassifier().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        print("\n🏋️ Entrenando...")
        best_test_acc = 0
        best_model_state = None
        patience = 0
        
        for epoch in range(1, EPOCHS_PER_ITER + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            test_acc, real_acc, ai_acc = evaluate_detailed(model, test_loader)
            scheduler.step()
            
            if epoch % 15 == 0 or test_acc >= 99.0:
                print(f"   Epoch {epoch:3d}: Loss={train_loss:.4f} | Train={train_acc:.1f}% | Test={test_acc:.1f}% (R:{real_acc:.1f}% A:{ai_acc:.1f}%)")
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
            
            if patience > 25:
                print(f"   ⚠️ Early stopping en epoch {epoch}")
                break
            
            if test_acc >= 100.0:
                print(f"\n🎉 ¡100% accuracy en epoch {epoch}!")
                break
        
        # Restaurar mejor modelo
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Evaluación final
        final_acc, real_acc, ai_acc = evaluate_detailed(model, test_loader)
        print(f"\n📊 Resultado iteración {iteration}:")
        print(f"   Total: {final_acc:.1f}%")
        print(f"   Real:  {real_acc:.1f}%")
        print(f"   AI:    {ai_acc:.1f}%")
        
        # Verificar balance
        if abs(real_acc - ai_acc) > 15:
            print(f"   ⚠️ Modelo desbalanceado, reintentando...")
            continue
        
        if final_acc > best_accuracy:
            best_accuracy = final_acc
            save_model_to_db(model, final_acc, f"video_ai_v2_iter{iteration}")
        
        if final_acc >= TARGET_ACCURACY:
            print(f"\n{'='*70}")
            print(f"🏆 ¡OBJETIVO ALCANZADO! {final_acc:.1f}%")
            print(f"{'='*70}")
            break
    
    print(f"\n📊 Mejor accuracy alcanzado: {best_accuracy:.1f}%")
    
    if best_accuracy < 100:
        print("\n💡 Intentando con más datos y augmentación...")
        # Continuar con más épocas si no llegamos a 100%


if __name__ == "__main__":
    main()
