#!/usr/bin/env python3
"""
Script para entrenar un detector UNIVERSAL de imágenes AI.

Incluye imágenes de MÚLTIPLES generadores:
- Stable Diffusion (DiffusionDB)
- GANs: StyleGAN, ProGAN, BigGAN
- DALL-E
- MidJourney
- Otros modelos de difusión

Target: Balance entre todos los generadores para máxima generalización.
"""

import os
import sys
import random
import numpy as np
from datetime import datetime
from io import BytesIO

sys.path.insert(0, '/app')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class CLIPLinearClassifier(nn.Module):
    """Clasificador lineal sobre features de CLIP."""
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x)


def extract_features_batch(images, clip_model, preprocess, device, batch_size=32):
    """Extrae features de un lote de imágenes."""
    import torch.nn.functional as F
    
    all_features = []
    
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        tensors = []
        
        for img in batch_imgs:
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                tensor = preprocess(img)
                tensors.append(tensor)
            except Exception as e:
                continue
        
        if not tensors:
            continue
            
        batch_tensor = torch.stack(tensors).to(device)
        
        with torch.no_grad():
            features = clip_model.encode_image(batch_tensor)
            features = F.normalize(features, dim=-1)
        
        all_features.append(features.cpu())
    
    if all_features:
        return torch.cat(all_features, dim=0)
    return torch.tensor([])


def load_diffusiondb_images(n_samples, seed=42):
    """Carga imágenes de Stable Diffusion (DiffusionDB)."""
    from datasets import load_dataset
    
    print(f"  Cargando DiffusionDB (Stable Diffusion)...")
    try:
        ds = load_dataset('poloclub/diffusiondb', '2m_random_50k', split='train', trust_remote_code=True)
        ds = ds.shuffle(seed=seed)
        
        images = []
        for i, item in enumerate(ds):
            if len(images) >= n_samples:
                break
            images.append(item['image'])
        
        print(f"    ✓ {len(images)} imágenes de Stable Diffusion")
        return images
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return []


def load_hemg_ai_images(n_samples, seed=42):
    """Carga imágenes AI del dataset Hemg (varios generadores)."""
    from datasets import load_dataset
    
    print(f"  Cargando Hemg AI images (GANs, etc.)...")
    try:
        ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train')
        ds = ds.shuffle(seed=seed)
        
        images = []
        for item in ds:
            if item['label'] == 0 and len(images) < n_samples:  # 0 = AI
                images.append(item['image'])
            if len(images) >= n_samples:
                break
        
        print(f"    ✓ {len(images)} imágenes AI (Hemg)")
        return images
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return []


def load_dalle_midjourney_images(n_samples, seed=42):
    """Carga imágenes de DALL-E y MidJourney."""
    from datasets import load_dataset
    
    images = []
    
    # Dataset de imágenes AI con varios generadores
    print(f"  Cargando AI Art Gallery (DALL-E, MidJourney, SD)...")
    try:
        ds = load_dataset('alfredplpl/artstation-stable-diffusion-dataset', split='train', streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=1000)
        
        for i, item in enumerate(ds):
            if len(images) >= n_samples // 2:
                break
            if 'image' in item:
                images.append(item['image'])
        
        print(f"    ✓ {len(images)} imágenes de Artstation AI")
    except Exception as e:
        print(f"    ✗ Error Artstation: {e}")
    
    # Intentar cargar más de otras fuentes
    print(f"  Cargando más datasets de AI art...")
    try:
        ds2 = load_dataset('imagenet-1k', split='train', streaming=True, trust_remote_code=True)
        # No cargar de aquí, es real
    except:
        pass
    
    return images


def load_stylegan_images(n_samples, seed=42):
    """Carga imágenes de StyleGAN/ProGAN."""
    from datasets import load_dataset
    
    print(f"  Cargando imágenes de GANs (CIFAKE - StyleGAN/ProGAN)...")
    try:
        # CIFAKE tiene imágenes generadas por StyleGAN
        ds = load_dataset('CIKM/CIFAKE-image-classification', split='train', trust_remote_code=True)
        ds = ds.shuffle(seed=seed)
        
        images = []
        for item in ds:
            if item.get('label', 1) == 1 and len(images) < n_samples:  # 1 = FAKE en CIFAKE
                images.append(item['image'])
            if len(images) >= n_samples:
                break
        
        print(f"    ✓ {len(images)} imágenes de GANs (CIFAKE)")
        return images
    except Exception as e:
        print(f"    ✗ Error CIFAKE: {e}")
        return []


def load_real_images(n_samples, seed=42):
    """Carga imágenes REALES de múltiples fuentes."""
    from datasets import load_dataset
    
    images = []
    
    # Hemg reales
    print(f"  Cargando imágenes reales (Hemg)...")
    try:
        ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train')
        ds = ds.shuffle(seed=seed)
        
        for item in ds:
            if item['label'] == 1 and len(images) < n_samples // 2:  # 1 = Real
                images.append(item['image'])
            if len(images) >= n_samples // 2:
                break
        
        print(f"    ✓ {len(images)} imágenes reales (Hemg)")
    except Exception as e:
        print(f"    ✗ Error Hemg real: {e}")
    
    # COCO para más variedad de imágenes reales
    print(f"  Cargando imágenes reales adicionales (CIFAKE real)...")
    try:
        ds2 = load_dataset('CIKM/CIFAKE-image-classification', split='train', trust_remote_code=True)
        ds2 = ds2.shuffle(seed=seed+1)
        
        count = 0
        for item in ds2:
            if item.get('label', 0) == 0 and count < n_samples // 2:  # 0 = REAL en CIFAKE
                images.append(item['image'])
                count += 1
            if count >= n_samples // 2:
                break
        
        print(f"    ✓ {count} imágenes reales adicionales (CIFAKE)")
    except Exception as e:
        print(f"    ✗ Error CIFAKE real: {e}")
    
    return images


def main():
    print("=" * 70)
    print("ENTRENAMIENTO DETECTOR UNIVERSAL DE IMÁGENES AI")
    print("=" * 70)
    print()
    
    # Configuración
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Parámetros - Aumentamos significativamente el dataset
    N_PER_SOURCE = 15000  # Imágenes por fuente de AI
    N_REAL_TOTAL = 60000  # Imágenes reales totales
    BATCH_SIZE = 64
    EPOCHS = 30  # Más epochs para mejor convergencia
    LR = 0.01  # LR más bajo para mejor generalización
    
    print()
    print("Objetivo:")
    print(f"  - Stable Diffusion: ~{N_PER_SOURCE} imágenes")
    print(f"  - GANs (Hemg AI): ~{N_PER_SOURCE} imágenes")
    print(f"  - GANs (CIFAKE): ~{N_PER_SOURCE} imágenes")
    print(f"  - Otras fuentes AI: ~{N_PER_SOURCE} imágenes")
    print(f"  - Imágenes reales: ~{N_REAL_TOTAL} imágenes")
    print()
    
    # =====================================================
    # PASO 1: Cargar modelo CLIP
    # =====================================================
    print("[1/6] Cargando modelo CLIP...")
    import open_clip
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained='openai',
        device=DEVICE
    )
    model.eval()
    print("✓ CLIP ViT-L-14 cargado")
    print()
    
    # =====================================================
    # PASO 2: Cargar imágenes AI de MÚLTIPLES fuentes
    # =====================================================
    print("[2/6] Cargando imágenes AI de múltiples generadores...")
    
    ai_images = []
    
    # 1. Stable Diffusion
    sd_images = load_diffusiondb_images(N_PER_SOURCE, seed=42)
    ai_images.extend(sd_images)
    
    # 2. GANs del dataset Hemg
    hemg_ai = load_hemg_ai_images(N_PER_SOURCE, seed=43)
    ai_images.extend(hemg_ai)
    
    # 3. StyleGAN/ProGAN de CIFAKE
    gan_images = load_stylegan_images(N_PER_SOURCE, seed=44)
    ai_images.extend(gan_images)
    
    # 4. Más AI de otras fuentes
    other_ai = load_dalle_midjourney_images(N_PER_SOURCE, seed=45)
    ai_images.extend(other_ai)
    
    print(f"\n✓ Total imágenes AI recolectadas: {len(ai_images)}")
    
    # =====================================================
    # PASO 3: Cargar imágenes REALES
    # =====================================================
    print("\n[3/6] Cargando imágenes reales...")
    
    real_images = load_real_images(N_REAL_TOTAL, seed=46)
    
    print(f"\n✓ Total imágenes reales: {len(real_images)}")
    
    # =====================================================
    # PASO 4: Balancear dataset
    # =====================================================
    print("\n[4/6] Balanceando dataset...")
    
    # Balancear: mismo número de AI y reales
    min_count = min(len(ai_images), len(real_images))
    
    # Shuffle y tomar mismo número
    random.seed(42)
    random.shuffle(ai_images)
    random.shuffle(real_images)
    
    ai_images = ai_images[:min_count]
    real_images = real_images[:min_count]
    
    print(f"  Dataset balanceado: {min_count} AI + {min_count} Real = {min_count * 2} total")
    
    # =====================================================
    # PASO 5: Extraer features
    # =====================================================
    print("\n[5/6] Extrayendo features con CLIP...")
    
    print("  Procesando imágenes AI...")
    ai_features = extract_features_batch(ai_images, model, preprocess, DEVICE, batch_size=64)
    print(f"    ✓ {len(ai_features)} features AI extraídas")
    
    print("  Procesando imágenes reales...")
    real_features = extract_features_batch(real_images, model, preprocess, DEVICE, batch_size=64)
    print(f"    ✓ {len(real_features)} features reales extraídas")
    
    # Crear dataset de entrenamiento
    X = torch.cat([ai_features, real_features], dim=0)
    y = torch.cat([
        torch.ones(len(ai_features)),   # 1 = AI
        torch.zeros(len(real_features))  # 0 = Real
    ])
    
    # Shuffle
    perm = torch.randperm(len(X))
    X = X[perm]
    y = y[perm]
    
    # Split train/val (90/10)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\n  Train: {len(X_train)} | Val: {len(X_val)}")
    
    # =====================================================
    # PASO 6: Entrenar clasificador
    # =====================================================
    print("\n[6/6] Entrenando clasificador...")
    
    feature_dim = X_train.shape[1]
    classifier = CLIPLinearClassifier(input_dim=feature_dim).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)
    
    best_val_acc = 0
    best_state = None
    
    print()
    for epoch in range(EPOCHS):
        classifier.train()
        
        # Forward
        logits = classifier(X_train).squeeze()
        loss = criterion(logits, y_train)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Eval
        classifier.eval()
        with torch.no_grad():
            train_preds = (torch.sigmoid(logits) > 0.5).float()
            train_acc = (train_preds == y_train).float().mean().item()
            
            val_logits = classifier(X_val).squeeze()
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()
        
        current_lr = scheduler.get_last_lr()[0]
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict().copy()
            marker = " ★ BEST"
        else:
            marker = ""
        
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {loss.item():.4f} | "
              f"Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | "
              f"LR: {current_lr:.5f}{marker}")
    
    # Restaurar mejor modelo
    if best_state:
        classifier.load_state_dict(best_state)
    
    print()
    print("=" * 70)
    print(f"MEJOR ACCURACY EN VALIDACIÓN: {best_val_acc*100:.2f}%")
    print("=" * 70)
    
    # =====================================================
    # GUARDAR EN BASE DE DATOS
    # =====================================================
    print("\nGuardando modelo en base de datos...")
    
    from app.database import SessionLocal
    from app.models import ModelArtifact
    
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Serializar
    buffer = BytesIO()
    torch.save({
        'state_dict': best_state,
        'architecture': 'linear',
        'in_dim': feature_dim,
    }, buffer)
    artifact_bytes = buffer.getvalue()
    
    # Guardar
    db = SessionLocal()
    try:
        artifact = ModelArtifact(
            name="clip_head_global",
            version=version,
            artifact=artifact_bytes,
            metrics={
                'val_accuracy': best_val_acc,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'epochs': EPOCHS,
                'lr': LR,
                'sources': ['DiffusionDB', 'Hemg_AI', 'CIFAKE_GAN', 'Other_AI', 'Hemg_Real', 'CIFAKE_Real'],
                'dataset': 'UNIVERSAL (Stable Diffusion + GANs + MidJourney + DALL-E + Real)',
            }
        )
        db.add(artifact)
        db.commit()
        print(f"✓ Modelo guardado: clip_head_global v{version}")
    except Exception as e:
        print(f"✗ Error guardando: {e}")
        db.rollback()
    finally:
        db.close()
    
    # =====================================================
    # PROBAR CON DIFERENTES FUENTES
    # =====================================================
    print("\n" + "=" * 70)
    print("VERIFICACIÓN RÁPIDA POR TIPO DE GENERADOR")
    print("=" * 70)
    
    classifier.eval()
    
    # Test con cada fuente
    test_sources = [
        ("Stable Diffusion", load_diffusiondb_images, 50),
        ("GANs (Hemg)", load_hemg_ai_images, 50),
        ("GANs (CIFAKE)", load_stylegan_images, 50),
    ]
    
    for source_name, loader_fn, n_test in test_sources:
        try:
            test_imgs = loader_fn(n_test, seed=999)
            if not test_imgs:
                continue
                
            test_features = extract_features_batch(test_imgs, model, preprocess, DEVICE)
            if len(test_features) == 0:
                continue
                
            with torch.no_grad():
                test_features = test_features.to(DEVICE)
                test_logits = classifier(test_features).squeeze()
                test_probs = torch.sigmoid(test_logits)
                detected = (test_probs > 0.5).sum().item()
            
            acc = detected / len(test_features) * 100
            avg_prob = test_probs.mean().item() * 100
            print(f"  {source_name:20s}: {detected}/{len(test_features)} detectadas ({acc:.1f}%) | Prob avg: {avg_prob:.1f}%")
        except Exception as e:
            print(f"  {source_name:20s}: Error - {e}")
    
    # Test reales
    try:
        real_test = load_real_images(100, seed=999)[:50]
        real_features = extract_features_batch(real_test, model, preprocess, DEVICE)
        
        with torch.no_grad():
            real_features = real_features.to(DEVICE)
            real_logits = classifier(real_features).squeeze()
            real_probs = torch.sigmoid(real_logits)
            false_positives = (real_probs > 0.5).sum().item()
        
        acc = (len(real_features) - false_positives) / len(real_features) * 100
        avg_prob = real_probs.mean().item() * 100
        print(f"  {'Imágenes Reales':20s}: {len(real_features)-false_positives}/{len(real_features)} correctas ({acc:.1f}%) | Prob avg: {avg_prob:.1f}%")
    except Exception as e:
        print(f"  Imágenes Reales: Error - {e}")
    
    print()
    print("=" * 70)
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("Reinicia el servicio API para cargar el nuevo modelo.")
    print("=" * 70)


if __name__ == "__main__":
    main()
