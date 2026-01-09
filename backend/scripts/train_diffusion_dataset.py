#!/usr/bin/env python3
"""
Script para entrenar el detector CLIP con imágenes de Stable Diffusion (DiffusionDB)
y imágenes reales del dataset Hemg.

Target: 25K AI (Stable Diffusion) + 25K Real = 50K total
"""

import os
import sys
import random
import numpy as np
from datetime import datetime
from io import BytesIO

# Añadir el path del backend
sys.path.insert(0, '/app')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

def main():
    print("=" * 60)
    print("ENTRENAMIENTO CLIP CON DIFFUSIONDB + IMAGENES REALES")
    print("=" * 60)
    
    # Configuración
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Parámetros
    N_AI_SAMPLES = 25000
    N_REAL_SAMPLES = 25000
    BATCH_SIZE = 64
    EPOCHS = 25
    LR = 0.05
    
    print(f"\nObjetivo: {N_AI_SAMPLES} AI + {N_REAL_SAMPLES} Real = {N_AI_SAMPLES + N_REAL_SAMPLES} total")
    
    # =====================================================
    # PASO 1: Cargar modelo CLIP
    # =====================================================
    print("\n[1/5] Cargando modelo CLIP...")
    import open_clip
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained='openai',
        device=DEVICE
    )
    model.eval()
    print("CLIP ViT-L-14 cargado")
    
    # =====================================================
    # PASO 2: Cargar DiffusionDB (imágenes de Stable Diffusion)
    # =====================================================
    print("\n[2/5] Cargando DiffusionDB (Stable Diffusion)...")
    from datasets import load_dataset
    
    # Usamos el subset "2m_random_50k" que tiene 50K imágenes
    # O podemos cargar múltiples subsets de 1K
    try:
        # Intentar cargar 50K para tener margen
        ds_ai = load_dataset('poloclub/diffusiondb', '2m_random_50k', split='train', trust_remote_code=True)
        print(f"DiffusionDB cargado: {len(ds_ai)} imágenes AI")
    except Exception as e:
        print(f"Error cargando 50k subset: {e}")
        print("Intentando con subsets más pequeños...")
        # Cargar múltiples subsets de 1K
        ds_parts = []
        for i in range(30):  # 30 x 1K = 30K
            try:
                ds = load_dataset('poloclub/diffusiondb', f'2m_random_1k', split='train', trust_remote_code=True)
                ds_parts.append(ds)
                print(f"  Subset {i+1}/30 cargado")
            except:
                break
        from datasets import concatenate_datasets
        ds_ai = concatenate_datasets(ds_parts)
        print(f"DiffusionDB cargado: {len(ds_ai)} imágenes AI")
    
    # =====================================================
    # PASO 3: Cargar imágenes reales del dataset Hemg
    # =====================================================
    print("\n[3/5] Cargando imágenes reales (Hemg dataset)...")
    
    ds_hemg = load_dataset(
        "Hemg/AI-Generated-vs-Real-Images-Datasets",
        split="train",
        trust_remote_code=True
    )
    
    # Filtrar solo imágenes reales (label=1 en Hemg)
    real_indices = [i for i in range(len(ds_hemg)) if ds_hemg[i]['label'] == 1]
    print(f"Imágenes reales en Hemg: {len(real_indices)}")
    
    # =====================================================
    # PASO 4: Preparar índices balanceados
    # =====================================================
    print("\n[4/5] Preparando datos balanceados...")
    
    # Seleccionar índices aleatorios
    n_ai = min(N_AI_SAMPLES, len(ds_ai))
    n_real = min(N_REAL_SAMPLES, len(real_indices))
    
    ai_indices = random.sample(range(len(ds_ai)), n_ai)
    real_sample_indices = random.sample(real_indices, n_real)
    
    print(f"Muestras AI (DiffusionDB): {n_ai}")
    print(f"Muestras Reales (Hemg): {n_real}")
    print(f"Total: {n_ai + n_real}")
    
    # Split train/val (80/20)
    random.shuffle(ai_indices)
    random.shuffle(real_sample_indices)
    
    train_split = 0.8
    n_ai_train = int(n_ai * train_split)
    n_real_train = int(n_real * train_split)
    
    train_ai = ai_indices[:n_ai_train]
    val_ai = ai_indices[n_ai_train:]
    train_real = real_sample_indices[:n_real_train]
    val_real = real_sample_indices[n_real_train:]
    
    print(f"Train: {len(train_ai)} AI + {len(train_real)} Real = {len(train_ai) + len(train_real)}")
    print(f"Val: {len(val_ai)} AI + {len(val_real)} Real = {len(val_ai) + len(val_real)}")
    
    # =====================================================
    # PASO 5: Extraer features con CLIP
    # =====================================================
    print("\n[5/5] Extrayendo features con CLIP...")
    
    def extract_features_batch(images, model, preprocess, device):
        """Extrae features de un batch de imágenes PIL"""
        tensors = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            tensors.append(preprocess(img))
        
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()
    
    all_features = []
    all_labels = []
    
    # Procesar imágenes AI (DiffusionDB) - Train
    print("\nExtrayendo features AI (train)...")
    batch_images = []
    for i, idx in enumerate(tqdm(train_ai, desc="AI train")):
        try:
            img = ds_ai[idx]['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            batch_images.append(img)
            
            if len(batch_images) >= BATCH_SIZE:
                feats = extract_features_batch(batch_images, model, preprocess, DEVICE)
                all_features.extend(feats)
                all_labels.extend([1] * len(batch_images))  # 1 = AI
                batch_images = []
        except Exception as e:
            continue
    
    if batch_images:
        feats = extract_features_batch(batch_images, model, preprocess, DEVICE)
        all_features.extend(feats)
        all_labels.extend([1] * len(batch_images))
    
    n_train_ai = len(all_labels)
    print(f"Features AI train extraídas: {n_train_ai}")
    
    # Procesar imágenes Reales (Hemg) - Train
    print("\nExtrayendo features Reales (train)...")
    batch_images = []
    for i, idx in enumerate(tqdm(train_real, desc="Real train")):
        try:
            img = ds_hemg[idx]['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            batch_images.append(img)
            
            if len(batch_images) >= BATCH_SIZE:
                feats = extract_features_batch(batch_images, model, preprocess, DEVICE)
                all_features.extend(feats)
                all_labels.extend([0] * len(batch_images))  # 0 = Real
                batch_images = []
        except Exception as e:
            continue
    
    if batch_images:
        feats = extract_features_batch(batch_images, model, preprocess, DEVICE)
        all_features.extend(feats)
        all_labels.extend([0] * len(batch_images))
    
    n_train_real = len(all_labels) - n_train_ai
    print(f"Features Real train extraídas: {n_train_real}")
    
    # Convertir a arrays
    train_features = np.array(all_features)
    train_labels = np.array(all_labels)
    print(f"Train total: {len(train_features)} samples")
    
    # Procesar validación
    print("\nExtrayendo features de validación...")
    val_features = []
    val_labels = []
    
    # Val AI
    batch_images = []
    for idx in tqdm(val_ai, desc="AI val"):
        try:
            img = ds_ai[idx]['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            batch_images.append(img)
            
            if len(batch_images) >= BATCH_SIZE:
                feats = extract_features_batch(batch_images, model, preprocess, DEVICE)
                val_features.extend(feats)
                val_labels.extend([1] * len(batch_images))
                batch_images = []
        except:
            continue
    
    if batch_images:
        feats = extract_features_batch(batch_images, model, preprocess, DEVICE)
        val_features.extend(feats)
        val_labels.extend([1] * len(batch_images))
    
    # Val Real
    batch_images = []
    for idx in tqdm(val_real, desc="Real val"):
        try:
            img = ds_hemg[idx]['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            batch_images.append(img)
            
            if len(batch_images) >= BATCH_SIZE:
                feats = extract_features_batch(batch_images, model, preprocess, DEVICE)
                val_features.extend(feats)
                val_labels.extend([0] * len(batch_images))
                batch_images = []
        except:
            continue
    
    if batch_images:
        feats = extract_features_batch(batch_images, model, preprocess, DEVICE)
        val_features.extend(feats)
        val_labels.extend([0] * len(batch_images))
    
    val_features = np.array(val_features)
    val_labels = np.array(val_labels)
    print(f"Val total: {len(val_features)} samples")
    
    # Liberar memoria
    del ds_ai, ds_hemg
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    
    # =====================================================
    # ENTRENAR LINEAR HEAD
    # =====================================================
    print("\n" + "=" * 60)
    print("ENTRENANDO LINEAR HEAD")
    print("=" * 60)
    
    feature_dim = train_features.shape[1]
    print(f"Feature dimension: {feature_dim}")
    
    # Crear tensores
    X_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    X_val = torch.tensor(val_features, dtype=torch.float32)
    y_val = torch.tensor(val_labels, dtype=torch.float32)
    
    # Modelo simple
    head = nn.Linear(feature_dim, 1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(head.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(EPOCHS):
        # Training
        head.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = head(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
        
        scheduler.step()
        train_acc = correct / total
        
        # Validation
        head.eval()
        with torch.no_grad():
            val_out = head(X_val.to(DEVICE)).squeeze()
            val_preds = (torch.sigmoid(val_out) > 0.5).float()
            val_acc = (val_preds == y_val.to(DEVICE)).float().mean().item()
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = head.state_dict().copy()
    
    print(f"\nMejor Val Accuracy: {best_val_acc:.4f}")
    
    # Cargar mejor modelo
    head.load_state_dict(best_state)
    
    # =====================================================
    # GUARDAR EN BASE DE DATOS
    # =====================================================
    print("\n" + "=" * 60)
    print("GUARDANDO MODELO EN BASE DE DATOS")
    print("=" * 60)
    
    from app.core.database import SessionLocal
    from app.models.ml_models import MLModel
    
    # Serializar pesos
    weight = head.weight.data.cpu().numpy()
    bias = head.bias.data.cpu().numpy()
    
    model_data = {
        "weight": weight.tolist(),
        "bias": bias.tolist(),
        "feature_dim": feature_dim,
        "training_info": {
            "dataset": "DiffusionDB + Hemg",
            "n_ai_samples": int(n_train_ai),
            "n_real_samples": int(n_train_real),
            "epochs": EPOCHS,
            "best_val_accuracy": float(best_val_acc),
            "lr": LR
        }
    }
    
    import json
    model_bytes = json.dumps(model_data).encode('utf-8')
    
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    
    db = SessionLocal()
    try:
        # Desactivar modelos anteriores
        db.query(MLModel).filter(
            MLModel.model_type == "clip_head_global",
            MLModel.is_active == True
        ).update({"is_active": False})
        
        # Crear nuevo modelo
        new_model = MLModel(
            model_type="clip_head_global",
            version=version,
            model_data=model_bytes,
            is_active=True,
            metadata_={
                "training_samples": n_train_ai + n_train_real,
                "val_accuracy": float(best_val_acc),
                "dataset": "DiffusionDB_StableDiffusion + Hemg_Real"
            }
        )
        db.add(new_model)
        db.commit()
        print(f"Modelo guardado: clip_head_global v{version}")
        print(f"Val Accuracy: {best_val_acc:.4f}")
    finally:
        db.close()
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO!")
    print("=" * 60)
    print(f"Reinicia el worker para cargar el nuevo modelo:")
    print(f"  docker-compose restart worker")

if __name__ == "__main__":
    main()
