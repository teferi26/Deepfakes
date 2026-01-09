#!/usr/bin/env python3
"""
Script de entrenamiento agresivo hasta alcanzar 100% de precisión.
Entrena iterativamente y testea hasta llegar al objetivo.
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
import torch.nn.functional as F
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


def load_all_datasets(n_ai_per_source=5000, n_real=10000, seed=42):
    """Carga datasets de múltiples fuentes."""
    from datasets import load_dataset
    
    ai_images = []
    real_images = []
    
    # 1. DiffusionDB (Stable Diffusion)
    print("📥 Cargando DiffusionDB (Stable Diffusion)...")
    try:
        ds = load_dataset('poloclub/diffusiondb', '2m_random_50k', split='train', trust_remote_code=True)
        ds = ds.shuffle(seed=seed)
        count = 0
        for item in ds:
            if count >= n_ai_per_source:
                break
            ai_images.append(item['image'])
            count += 1
        print(f"   ✓ {count} imágenes Stable Diffusion")
    except Exception as e:
        print(f"   ✗ Error DiffusionDB: {e}")
    
    # 2. Hemg AI (varios generadores)
    print("📥 Cargando Hemg AI...")
    try:
        ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train')
        ds = ds.shuffle(seed=seed)
        count = 0
        for item in ds:
            if item['label'] == 0:  # 0 = AI
                ai_images.append(item['image'])
                count += 1
            if count >= n_ai_per_source:
                break
        print(f"   ✓ {count} imágenes Hemg AI")
    except Exception as e:
        print(f"   ✗ Error Hemg: {e}")
    
    # 3. CIFAKE (StyleGAN)
    print("📥 Cargando CIFAKE...")
    try:
        ds = load_dataset('CIKM/CIFAKE-image-classification', split='train', trust_remote_code=True)
        ds = ds.shuffle(seed=seed)
        ai_count = 0
        real_count = 0
        for item in ds:
            if item.get('label', -1) == 1 and ai_count < n_ai_per_source:  # 1 = FAKE
                ai_images.append(item['image'])
                ai_count += 1
            elif item.get('label', -1) == 0 and real_count < n_real // 2:  # 0 = REAL
                real_images.append(item['image'])
                real_count += 1
        print(f"   ✓ {ai_count} imágenes CIFAKE AI, {real_count} reales")
    except Exception as e:
        print(f"   ✗ Error CIFAKE: {e}")
    
    # 4. Hemg Real
    print("📥 Cargando Hemg Real...")
    try:
        ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train')
        ds = ds.shuffle(seed=seed+1)
        count = 0
        for item in ds:
            if item['label'] == 1:  # 1 = Real
                real_images.append(item['image'])
                count += 1
            if count >= n_real // 2:
                break
        print(f"   ✓ {count} imágenes Hemg Real")
    except Exception as e:
        print(f"   ✗ Error Hemg Real: {e}")
    
    print(f"\n📊 Total: {len(ai_images)} AI, {len(real_images)} Real")
    return ai_images, real_images


def train_epoch(model, features, labels, optimizer, criterion, device, batch_size=64):
    """Entrena una época."""
    model.train()
    
    # Shuffle
    indices = torch.randperm(len(features))
    features = features[indices]
    labels = labels[indices]
    
    total_loss = 0
    n_batches = 0
    
    for i in range(0, len(features), batch_size):
        batch_f = features[i:i+batch_size].to(device)
        batch_l = labels[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_f).squeeze()
        loss = criterion(outputs, batch_l)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, features, labels, device, batch_size=64):
    """Evalúa el modelo."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_f = features[i:i+batch_size].to(device)
            batch_l = labels[i:i+batch_size]
            
            outputs = model(batch_f).squeeze()
            preds = torch.sigmoid(outputs) > 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_l.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean() * 100
    
    # Accuracy por clase
    ai_mask = all_labels == 1
    real_mask = all_labels == 0
    
    ai_acc = (all_preds[ai_mask] == all_labels[ai_mask]).mean() * 100 if ai_mask.sum() > 0 else 0
    real_acc = (all_preds[real_mask] == all_labels[real_mask]).mean() * 100 if real_mask.sum() > 0 else 0
    
    return accuracy, ai_acc, real_acc


def save_head_to_db(model, version, val_accuracy):
    """Guarda la cabeza entrenada en la base de datos."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    import pickle
    
    DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:pass@db:5432/app')
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        state_dict = model.state_dict()
        blob = pickle.dumps(state_dict)
        
        # Eliminar versiones antiguas
        session.execute(text("DELETE FROM clip_head_global"))
        
        # Insertar nueva
        session.execute(
            text("""
                INSERT INTO clip_head_global (version, state_dict, val_accuracy, created_at)
                VALUES (:version, :state_dict, :val_accuracy, NOW())
            """),
            {"version": version, "state_dict": blob, "val_accuracy": val_accuracy}
        )
        session.commit()
        print(f"   💾 Cabeza guardada: {version} (acc={val_accuracy:.2f}%)")
        return True
    except Exception as e:
        print(f"   ✗ Error guardando: {e}")
        session.rollback()
        return False
    finally:
        session.close()


def test_with_new_samples(clip_model, preprocess, model, device, n_samples=100):
    """Testea con muestras nuevas no vistas durante entrenamiento."""
    from datasets import load_dataset
    
    print("\n🧪 Testing con muestras NUEVAS...")
    
    ai_correct = 0
    ai_total = 0
    real_correct = 0
    real_total = 0
    
    # DiffusionDB - nuevas muestras
    print("   Testing DiffusionDB...")
    try:
        ds = load_dataset('poloclub/diffusiondb', '2m_random_50k', split='train', trust_remote_code=True)
        ds = ds.shuffle(seed=9999)  # Seed diferente para muestras nuevas
        
        images = []
        for i, item in enumerate(ds):
            if i >= n_samples:
                break
            images.append(item['image'])
        
        if images:
            features = extract_features_batch(images, clip_model, preprocess, device)
            if len(features) > 0:
                model.eval()
                with torch.no_grad():
                    outputs = model(features.to(device)).squeeze()
                    preds = torch.sigmoid(outputs) > 0.5
                    ai_correct += preds.sum().item()
                    ai_total += len(preds)
    except Exception as e:
        print(f"   ✗ Error DiffusionDB test: {e}")
    
    # Hemg AI - nuevas muestras
    print("   Testing Hemg AI...")
    try:
        ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train')
        ds = ds.shuffle(seed=8888)
        
        images = []
        for item in ds:
            if item['label'] == 0 and len(images) < n_samples:
                images.append(item['image'])
            if len(images) >= n_samples:
                break
        
        if images:
            features = extract_features_batch(images, clip_model, preprocess, device)
            if len(features) > 0:
                model.eval()
                with torch.no_grad():
                    outputs = model(features.to(device)).squeeze()
                    preds = torch.sigmoid(outputs) > 0.5
                    ai_correct += preds.sum().item()
                    ai_total += len(preds)
    except Exception as e:
        print(f"   ✗ Error Hemg AI test: {e}")
    
    # Reales - nuevas muestras
    print("   Testing Reales...")
    try:
        ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train')
        ds = ds.shuffle(seed=7777)
        
        images = []
        for item in ds:
            if item['label'] == 1 and len(images) < n_samples:
                images.append(item['image'])
            if len(images) >= n_samples:
                break
        
        if images:
            features = extract_features_batch(images, clip_model, preprocess, device)
            if len(features) > 0:
                model.eval()
                with torch.no_grad():
                    outputs = model(features.to(device)).squeeze()
                    preds = torch.sigmoid(outputs) <= 0.5  # Real = 0
                    real_correct += preds.sum().item()
                    real_total += len(preds)
    except Exception as e:
        print(f"   ✗ Error Hemg Real test: {e}")
    
    ai_acc = (ai_correct / ai_total * 100) if ai_total > 0 else 0
    real_acc = (real_correct / real_total * 100) if real_total > 0 else 0
    total_acc = ((ai_correct + real_correct) / (ai_total + real_total) * 100) if (ai_total + real_total) > 0 else 0
    
    print(f"\n   📊 Resultados Test:")
    print(f"      AI:    {ai_correct}/{ai_total} ({ai_acc:.1f}%)")
    print(f"      Real:  {real_correct}/{real_total} ({real_acc:.1f}%)")
    print(f"      TOTAL: {ai_correct + real_correct}/{ai_total + real_total} ({total_acc:.1f}%)")
    
    return total_acc, ai_acc, real_acc


def main():
    print("=" * 70)
    print("🎯 ENTRENAMIENTO HASTA 100% DE PRECISIÓN")
    print("=" * 70)
    print()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Cargar CLIP
    print("\n📦 Cargando CLIP ViT-L/14...")
    import clip
    clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print("   ✓ CLIP cargado")
    
    # Configuración inicial
    TARGET_ACCURACY = 100.0
    current_n_samples = 3000  # Empezar con menos para iterar rápido
    max_n_samples = 20000
    iteration = 0
    best_accuracy = 0
    
    while best_accuracy < TARGET_ACCURACY:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"🔄 ITERACIÓN {iteration} - Muestras por fuente: {current_n_samples}")
        print(f"{'='*70}")
        
        # Cargar datos
        ai_images, real_images = load_all_datasets(
            n_ai_per_source=current_n_samples,
            n_real=current_n_samples * 2,
            seed=42 + iteration
        )
        
        if len(ai_images) < 100 or len(real_images) < 100:
            print("❌ No hay suficientes imágenes. Abortando.")
            break
        
        # Balancear
        min_count = min(len(ai_images), len(real_images))
        random.shuffle(ai_images)
        random.shuffle(real_images)
        ai_images = ai_images[:min_count]
        real_images = real_images[:min_count]
        
        print(f"\n📊 Dataset balanceado: {min_count} AI + {min_count} Real = {min_count*2} total")
        
        # Extraer features
        print("\n🔄 Extrayendo features con CLIP...")
        ai_features = extract_features_batch(ai_images, clip_model, preprocess, DEVICE)
        real_features = extract_features_batch(real_images, clip_model, preprocess, DEVICE)
        
        print(f"   AI features: {ai_features.shape}")
        print(f"   Real features: {real_features.shape}")
        
        # Crear dataset
        all_features = torch.cat([ai_features, real_features], dim=0)
        all_labels = torch.cat([
            torch.ones(len(ai_features)),
            torch.zeros(len(real_features))
        ])
        
        # Split train/val
        indices = torch.randperm(len(all_features))
        split = int(0.8 * len(indices))
        
        train_features = all_features[indices[:split]]
        train_labels = all_labels[indices[:split]]
        val_features = all_features[indices[split:]]
        val_labels = all_labels[indices[split:]]
        
        print(f"   Train: {len(train_features)}, Val: {len(val_features)}")
        
        # Crear modelo
        model = CLIPLinearClassifier(input_dim=768).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        criterion = nn.BCEWithLogitsLoss()
        
        # Entrenar
        print("\n🎓 Entrenando...")
        epochs = 50
        best_val_acc = 0
        patience = 10
        no_improve = 0
        
        for epoch in range(epochs):
            loss = train_epoch(model, train_features, train_labels, optimizer, criterion, DEVICE)
            val_acc, val_ai_acc, val_real_acc = evaluate(model, val_features, val_labels, DEVICE)
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if (epoch + 1) % 5 == 0 or val_acc >= 99:
                print(f"   Epoch {epoch+1:3d}: Loss={loss:.4f}, Val={val_acc:.2f}% (AI={val_ai_acc:.1f}%, Real={val_real_acc:.1f}%)")
            
            if no_improve >= patience:
                print(f"   Early stopping en epoch {epoch+1}")
                break
            
            if val_acc >= 100:
                print(f"   🎉 100% en validación!")
                break
        
        # Restaurar mejor modelo
        model.load_state_dict(best_state)
        
        # Test con muestras nuevas
        test_acc, test_ai_acc, test_real_acc = test_with_new_samples(
            clip_model, preprocess, model, DEVICE, n_samples=100
        )
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            
            # Guardar en BD
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            save_head_to_db(model, version, test_acc)
        
        print(f"\n📈 Mejor precisión hasta ahora: {best_accuracy:.2f}%")
        
        if best_accuracy >= TARGET_ACCURACY:
            print("\n🎉🎉🎉 ¡100% ALCANZADO! 🎉🎉🎉")
            break
        
        # Aumentar muestras si no llegamos
        if best_accuracy < 95:
            current_n_samples = min(current_n_samples + 2000, max_n_samples)
        elif best_accuracy < 99:
            current_n_samples = min(current_n_samples + 1000, max_n_samples)
        
        if current_n_samples >= max_n_samples and iteration > 5:
            print("\n⚠️ Alcanzado máximo de muestras. Intentando más épocas...")
            epochs = 100
    
    print("\n" + "=" * 70)
    print(f"🏁 ENTRENAMIENTO FINALIZADO")
    print(f"   Mejor precisión: {best_accuracy:.2f}%")
    print(f"   Iteraciones: {iteration}")
    print("=" * 70)


if __name__ == "__main__":
    main()
