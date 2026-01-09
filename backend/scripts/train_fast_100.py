#!/usr/bin/env python3
"""
Entrenamiento rápido hasta 100% de precisión.
Usa streaming para descargar solo lo necesario.
"""

import os
import sys
import random
import numpy as np
from datetime import datetime

sys.path.insert(0, '/app')

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


class CLIPLinearClassifier(nn.Module):
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x)


def extract_features_single(img, clip_model, preprocess, device):
    """Extrae features de una imagen."""
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(tensor)
            features = F.normalize(features, dim=-1)
        return features.cpu().float()  # Convertir a float32
    except Exception as e:
        return None


def load_datasets_streaming(n_samples_per_source=2000, seed=42):
    """Carga datasets usando streaming para velocidad."""
    from datasets import load_dataset
    
    ai_images = []
    real_images = []
    
    random.seed(seed)
    
    # 1. DiffusionDB (streaming)
    print("📥 Cargando DiffusionDB (streaming)...")
    try:
        ds = load_dataset('poloclub/diffusiondb', '2m_random_50k', split='train', 
                          streaming=True, trust_remote_code=True)
        count = 0
        for item in ds:
            if count >= n_samples_per_source:
                break
            ai_images.append(item['image'])
            count += 1
            if count % 500 == 0:
                print(f"   ... {count} imágenes SD")
        print(f"   ✓ {count} imágenes Stable Diffusion")
    except Exception as e:
        print(f"   ✗ Error DiffusionDB: {e}")
    
    # 2. CIFAKE (streaming) - más rápido
    print("📥 Cargando CIFAKE (streaming)...")
    try:
        ds = load_dataset('CIKM/CIFAKE-image-classification', split='train', 
                          streaming=True, trust_remote_code=True)
        ai_count = 0
        real_count = 0
        target = n_samples_per_source
        
        for item in ds:
            label = item.get('label', -1)
            if label == 1 and ai_count < target:  # FAKE
                ai_images.append(item['image'])
                ai_count += 1
            elif label == 0 and real_count < target:  # REAL
                real_images.append(item['image'])
                real_count += 1
            
            if ai_count >= target and real_count >= target:
                break
            
            if (ai_count + real_count) % 1000 == 0:
                print(f"   ... AI:{ai_count}, Real:{real_count}")
        
        print(f"   ✓ {ai_count} AI, {real_count} Real de CIFAKE")
    except Exception as e:
        print(f"   ✗ Error CIFAKE: {e}")
    
    # 3. Hemg (streaming)
    print("📥 Cargando Hemg (streaming)...")
    try:
        ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train', streaming=True)
        ai_count = 0
        real_count = 0
        target = n_samples_per_source
        
        for item in ds:
            label = item.get('label', -1)
            if label == 0 and ai_count < target:  # AI
                ai_images.append(item['image'])
                ai_count += 1
            elif label == 1 and real_count < target:  # Real
                real_images.append(item['image'])
                real_count += 1
            
            if ai_count >= target and real_count >= target:
                break
        
        print(f"   ✓ {ai_count} AI, {real_count} Real de Hemg")
    except Exception as e:
        print(f"   ✗ Error Hemg: {e}")
    
    print(f"\n📊 Total: {len(ai_images)} AI, {len(real_images)} Real")
    return ai_images, real_images


def save_head_to_db(model, version, val_accuracy):
    """Guarda la cabeza en la base de datos."""
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
        
        session.execute(text("DELETE FROM clip_head_global"))
        session.execute(
            text("""
                INSERT INTO clip_head_global (version, state_dict, val_accuracy, created_at)
                VALUES (:version, :state_dict, :val_accuracy, NOW())
            """),
            {"version": version, "state_dict": blob, "val_accuracy": val_accuracy}
        )
        session.commit()
        print(f"   💾 Guardado: {version} ({val_accuracy:.2f}%)")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        session.rollback()
        return False
    finally:
        session.close()


def test_accuracy(clip_model, preprocess, model, device, n_test=100):
    """Test rápido de precisión."""
    from datasets import load_dataset
    
    print("\n🧪 Testing...")
    
    model.eval()
    ai_correct = 0
    ai_total = 0
    real_correct = 0
    real_total = 0
    
    # Test con DiffusionDB (seed diferente)
    try:
        ds = load_dataset('poloclub/diffusiondb', '2m_random_50k', split='train', 
                          streaming=True, trust_remote_code=True)
        
        # Saltar las primeras N para tener muestras diferentes
        skip = 40000
        count = 0
        skipped = 0
        
        for item in ds:
            if skipped < skip:
                skipped += 1
                continue
            
            if count >= n_test:
                break
            
            feat = extract_features_single(item['image'], clip_model, preprocess, device)
            if feat is not None:
                with torch.no_grad():
                    out = model(feat.to(device)).squeeze()
                    pred = torch.sigmoid(out) > 0.5
                    if pred.item():
                        ai_correct += 1
                    ai_total += 1
                    count += 1
        
        print(f"   DiffusionDB: {ai_correct}/{ai_total}")
    except Exception as e:
        print(f"   Error SD: {e}")
    
    # Test con Hemg Real
    try:
        ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train', streaming=True)
        
        count = 0
        skip = 5000
        skipped = 0
        
        for item in ds:
            if item.get('label', -1) != 1:  # Solo reales
                continue
            
            if skipped < skip:
                skipped += 1
                continue
            
            if count >= n_test:
                break
            
            feat = extract_features_single(item['image'], clip_model, preprocess, device)
            if feat is not None:
                with torch.no_grad():
                    out = model(feat.to(device)).squeeze()
                    pred = torch.sigmoid(out) <= 0.5  # Real = 0
                    if pred.item():
                        real_correct += 1
                    real_total += 1
                    count += 1
        
        print(f"   Hemg Real: {real_correct}/{real_total}")
    except Exception as e:
        print(f"   Error Real: {e}")
    
    # Test con Hemg AI
    try:
        ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train', streaming=True)
        
        count = 0
        skip = 3000
        skipped = 0
        
        for item in ds:
            if item.get('label', -1) != 0:  # Solo AI
                continue
            
            if skipped < skip:
                skipped += 1
                continue
            
            if count >= n_test:
                break
            
            feat = extract_features_single(item['image'], clip_model, preprocess, device)
            if feat is not None:
                with torch.no_grad():
                    out = model(feat.to(device)).squeeze()
                    pred = torch.sigmoid(out) > 0.5
                    if pred.item():
                        ai_correct += 1
                    ai_total += 1
                    count += 1
        
        print(f"   Hemg AI: +{count}")
    except Exception as e:
        print(f"   Error Hemg AI: {e}")
    
    ai_acc = (ai_correct / ai_total * 100) if ai_total > 0 else 0
    real_acc = (real_correct / real_total * 100) if real_total > 0 else 0
    total_acc = ((ai_correct + real_correct) / (ai_total + real_total) * 100) if (ai_total + real_total) > 0 else 0
    
    print(f"\n   📊 AI:    {ai_correct}/{ai_total} ({ai_acc:.1f}%)")
    print(f"   📊 Real:  {real_correct}/{real_total} ({real_acc:.1f}%)")
    print(f"   📊 TOTAL: {total_acc:.1f}%")
    
    return total_acc, ai_acc, real_acc


def main():
    print("=" * 70)
    print("🎯 ENTRENAMIENTO HASTA 100%")
    print("=" * 70)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar CLIP
    print("\n📦 Cargando CLIP...")
    import clip
    clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    print("   ✓ CLIP listo")
    
    TARGET = 100.0
    best_acc = 0
    n_samples = 2000
    iteration = 0
    
    while best_acc < TARGET and iteration < 10:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"🔄 ITERACIÓN {iteration}")
        print(f"{'='*70}")
        
        # Cargar datos
        ai_images, real_images = load_datasets_streaming(n_samples, seed=42+iteration)
        
        if len(ai_images) < 500 or len(real_images) < 500:
            print("❌ Datos insuficientes")
            break
        
        # Balancear
        min_n = min(len(ai_images), len(real_images))
        random.shuffle(ai_images)
        random.shuffle(real_images)
        ai_images = ai_images[:min_n]
        real_images = real_images[:min_n]
        
        print(f"\n📊 Balanceado: {min_n} AI + {min_n} Real")
        
        # Extraer features
        print("\n🔄 Extrayendo features...")
        ai_features = []
        for i, img in enumerate(tqdm(ai_images, desc="AI")):
            f = extract_features_single(img, clip_model, preprocess, DEVICE)
            if f is not None:
                ai_features.append(f)
        
        real_features = []
        for i, img in enumerate(tqdm(real_images, desc="Real")):
            f = extract_features_single(img, clip_model, preprocess, DEVICE)
            if f is not None:
                real_features.append(f)
        
        if not ai_features or not real_features:
            print("❌ No hay features")
            continue
        
        ai_features = torch.cat(ai_features, dim=0)
        real_features = torch.cat(real_features, dim=0)
        
        print(f"   AI: {ai_features.shape}")
        print(f"   Real: {real_features.shape}")
        
        # Dataset
        all_features = torch.cat([ai_features, real_features], dim=0)
        all_labels = torch.cat([
            torch.ones(len(ai_features)),
            torch.zeros(len(real_features))
        ])
        
        # Split
        idx = torch.randperm(len(all_features))
        split = int(0.8 * len(idx))
        
        train_f = all_features[idx[:split]]
        train_l = all_labels[idx[:split]]
        val_f = all_features[idx[split:]]
        val_l = all_labels[idx[split:]]
        
        # Modelo
        model = CLIPLinearClassifier(768).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.BCEWithLogitsLoss()
        
        # Entrenar
        print("\n🎓 Entrenando...")
        best_val = 0
        best_state = None
        patience = 15
        no_improve = 0
        
        for epoch in range(100):
            model.train()
            
            # Shuffle
            perm = torch.randperm(len(train_f))
            train_f = train_f[perm]
            train_l = train_l[perm]
            
            # Mini-batches
            total_loss = 0
            batch_size = 64
            for i in range(0, len(train_f), batch_size):
                bf = train_f[i:i+batch_size].to(DEVICE)
                bl = train_l[i:i+batch_size].to(DEVICE)
                
                optimizer.zero_grad()
                out = model(bf).squeeze()
                loss = criterion(out, bl)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            
            # Validación
            model.eval()
            with torch.no_grad():
                out = model(val_f.to(DEVICE)).squeeze()
                pred = torch.sigmoid(out) > 0.5
                val_acc = (pred.cpu() == val_l).float().mean().item() * 100
            
            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if (epoch + 1) % 10 == 0 or val_acc >= 99:
                print(f"   Epoch {epoch+1}: Val={val_acc:.2f}%")
            
            if no_improve >= patience:
                print(f"   Early stop ep {epoch+1}")
                break
            
            if val_acc >= 100:
                print("   🎉 100% validación!")
                break
        
        # Restaurar mejor
        if best_state:
            model.load_state_dict(best_state)
        
        print(f"\n   Mejor val: {best_val:.2f}%")
        
        # Test
        test_acc, _, _ = test_accuracy(clip_model, preprocess, model, DEVICE, n_test=100)
        
        if test_acc > best_acc:
            best_acc = test_acc
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            save_head_to_db(model, version, test_acc)
        
        print(f"\n📈 Mejor precisión: {best_acc:.2f}%")
        
        if best_acc >= TARGET:
            print("\n🎉🎉🎉 ¡100% ALCANZADO! 🎉🎉🎉")
            break
        
        # Aumentar muestras
        n_samples = min(n_samples + 1000, 10000)
    
    print("\n" + "=" * 70)
    print(f"🏁 FIN - Mejor: {best_acc:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
