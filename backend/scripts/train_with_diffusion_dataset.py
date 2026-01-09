#!/usr/bin/env python3
"""
Script para entrenar el modelo CLIP head con imágenes de Stable Diffusion (DiffusionDB).

Combina:
- Imágenes AI de DiffusionDB (Stable Diffusion)
- Imágenes reales del dataset Hemg

Uso:
    python scripts/train_with_diffusion_dataset.py --ai-samples 25000 --real-samples 25000 --save
"""

import os
import sys
import argparse
import logging
import random
from io import BytesIO
from datetime import datetime
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CombinedDataset(Dataset):
    """Dataset que combina imágenes AI (DiffusionDB) con imágenes reales (Hemg)."""
    
    def __init__(
        self,
        ai_dataset,
        real_dataset,
        preprocess,
        ai_samples: int = 25000,
        real_samples: int = 25000
    ):
        self.preprocess = preprocess
        self.samples = []  # Lista de (dataset, index, label)
        
        logger.info(f"Preparando dataset combinado...")
        logger.info(f"  AI samples target: {ai_samples}")
        logger.info(f"  Real samples target: {real_samples}")
        
        # Añadir imágenes AI de DiffusionDB (todas son AI, label=1)
        ai_count = 0
        for i, item in enumerate(ai_dataset):
            if ai_count >= ai_samples:
                break
            try:
                if item.get('image') is not None:
                    self.samples.append(('ai', i, 1))
                    ai_count += 1
                    if ai_count % 5000 == 0:
                        logger.info(f"  Indexadas {ai_count} imágenes AI...")
            except:
                continue
        
        logger.info(f"  ✓ {ai_count} imágenes AI indexadas")
        
        # Añadir imágenes reales del dataset Hemg (label=1 en Hemg = Real, nuestro label=0)
        real_count = 0
        for i, item in enumerate(real_dataset):
            if real_count >= real_samples:
                break
            try:
                if item.get('image') is not None and item.get('label') == 1:  # label=1 = Real en Hemg
                    self.samples.append(('real', i, 0))
                    real_count += 1
                    if real_count % 5000 == 0:
                        logger.info(f"  Indexadas {real_count} imágenes reales...")
            except:
                continue
        
        logger.info(f"  ✓ {real_count} imágenes reales indexadas")
        
        # Guardar referencias a los datasets
        self.ai_dataset = ai_dataset
        self.real_dataset = real_dataset
        
        # Shuffle
        random.shuffle(self.samples)
        
        logger.info(f"Dataset combinado: {len(self.samples)} total ({ai_count} AI + {real_count} Real)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        source, data_idx, label = self.samples[idx]
        
        try:
            if source == 'ai':
                item = self.ai_dataset[data_idx]
            else:
                item = self.real_dataset[data_idx]
            
            image = item.get('image')
            if image is None:
                return torch.zeros(3, 224, 224), torch.tensor(0.0)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_tensor = self.preprocess(image)
            return image_tensor, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            return torch.zeros(3, 224, 224), torch.tensor(0.0)


def extract_clip_features(
    model,
    dataloader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae features CLIP de todas las imágenes."""
    
    all_features = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Extraídas features de {(batch_idx + 1) * len(images)} imágenes...")
    
    return np.vstack(all_features), np.concatenate(all_labels)


def train_linear_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    device: str,
    learning_rate: float = 0.1,
    epochs: int = 30,
    batch_size: int = 256
) -> Tuple[nn.Module, dict]:
    """Entrena un head linear."""
    
    input_dim = train_features.shape[1]
    logger.info(f"Entrenando head linear: input_dim={input_dim}, lr={learning_rate}, epochs={epochs}")
    
    classifier = nn.Sequential(
        nn.Linear(input_dim, 1)
    ).to(device)
    
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)
    X_val = torch.tensor(val_features, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_labels, dtype=torch.float32).to(device)
    
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        classifier.train()
        
        indices = torch.randperm(len(X_train))
        total_loss = 0.0
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_x = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            optimizer.zero_grad()
            outputs = classifier(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validación
        classifier.eval()
        with torch.no_grad():
            val_outputs = classifier(X_val).squeeze()
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()
            
            train_outputs = classifier(X_train).squeeze()
            train_preds = (torch.sigmoid(train_outputs) > 0.5).float()
            train_acc = (train_preds == y_train).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}: loss={total_loss/(len(X_train)//batch_size):.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    if best_state:
        classifier.load_state_dict(best_state)
    
    classifier.eval()
    with torch.no_grad():
        val_outputs = classifier(X_val).squeeze()
        val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
        final_val_acc = (val_preds == y_val).float().mean().item()
        
        train_outputs = classifier(X_train).squeeze()
        train_preds = (torch.sigmoid(train_outputs) > 0.5).float()
        final_train_acc = (train_preds == y_train).float().mean().item()
    
    metrics = {
        'val_acc': final_val_acc,
        'train_acc': final_train_acc,
        'n_train': len(train_labels),
        'n_val': len(val_labels),
        'learning_rate': learning_rate,
        'epochs': epochs,
        'architecture': 'linear',
        'dataset': 'diffusiondb+hemg'
    }
    
    logger.info(f"Entrenamiento completado: val_acc={final_val_acc:.4f}, train_acc={final_train_acc:.4f}")
    
    return classifier, metrics


def save_model_to_db(classifier: nn.Module, metrics: dict, input_dim: int):
    """Guarda el modelo entrenado en la base de datos."""
    
    from app.database import SessionLocal
    from app.models import ModelArtifact
    
    state_dict = {}
    for name, param in classifier.named_parameters():
        new_name = name.replace('0.', 'fc.')
        state_dict[new_name] = param.data.clone()
    
    payload = {
        'state_dict': state_dict,
        'architecture': 'linear',
        'in_dim': input_dim,
        'metrics': metrics
    }
    
    buffer = BytesIO()
    torch.save(payload, buffer)
    artifact_bytes = buffer.getvalue()
    
    version = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    
    db = SessionLocal()
    try:
        artifact = ModelArtifact(
            name='clip_head_global',
            version=version,
            artifact=artifact_bytes,
            metrics=metrics
        )
        db.add(artifact)
        db.commit()
        
        logger.info(f"✓ Modelo guardado: clip_head_global v{version}")
        return version
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description='Entrenar CLIP head con DiffusionDB + imágenes reales')
    parser.add_argument('--ai-samples', type=int, default=25000,
                        help='Número de imágenes AI de DiffusionDB')
    parser.add_argument('--real-samples', type=int, default=25000,
                        help='Número de imágenes reales de Hemg')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size para extracción')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs de entrenamiento')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Proporción de validación')
    parser.add_argument('--save', action='store_true',
                        help='Guardar modelo en base de datos')
    
    args = parser.parse_args()
    
    from datasets import load_dataset
    import open_clip
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Usando dispositivo: {device}")
    
    # Cargar CLIP
    logger.info("Cargando CLIP ViT-L-14...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained='openai',
        device=device
    )
    model.eval()
    logger.info("✓ CLIP cargado")
    
    # Cargar datasets
    logger.info("Descargando DiffusionDB (imágenes AI de Stable Diffusion)...")
    # Usar 2m_random_10k para descarga más rápida y estable
    ai_dataset = load_dataset('poloclub/diffusiondb', '2m_random_10k', 
                               trust_remote_code=True, split='train')
    logger.info(f"✓ DiffusionDB cargado: {len(ai_dataset)} imágenes")
    
    logger.info("Descargando Hemg dataset (imágenes reales)...")
    real_dataset = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train')
    logger.info(f"✓ Hemg cargado: {len(real_dataset)} imágenes")
    
    # Shuffle datasets
    ai_dataset = ai_dataset.shuffle(seed=42)
    real_dataset = real_dataset.shuffle(seed=42)
    
    # Crear dataset combinado
    dataset = CombinedDataset(
        ai_dataset,
        real_dataset,
        preprocess,
        ai_samples=args.ai_samples,
        real_samples=args.real_samples
    )
    
    if len(dataset) == 0:
        logger.error("Dataset vacío!")
        sys.exit(1)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Extraer features
    logger.info("Extrayendo features CLIP...")
    features, labels = extract_clip_features(model, dataloader, device)
    logger.info(f"✓ Features extraídas: shape={features.shape}")
    
    # Split train/val
    n_samples = len(labels)
    n_val = int(n_samples * args.val_split)
    indices = np.random.permutation(n_samples)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    
    # Mostrar distribución
    train_ai = sum(train_labels)
    train_real = len(train_labels) - train_ai
    val_ai = sum(val_labels)
    val_real = len(val_labels) - val_ai
    
    logger.info(f"Split: {len(train_labels)} train ({int(train_ai)} AI, {int(train_real)} Real)")
    logger.info(f"       {len(val_labels)} val ({int(val_ai)} AI, {int(val_real)} Real)")
    
    # Entrenar
    classifier, metrics = train_linear_head(
        train_features, train_labels,
        val_features, val_labels,
        device=device,
        learning_rate=args.lr,
        epochs=args.epochs
    )
    
    # Guardar
    if args.save:
        version = save_model_to_db(classifier, metrics, features.shape[1])
        logger.info(f"\n🎉 Modelo guardado exitosamente!")
        logger.info(f"   Versión: {version}")
        logger.info(f"   Val Accuracy: {metrics['val_acc']*100:.1f}%")
        logger.info(f"   Dataset: DiffusionDB (Stable Diffusion) + Hemg (Real)")
        logger.info(f"\n   Para usar el nuevo modelo, reinicia el worker:")
        logger.info(f"   docker-compose restart worker")
    else:
        logger.info("\n✓ Entrenamiento completado (sin guardar)")
        logger.info(f"   Val Accuracy: {metrics['val_acc']*100:.1f}%")


if __name__ == '__main__':
    main()
