#!/usr/bin/env python3
"""
Script para entrenar el modelo CLIP head con datasets modernos de IA.

Usa el dataset Hemg/AI-Generated-vs-Real-Images-Datasets de HuggingFace
que contiene ~152K imágenes de AI Art vs Real Images.

Uso:
    python scripts/train_with_modern_dataset.py --max-samples 50000 --batch-size 64
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


def check_dependencies():
    """Verificar que las dependencias están instaladas."""
    try:
        from datasets import load_dataset
        import open_clip
        logger.info("✓ Dependencias OK")
        return True
    except ImportError as e:
        logger.error(f"Falta dependencia: {e}")
        logger.info("Instala con: pip install datasets open_clip_torch")
        return False


class ModernAIDataset(Dataset):
    """Dataset para imágenes modernas de IA vs Reales."""
    
    def __init__(
        self,
        hf_dataset,
        preprocess,
        max_samples: Optional[int] = None,
        split: str = "train"
    ):
        self.preprocess = preprocess
        self.samples = []
        
        logger.info(f"Procesando dataset (max_samples={max_samples})...")
        
        # El dataset Hemg/AI-Generated-vs-Real-Images-Datasets tiene:
        # label: 0 = AiArtData (IA), 1 = RealArt (Real)
        # Para nuestro modelo: 1 = AI, 0 = Real
        # Por tanto: invertir el label
        for i, item in enumerate(hf_dataset):
            if max_samples and i >= max_samples:
                break
            
            try:
                image = item.get('image')
                label_raw = item.get('label', item.get('class', 0))
                
                # Dataset usa: 0=AI, 1=Real
                # Nuestro modelo usa: 1=AI, 0=Real
                # Por tanto: invertir (1 - label_raw)
                if isinstance(label_raw, str):
                    label = 1 if 'ai' in label_raw.lower() or 'gen' in label_raw.lower() else 0
                else:
                    # Hemg dataset: 0=AI, 1=Real -> invertir a 1=AI, 0=Real
                    label = 1 - int(label_raw)
                
                if image is not None:
                    self.samples.append((image, label))
                    
                if (i + 1) % 10000 == 0:
                    logger.info(f"  Procesadas {i + 1} imágenes...")
                    
            except Exception as e:
                continue
        
        # Contar distribución
        ai_count = sum(1 for _, l in self.samples if l == 1)
        real_count = len(self.samples) - ai_count
        logger.info(f"Dataset cargado: {len(self.samples)} total, {ai_count} AI, {real_count} Real")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image, label = self.samples[idx]
        
        try:
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Aplicar preprocesamiento CLIP
            image_tensor = self.preprocess(image)
            return image_tensor, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            # Imagen dummy en caso de error
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
            
            # Extraer features
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
    learning_rate: float = 0.05,
    epochs: int = 20,
    batch_size: int = 256
) -> Tuple[nn.Module, dict]:
    """Entrena un head linear simple (como el modelo que funcionaba)."""
    
    input_dim = train_features.shape[1]
    logger.info(f"Entrenando head linear: input_dim={input_dim}, lr={learning_rate}")
    
    # Crear clasificador linear simple
    classifier = nn.Sequential(
        nn.Linear(input_dim, 1)
    ).to(device)
    
    # Convertir a tensores
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)
    X_val = torch.tensor(val_features, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_labels, dtype=torch.float32).to(device)
    
    # Optimizer y loss
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        classifier.train()
        
        # Mini-batch training
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
        
        # Validación
        classifier.eval()
        with torch.no_grad():
            val_outputs = classifier(X_val).squeeze()
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()
            
            train_outputs = classifier(X_train).squeeze()
            train_preds = (torch.sigmoid(train_outputs) > 0.5).float()
            train_acc = (train_preds == y_train).float().mean().item()
        
        avg_loss = total_loss / (len(X_train) // batch_size)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    # Restaurar mejor modelo
    if best_state:
        classifier.load_state_dict(best_state)
    
    # Evaluación final
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
        'architecture': 'linear'
    }
    
    logger.info(f"Entrenamiento completado: val_acc={final_val_acc:.4f}, train_acc={final_train_acc:.4f}")
    
    return classifier, metrics


def save_model_to_db(classifier: nn.Module, metrics: dict, input_dim: int):
    """Guarda el modelo entrenado en la base de datos."""
    
    from app.database import SessionLocal
    from app.models import ModelArtifact
    
    # Preparar state dict para el formato esperado por CLIPDetector
    state_dict = {}
    for name, param in classifier.named_parameters():
        # Convertir nombres: 0.weight -> fc.weight
        new_name = name.replace('0.', 'fc.')
        state_dict[new_name] = param.data.clone()
    
    payload = {
        'state_dict': state_dict,
        'architecture': 'linear',
        'in_dim': input_dim,
        'metrics': metrics
    }
    
    # Serializar
    buffer = BytesIO()
    torch.save(payload, buffer)
    artifact_bytes = buffer.getvalue()
    
    # Guardar en DB
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
        logger.info(f"  Métricas: val_acc={metrics['val_acc']:.4f}, train_acc={metrics['train_acc']:.4f}")
        
        return version
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description='Entrenar CLIP head con dataset moderno de IA')
    parser.add_argument('--dataset', default='Hemg/AI-Generated-vs-Real-Images-Datasets',
                        help='Dataset de HuggingFace')
    parser.add_argument('--max-samples', type=int, default=100000,
                        help='Máximo de samples a usar (default: 100000)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size para extracción de features')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate (default: 0.05)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs de entrenamiento')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Proporción de validación')
    parser.add_argument('--save', action='store_true',
                        help='Guardar modelo en base de datos')
    
    args = parser.parse_args()
    
    if not check_dependencies():
        sys.exit(1)
    
    from datasets import load_dataset
    import open_clip
    
    # Determinar dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Usando dispositivo: {device}")
    
    # Cargar modelo CLIP
    logger.info("Cargando CLIP ViT-L-14...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained='openai',
        device=device
    )
    model.eval()
    logger.info("✓ CLIP cargado")
    
    # Cargar dataset
    logger.info(f"Descargando dataset: {args.dataset}")
    logger.info("(Esto puede tardar unos minutos la primera vez...)")
    
    hf_dataset = load_dataset(args.dataset, split='train', streaming=False)
    logger.info(f"✓ Dataset descargado: {len(hf_dataset)} imágenes")
    
    # IMPORTANTE: Shuffle del dataset (está ordenado por clase)
    hf_dataset = hf_dataset.shuffle(seed=42)
    logger.info("✓ Dataset shuffled")
    
    # Crear dataset
    dataset = ModernAIDataset(
        hf_dataset,
        preprocess,
        max_samples=args.max_samples
    )
    
    if len(dataset) == 0:
        logger.error("Dataset vacío!")
        sys.exit(1)
    
    # DataLoader
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
    
    logger.info(f"Split: {len(train_labels)} train, {len(val_labels)} val")
    
    # Entrenar head linear
    classifier, metrics = train_linear_head(
        train_features, train_labels,
        val_features, val_labels,
        device=device,
        learning_rate=args.lr,
        epochs=args.epochs
    )
    
    # Guardar modelo
    if args.save:
        version = save_model_to_db(classifier, metrics, features.shape[1])
        logger.info(f"\n🎉 Modelo guardado exitosamente!")
        logger.info(f"   Versión: {version}")
        logger.info(f"   Val Accuracy: {metrics['val_acc']*100:.1f}%")
        logger.info(f"   Train Accuracy: {metrics['train_acc']*100:.1f}%")
        logger.info(f"\n   Para usar el nuevo modelo, reinicia el worker:")
        logger.info(f"   docker-compose restart worker")
    else:
        logger.info("\n✓ Entrenamiento completado (sin guardar)")
        logger.info(f"   Val Accuracy: {metrics['val_acc']*100:.1f}%")
        logger.info(f"   Para guardar, usa --save")


if __name__ == '__main__':
    main()
