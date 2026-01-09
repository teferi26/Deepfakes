#!/usr/bin/env python3
"""Test del modelo video V2."""

import torch
import numpy as np
from PIL import Image
import clip
import psycopg2
import pickle
import torch.nn as nn

print('Cargando CLIP...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, preprocess = clip.load('ViT-L/14', device=device)
clip_model.eval()

class VideoClassifier(nn.Module):
    def __init__(self, input_dim=768):
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

print('Cargando modelo V2...')
conn = psycopg2.connect(host='db', database='frauddb', user='fraudapp', password='fraudpass')
cur = conn.cursor()
cur.execute("SELECT model_data, accuracy, source FROM video_heads WHERE source LIKE 'video_ai_v2%' ORDER BY accuracy DESC LIMIT 1")
row = cur.fetchone()

if not row:
    print("ERROR: No hay modelo V2 en la base de datos")
    exit(1)

print(f'Modelo: {row[2]}, accuracy: {row[1]}%')

model_data = pickle.loads(row[0])
classifier = VideoClassifier().to(device)
classifier.load_state_dict(model_data['state_dict'])
classifier.eval()
print('✓ Modelo V2 cargado correctamente')

# Frame AI-like (gradientes suaves sin ruido)
arr = np.zeros((224, 224, 3), dtype=np.uint8)
for y in range(224):
    for x in range(224):
        factor = (np.sin(x/30) * np.cos(y/30) + 1) / 2
        arr[y, x] = int(150 * (0.7 + 0.3 * factor))
img_ai = Image.fromarray(arr)

# Frame real (con ruido natural)
arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
img_real = Image.fromarray(arr)

with torch.no_grad():
    feat_ai = clip_model.encode_image(preprocess(img_ai).unsqueeze(0).to(device)).float()
    feat_ai = feat_ai / feat_ai.norm(dim=-1, keepdim=True)
    feat_real = clip_model.encode_image(preprocess(img_real).unsqueeze(0).to(device)).float()
    feat_real = feat_real / feat_real.norm(dim=-1, keepdim=True)
    prob_ai = torch.sigmoid(classifier(feat_ai)).item()
    prob_real = torch.sigmoid(classifier(feat_real)).item()

print(f'\nResultados de prueba:')
label_ai = 'AI' if prob_ai > 0.5 else 'REAL'
label_real = 'AI' if prob_real > 0.5 else 'REAL'
print(f'  Frame AI-like:   prob={prob_ai:.4f} -> {label_ai} {"✓" if label_ai == "AI" else "✗"}')
print(f'  Frame Real-like: prob={prob_real:.4f} -> {label_real} {"✓" if label_real == "REAL" else "✗"}')

conn.close()
print('\n✓ Test completado')
