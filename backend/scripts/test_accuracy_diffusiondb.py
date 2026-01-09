#!/usr/bin/env python3
"""
Script para probar la precisión del detector con:
- 100 imágenes AI de DiffusionDB (Stable Diffusion) - mismo tipo que entrenamiento
- 100 imágenes reales del dataset Hemg
"""
import sys
import os
sys.path.insert(0, '/app')

from io import BytesIO
from datasets import load_dataset


def main():
    from app.detectors.clip_detector import CLIPDetector
    
    print("=" * 70)
    print("TEST DE PRECISIÓN: 100 AI (DiffusionDB) + 100 reales (Hemg)")
    print("=" * 70)
    print()
    
    print("Inicializando detector CLIP...")
    detector = CLIPDetector()
    print(f"✓ Detector CLIP cargado correctamente")
    print(f"  Head version: {detector._head_version}")
    print(f"  Head source: {detector._head_source}")
    print(f"  Classifier type: {type(detector.classifier).__name__}")
    print()
    
    # Cargar DiffusionDB para imágenes AI (mismo dataset de entrenamiento)
    print("Cargando DiffusionDB (Stable Diffusion AI images)...")
    ds_ai = load_dataset('poloclub/diffusiondb', '2m_random_1k', split='train', trust_remote_code=True)
    ds_ai = ds_ai.shuffle(seed=999)
    print(f"  ✓ DiffusionDB cargado: {len(ds_ai)} imágenes disponibles")
    
    # Cargar Hemg para imágenes reales
    print("Cargando Hemg (imágenes reales)...")
    ds_hemg = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")
    ds_hemg = ds_hemg.shuffle(seed=999)
    print(f"  ✓ Hemg cargado: {len(ds_hemg)} imágenes disponibles")
    print()
    
    # Recolectar imágenes
    ai_images = []
    real_images = []
    
    print("Recolectando imágenes del dataset...")
    
    # 100 imágenes AI de DiffusionDB
    for i, item in enumerate(ds_ai):
        if len(ai_images) >= 100:
            break
        ai_images.append(item['image'])
    
    # 100 imágenes reales de Hemg (label=1)
    for item in ds_hemg:
        if item['label'] == 1 and len(real_images) < 100:  # 1 = Real
            real_images.append(item['image'])
        if len(real_images) >= 100:
            break
    
    print(f"✓ Recolectadas {len(ai_images)} imágenes AI (DiffusionDB/Stable Diffusion)")
    print(f"✓ Recolectadas {len(real_images)} imágenes reales (Hemg)")
    print()
    
    # Test de imágenes AI
    print("-" * 70)
    print("FASE 1: Analizando 100 imágenes de Stable Diffusion (DiffusionDB)")
    print("-" * 70)
    
    ai_results = []
    for i, img in enumerate(ai_images):
        try:
            # Convertir a bytes
            buf = BytesIO()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(buf, format='PNG')
            buf.seek(0)
            
            # Analizar
            result = detector.analyze(buf.read())
            prob = result.get('probability', 0)
            # CLIPDetector no devuelve is_synthetic, calcularlo
            is_synthetic = prob > 0.5
            
            ai_results.append({
                'prob': prob,
                'is_synthetic': is_synthetic,
                'correct': is_synthetic  # Debería ser detectada como sintética
            })
            
            status = "✓" if is_synthetic else "✗"
            print(f"  {status} [{i+1:3d}/100] Prob: {prob*100:5.1f}% | Sintética: {is_synthetic}")
            
        except Exception as e:
            print(f"  ? [{i+1:3d}/100] Error: {e}")
            ai_results.append({'prob': 0, 'is_synthetic': False, 'correct': False})
    
    print()
    
    # Test de imágenes reales
    print("-" * 70)
    print("FASE 2: Analizando 100 imágenes reales (Hemg)")
    print("-" * 70)
    
    real_results = []
    for i, img in enumerate(real_images):
        try:
            # Convertir a bytes
            buf = BytesIO()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(buf, format='PNG')
            buf.seek(0)
            
            # Analizar
            result = detector.analyze(buf.read())
            prob = result.get('probability', 0)
            # CLIPDetector no devuelve is_synthetic, calcularlo
            is_synthetic = prob > 0.5
            
            real_results.append({
                'prob': prob,
                'is_synthetic': is_synthetic,
                'correct': not is_synthetic  # NO debería ser detectada como sintética
            })
            
            status = "✓" if not is_synthetic else "✗"
            print(f"  {status} [{i+1:3d}/100] Prob: {prob*100:5.1f}% | Sintética: {is_synthetic}")
            
        except Exception as e:
            print(f"  ? [{i+1:3d}/100] Error: {e}")
            real_results.append({'prob': 0, 'is_synthetic': False, 'correct': True})
    
    print()
    
    # Calcular estadísticas
    print("=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)
    print()
    
    # Estadísticas de imágenes AI
    ai_correct = sum(1 for r in ai_results if r['correct'])
    ai_incorrect = len(ai_results) - ai_correct
    ai_avg_prob = sum(r['prob'] for r in ai_results) / len(ai_results)
    
    print("IMÁGENES AI - Stable Diffusion (100 imágenes):")
    print(f"  Probabilidad promedio: {ai_avg_prob*100:.1f}%")
    print(f"  ✓ Correctamente detectadas como AI:  {ai_correct:3d} ({ai_correct/len(ai_results)*100:.1f}%)")
    print(f"  ✗ Incorrectamente marcadas como real: {ai_incorrect:3d} ({ai_incorrect/len(ai_results)*100:.1f}%)")
    print()
    
    # Estadísticas de imágenes reales
    real_correct = sum(1 for r in real_results if r['correct'])
    real_incorrect = len(real_results) - real_correct
    real_avg_prob = sum(r['prob'] for r in real_results) / len(real_results)
    
    print("IMÁGENES REALES (100 imágenes):")
    print(f"  Probabilidad promedio: {real_avg_prob*100:.1f}%")
    print(f"  ✓ Correctamente detectadas como real: {real_correct:3d} ({real_correct/len(real_results)*100:.1f}%)")
    print(f"  ✗ Incorrectamente marcadas como AI:  {real_incorrect:3d} ({real_incorrect/len(real_results)*100:.1f}%)")
    print()
    
    # Accuracy total
    total_correct = ai_correct + real_correct
    total_images = len(ai_results) + len(real_results)
    accuracy = (total_correct / total_images) * 100
    
    print("=" * 70)
    print(f"ACCURACY TOTAL: {accuracy:.2f}%")
    print(f"  Aciertos: {total_correct}/{total_images}")
    print(f"  - True Positives (AI detectada):    {ai_correct}/100")
    print(f"  - True Negatives (Real detectada):  {real_correct}/100")
    print(f"  - False Negatives (AI como real):   {ai_incorrect}/100")
    print(f"  - False Positives (Real como AI):   {real_incorrect}/100")
    print("=" * 70)
    
    # Métricas adicionales
    precision = ai_correct / (ai_correct + real_incorrect) if (ai_correct + real_incorrect) > 0 else 0
    recall = ai_correct / len(ai_results) if len(ai_results) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print()
    print("MÉTRICAS ADICIONALES:")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1_score*100:.2f}%")
    print()


if __name__ == "__main__":
    main()
