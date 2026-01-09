#!/usr/bin/env python3
"""
Script para probar detección de imágenes AI.
Usa imágenes del dataset de HuggingFace.
"""
import sys
import os
sys.path.insert(0, '/app')

from io import BytesIO


def main():
    from datasets import load_dataset
    from app.detectors.clip_detector import CLIPDetector
    
    print("Cargando detector CLIP...")
    detector = CLIPDetector()
    print(f"  Head version: {detector._head_version}")
    print(f"  Head source: {detector._head_source}")
    print()
    
    print("Cargando dataset de HuggingFace...")
    ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train', streaming=False)
    ds = ds.shuffle(seed=789)
    
    # Filtrar solo imágenes AI (label=0 en este dataset = AI)
    ai_images = []
    for i, item in enumerate(ds):
        if item['label'] == 0:  # 0 = AI en este dataset
            ai_images.append(item['image'])
        if len(ai_images) >= 50:
            break
    
    print(f"Encontradas {len(ai_images)} imágenes AI para probar")
    print()
    
    # Analizar cada imagen con CLIP directamente
    results = []
    
    for i, img in enumerate(ai_images):
        try:
            # Convertir a bytes
            buf = BytesIO()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(buf, format='PNG')
            buf.seek(0)
            
            # Analizar con CLIP
            result = detector.analyze(buf.read())
            prob = result.get('probability', 0)
            
            results.append({'prob': prob})
            status = "✓" if prob >= 0.5 else "✗"
            print(f"  {status} [{i+1:2d}/50] CLIP prob: {prob*100:.1f}%")
            
        except Exception as e:
            print(f"  ? [{i+1:2d}/50] Error: {e}")
    
    # Estadísticas
    if results:
        avg_prob = sum(r['prob'] for r in results) / len(results)
        detected = sum(1 for r in results if r['prob'] >= 0.5)
        missed = len(results) - detected
        
        print()
        print("=" * 60)
        print(f"RESULTADOS: {len(results)} imágenes AI analizadas con CLIP")
        print("=" * 60)
        print(f"Probabilidad CLIP promedio: {avg_prob*100:.1f}%")
        print()
        print(f"✓ Detectadas como IA (>=50%): {detected:3d} ({detected/len(results)*100:.1f}%)")
        print(f"✗ No detectadas (<50%):       {missed:3d} ({missed/len(results)*100:.1f}%)")
        print()
        print(f"ACCURACY CLIP: {detected/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()
