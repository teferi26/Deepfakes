"""Test del Ensemble Detector ejecutado dentro del contenedor api.

Este script replica la lógica de `backend/test_ensemble.py` pero viviendo
directamente dentro del paquete `app`, de forma que la imagen Docker no
dependa de archivos fuera de `app/`.
"""

import urllib.request

from detectors import ImageDetector


def main() -> None:
    print("=== Probando Ensemble Detector ===")
    print()

    # Inicializar ensemble
    print("Inicializando ensemble...")
    detector = ImageDetector()
    print(f"Detectores disponibles: {len(detector.detectors)}")
    for d in detector.detectors:
        print(f"  - {d.name} (peso: {d.default_weight})")
    print()

    # Test 1: Imagen real
    print("Test 1: Imagen real (Unsplash)")
    url1 = "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400"
    img1 = urllib.request.urlopen(url1).read()
    result1 = detector.analyze(img1)
    print(f"  Probabilidad: {result1['probability']:.1%}")
    print(f"  Es sintetica: {result1['is_synthetic']}")
    print(f"  Confianza: {result1['confidence']}")
    print(f"  Tipo: {result1['suspected_type']}")
    if "individual_results" in result1:
        print("  Resultados individuales:")
        for ir in result1["individual_results"]:
            print(f"    - {ir['detector_name']}: {ir['probability']:.1%}")
    print()

    # Test 2: Imagen IA
    print("Test 2: Imagen generada por IA (thispersondoesnotexist)")
    url2 = "https://thispersondoesnotexist.com"
    req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
    img2 = urllib.request.urlopen(req2).read()
    result2 = detector.analyze(img2)
    print(f"  Probabilidad: {result2['probability']:.1%}")
    print(f"  Es sintetica: {result2['is_synthetic']}")
    print(f"  Confianza: {result2['confidence']}")
    print(f"  Tipo: {result2['suspected_type']}")
    if "individual_results" in result2:
        print("  Resultados individuales:")
        for ir in result2["individual_results"]:
            print(f"    - {ir['detector_name']}: {ir['probability']:.1%}")
    print()
    print("=== Test completado ===")


if __name__ == "__main__":
    main()
