# Alcance MVP - Plataforma de Detección de Fraude Audiovisual

## Soporte de medios
- Imágenes: JPG, PNG, WebP hasta 20 MB.
- Video: MP4 (H.264) hasta 120 segundos, 1080p, 200 MB. Procesamiento asíncrono con cola.
- Audio: pendiente fase siguiente (no bloquea MVP).

## Idiomas
- Español e inglés (UI e informes).

## Autenticación y acceso
- Registro/login básico con JWT.
- API keys para clientes.

## Salida del informe (MVP)
- Probabilidad 0-100% de manipulación por IA.
- Tipo sospechado: deepfake rostro, manipulación visual, síntesis completa, indeterminado.
- Explicación breve para público no técnico.
- Timestamp, hash del archivo, ID de caso, versión de modelo.
- Informe ampliado (pago futuro): métricas del modelo, heatmaps/frames clave, trazas, desglose por detector.

## Retención y privacidad
- Guardar binario original + resultado.
- Retención inicial sugerida: 30-90 días configurable; borrado bajo solicitud.
- Cifrado en tránsito y reposo.
- Uso de datos para mejora solo con opt-in explícito (desactivado por defecto).

## KPIs del panel
- Tasa de detección por tipo de medio.
- Falsos positivos estimados.
- Tiempo de procesamiento p50/p95.
- Volumen de análisis por día/cliente.
- Coste por job aproximado.
- Versiones de modelo en producción.

## Arquitectura resumida (MVP)
- Frontend: Next.js (SSR/ISR), bilingüe.
- Backend API: FastAPI (Python).
- Asíncrono: Celery + Redis.
- Almacenamiento binarios: S3 compatible (AWS S3 o MinIO local).
- Base de datos: PostgreSQL (metadatos, usuarios, casos, resultados).
- Workers IA: contenedores GPU-ready con detectores open-source.
- Observabilidad: Prometheus + Grafana (a integrar), logs estructurados, trazas OpenTelemetry (posterior).
- Seguridad: HTTPS, JWT, API keys, rate limiting, CORS, antivirus ligero (ClamAV) en uploads.

## Roadmap alto nivel
1. Setup repo y contenedores locales (Postgres, Redis, MinIO, API, worker, frontend).
2. Auth básica y API keys.
3. Upload/cola y almacenamiento.
4. Detector imagen v1 (pretrained open-source) y endpoint de informe básico.
5. Agregación video v1.
6. UI básica bilingüe.
7. Observabilidad inicial.
8. Seguridad y límites.
9. Consentimiento y retención.
10. Informe ampliado (feature flag).
11. Camino forense (cadena de custodia y estándares).
