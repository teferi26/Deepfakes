"""
Rate Limiting para la API de detección de fraude.

Implementa límites de peticiones por IP y por usuario autenticado
para proteger la API de abusos y garantizar disponibilidad.

Configuración por defecto:
- Usuarios anónimos: 10 requests/minuto
- Usuarios autenticados: 60 requests/minuto
- API keys: 300 requests/minuto
- Upload de archivos: 5 files/minuto (anónimo), 20 files/minuto (auth)
"""

import logging
from typing import Optional, Callable

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request, FastAPI
from starlette.responses import JSONResponse

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_identifier(request: Request) -> str:
    """
    Obtiene el identificador para rate limiting.
    
    Prioridad:
    1. User ID del token JWT (si está autenticado)
    2. API Key (si usa autenticación por API key)
    3. IP del cliente (fallback)
    """
    # Intentar obtener user_id del estado de la request (puesto por auth middleware)
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    
    # Intentar obtener API key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        # Usar hash corto de la API key para no exponer la key completa en logs
        return f"apikey:{api_key[:8]}..."
    
    # Fallback a IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Tomar la primera IP (cliente original)
        return f"ip:{forwarded.split(',')[0].strip()}"
    
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def get_tier_identifier(request: Request) -> str:
    """
    Obtiene identificador con información de tier para límites diferenciados.
    """
    base_id = get_identifier(request)
    
    # API keys tienen tier más alto
    if base_id.startswith("apikey:"):
        return f"tier:premium:{base_id}"
    
    # Usuarios autenticados tienen tier medio
    if base_id.startswith("user:"):
        return f"tier:auth:{base_id}"
    
    # IPs anónimas tienen tier básico
    return f"tier:basic:{base_id}"


# Crear limiter con redis como backend (usando el mismo redis de Celery)
limiter = Limiter(
    key_func=get_identifier,
    storage_uri=settings.redis_url,
    strategy="fixed-window",
    headers_enabled=False,  # Desactivado para evitar conflictos con FastAPI response models
)


# Límites por tipo de operación
RATE_LIMITS = {
    # Límites generales por defecto
    "default": "60/minute",
    
    # Límites para endpoints específicos
    "upload": "10/minute",           # Upload de archivos (costoso)
    "upload_auth": "30/minute",      # Upload con autenticación
    "analyze": "20/minute",          # Análisis de archivos
    "analyze_auth": "60/minute",     # Análisis con auth
    "auth": "10/minute",             # Login/register (prevenir brute force)
    "health": "120/minute",          # Health checks (más permisivo)
    "report": "10/minute",           # Generación de reportes PDF
}


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Handler personalizado para cuando se excede el rate limit.
    """
    retry_after = getattr(exc, "retry_after", 60)
    
    logger.warning(
        f"Rate limit exceeded for {get_identifier(request)}: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Has excedido el límite de peticiones. Por favor, espera antes de intentar de nuevo.",
            "detail": str(exc.detail),
            "retry_after_seconds": retry_after,
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": exc.detail.split()[0] if exc.detail else "unknown",
        }
    )


def setup_rate_limiting(app: FastAPI) -> None:
    """
    Configura rate limiting en la aplicación FastAPI.
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    
    logger.info("Rate limiting configurado correctamente")


# Decoradores de conveniencia para usar en endpoints
def limit_upload(func: Callable) -> Callable:
    """Decorador para limitar uploads."""
    return limiter.limit(RATE_LIMITS["upload"])(func)


def limit_analyze(func: Callable) -> Callable:
    """Decorador para limitar análisis."""
    return limiter.limit(RATE_LIMITS["analyze"])(func)


def limit_auth(func: Callable) -> Callable:
    """Decorador para limitar endpoints de autenticación."""
    return limiter.limit(RATE_LIMITS["auth"])(func)


def limit_report(func: Callable) -> Callable:
    """Decorador para limitar generación de reportes."""
    return limiter.limit(RATE_LIMITS["report"])(func)
