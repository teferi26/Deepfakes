from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy.orm import Session

from .database import get_db
from .auth import decode_access_token, get_user_by_id, get_user_by_api_key
from .models import User

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db),
) -> User:
    """
    Obtiene usuario actual desde JWT token o API key.
    Prioriza JWT si ambos están presentes.
    """
    # Intentar con JWT Bearer token
    if credentials:
        payload = decode_access_token(credentials.credentials)
        if payload:
            user_id = payload.get("sub")
            if user_id:
                user = get_user_by_id(db, user_id)
                if user:
                    return user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Intentar con API Key
    if api_key:
        user = get_user_by_api_key(db, api_key)
        if user:
            return user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida",
        )
    
    # Sin credenciales
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciales requeridas (Bearer token o X-API-Key)",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Obtiene usuario actual si hay credenciales, None si no.
    No lanza excepción si no hay auth.
    """
    if not credentials and not api_key:
        return None
    
    try:
        return await get_current_user(credentials, api_key, db)
    except HTTPException:
        return None
