from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import UserCreate, UserLogin, Token, UserResponse, APIKeyResponse
from ..auth import (
    create_user,
    authenticate_user,
    create_access_token,
    get_user_by_email,
    regenerate_api_key,
    JWT_EXPIRATION_HOURS,
)
from ..deps import get_current_user
from ..models import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(data: UserCreate, db: Session = Depends(get_db)):
    """Registra un nuevo usuario."""
    existing = get_user_by_email(db, data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El email ya está registrado"
        )
    
    user = create_user(db, data.email, data.password)
    return UserResponse(
        id=str(user.id),
        email=user.email,
        role=user.role.value,
        api_key=user.api_key,
        created_at=user.created_at,
    )


@router.post("/login", response_model=Token)
def login(data: UserLogin, db: Session = Depends(get_db)):
    """Inicia sesión y devuelve JWT token."""
    user = authenticate_user(db, data.email, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos",
        )
    
    token = create_access_token(str(user.id))
    return Token(
        access_token=token,
        expires_in=JWT_EXPIRATION_HOURS * 3600,
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Obtiene información del usuario actual."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        role=current_user.role.value,
        api_key=current_user.api_key,
        created_at=current_user.created_at,
    )


@router.post("/api-key/regenerate", response_model=APIKeyResponse)
def regenerate_key(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Regenera la API key del usuario actual."""
    new_key = regenerate_api_key(db, current_user)
    return APIKeyResponse(api_key=new_key)
