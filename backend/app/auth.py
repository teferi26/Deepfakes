import secrets
from datetime import datetime, timedelta
from typing import Optional

import jwt
import bcrypt
from sqlalchemy.orm import Session

from .config import settings
from .models import User


# Configuración JWT
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


def hash_password(password: str) -> str:
    """Hash de contraseña con bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica contraseña contra hash."""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def generate_api_key() -> str:
    """Genera API key única de 64 caracteres."""
    return secrets.token_hex(32)


def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Crea JWT token."""
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=JWT_EXPIRATION_HOURS))
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Decodifica y valida JWT token."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Busca usuario por email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    """Busca usuario por ID."""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_api_key(db: Session, api_key: str) -> Optional[User]:
    """Busca usuario por API key."""
    return db.query(User).filter(User.api_key == api_key).first()


def create_user(db: Session, email: str, password: str) -> User:
    """Crea nuevo usuario."""
    user = User(
        email=email,
        hashed_password=hash_password(password),
        api_key=generate_api_key(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Autentica usuario por email y contraseña."""
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def regenerate_api_key(db: Session, user: User) -> str:
    """Regenera API key para un usuario."""
    user.api_key = generate_api_key()
    db.commit()
    db.refresh(user)
    return user.api_key
