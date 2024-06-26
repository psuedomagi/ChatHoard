from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import HashingError, InvalidHashError, VerifyMismatchError
from jwt import PyJWTError

from app.core.config import settings

# TODO: Add key generation script to container instantiation

# Load the private and public keys
with open(settings.PRIVATE_KEY_PATH, "rb") as key_file:
    PRIVATE_KEY = key_file.read()

with open(settings.PUBLIC_KEY_PATH, "rb") as key_file:
    PUBLIC_KEY = key_file.read()


@contextmanager
def hasher_context() -> Any:
    try:
        yield PasswordHasher(settings.HASH_PARAMS)
    except VerifyMismatchError as e:
        raise ValueError(f"Hash verification failed: {e}") from e
    except HashingError as e:
        raise ValueError(f"Hashing failed: {e}") from e
    except InvalidHashError as e:
        raise ValueError(f"Invalid hash: {e}") from e


def create_access_token(subject: str | Any, expires_delta: timedelta) -> str:
    expire = datetime.now(UTC) + expires_delta
    to_encode = {"exp": expire, "sub": str(subject)}
    return jwt.encode(to_encode, PRIVATE_KEY, algorithm=settings.JWT_ALGORITHM)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    with hasher_context() as hasher:
        return hasher.verify(hashed_password, plain_password)


def get_password_hash(password: str) -> str:
    with hasher_context() as hasher:
        return hasher.hash(password)


def decode_access_token(token: str) -> Any:
    try:
        return jwt.decode(token, PUBLIC_KEY, algorithms=[settings.JWT_ALGORITHM])
    except PyJWTError as e:
        raise ValueError(f"Invalid token: {e}") from e


def verify_access_token(token: str) -> Any:
    try:
        payload = decode_access_token(token)
        return payload["sub"]
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        raise ValueError(f"Token verification failed: {e}") from e
