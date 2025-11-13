from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24 * 365)))
DEFAULT_USER_USERNAME = os.getenv("DEFAULT_USER_USERNAME", "darius")
DEFAULT_USER_PASSWORD = os.getenv("DEFAULT_USER_PASSWORD", "")

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=128)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=128)
    password: str = Field(..., min_length=6, max_length=128)


class UserPublic(BaseModel):
    username: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime


class CurrentUser(BaseModel):
    username: str
    session_id: Optional[str] = None


_credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def _get_collection(request: Request, attr: str):
    collection = getattr(request.app.state, attr, None)
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB connection not available")
    return collection


def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


async def _get_user_doc(request: Request, username: str) -> Optional[dict]:
    collection = _get_collection(request, "mongo_user_collection")
    return await asyncio.to_thread(collection.find_one, {"username": username})


def _create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> tuple[str, datetime]:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": subject, "exp": expire}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token, expire


async def _store_token_doc(
    request: Request,
    token: str,
    username: str,
    expires_at: datetime,
    session_id: Optional[str],
) -> None:
    collection = _get_collection(request, "mongo_jwt_collection")
    token_doc = {
        "token": token,
        "username": username,
        "issued_at": datetime.utcnow(),
        "expires_at": expires_at,
        "revoked": False,
    }
    if session_id:
        token_doc["session_id"] = session_id
    await asyncio.to_thread(collection.insert_one, token_doc)


async def _ensure_default_user(request: Request) -> dict:
    existing = await _get_user_doc(request, DEFAULT_USER_USERNAME)
    if existing:
        return existing
    collection = _get_collection(request, "mongo_user_collection")
    user_doc = {
        "username": DEFAULT_USER_USERNAME,
        "hashed_password": _hash_password(DEFAULT_USER_PASSWORD),
        "created_at": datetime.utcnow(),
        "is_default": True,
    }
    await asyncio.to_thread(collection.insert_one, user_doc)
    return user_doc


async def register_user(request: Request, payload: UserCreate) -> UserPublic:
    existing = await _get_user_doc(request, payload.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    collection = _get_collection(request, "mongo_user_collection")
    user_doc = {
        "username": payload.username,
        "hashed_password": _hash_password(payload.password),
        "created_at": datetime.utcnow(),
    }
    await asyncio.to_thread(collection.insert_one, user_doc)
    return UserPublic(username=payload.username)


async def authenticate_and_create_token(
    request: Request,
    credentials: LoginRequest,
    session_id: Optional[str],
) -> Token:
    user_doc = await _get_user_doc(request, credentials.username)
    if not user_doc or not _verify_password(credentials.password, user_doc["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    token, expires_at = _create_access_token(credentials.username)
    await _store_token_doc(request, token, credentials.username, expires_at, session_id)
    return Token(access_token=token, expires_at=expires_at)


async def revoke_access_token(request: Request, token: str) -> None:
    collection = _get_collection(request, "mongo_jwt_collection")
    await asyncio.to_thread(
        collection.update_one,
        {"token": token},
        {"$set": {"revoked": True, "revoked_at": datetime.utcnow()}},
    )


async def get_current_token(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[str]:
    if token is None:
        return None
    token = token.strip()
    if not token or token.lower() in {"null", "undefined"}:
        return None
    return token


async def get_current_user(request: Request, token: Optional[str] = Depends(get_current_token)) -> CurrentUser:
    if not token:
        user_doc = await _ensure_default_user(request)
        return CurrentUser(username=user_doc["username"])
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise ValueError("Missing subject")
    except (JWTError, ValueError):
        user_doc = await _ensure_default_user(request)
        return CurrentUser(username=user_doc["username"])
    collection = _get_collection(request, "mongo_jwt_collection")
    token_doc = await asyncio.to_thread(collection.find_one, {"token": token})
    if not token_doc or token_doc.get("revoked"):
        user_doc = await _ensure_default_user(request)
        return CurrentUser(username=user_doc["username"])
    expires_at = token_doc.get("expires_at")
    if isinstance(expires_at, datetime) and expires_at < datetime.utcnow():
        await asyncio.to_thread(
            collection.update_one,
            {"_id": token_doc["_id"]},
            {"$set": {"revoked": True, "revoked_at": datetime.utcnow()}},
        )
        user_doc = await _ensure_default_user(request)
        return CurrentUser(username=user_doc["username"])
    user_doc = await _get_user_doc(request, username)
    if not user_doc:
        user_doc = await _ensure_default_user(request)
        return CurrentUser(username=user_doc["username"])
    session_id = token_doc.get("session_id")
    return CurrentUser(username=username, session_id=session_id)


__all__ = [
    "UserCreate",
    "LoginRequest",
    "UserPublic",
    "Token",
    "CurrentUser",
    "register_user",
    "authenticate_and_create_token",
    "revoke_access_token",
    "get_current_token",
    "get_current_user",
]
