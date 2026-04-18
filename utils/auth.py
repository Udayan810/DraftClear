import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config.settings import DATA_DIR
import logging

logger = logging.getLogger(__name__)

# Security config
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "draftclear_super_secret_prototype_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

USERS_FILE = DATA_DIR / "users.json"


def init_users_file():
    """Ensure the users database file exists"""
    if not USERS_FILE.exists():
        logger.info("Initializing users.json database")
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Create default admin
        default_admin = {
            "email": "admin@draftclear.com",
            "password": get_password_hash("admin123"),
            "role": "admin",
            "created_at": datetime.utcnow().isoformat()
        }
        with open(USERS_FILE, "w") as f:
            json.dump({"admin@draftclear.com": default_admin}, f, indent=4)


def get_users() -> Dict[str, Any]:
    """Load users from file"""
    if not USERS_FILE.exists():
        init_users_file()
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading users file: {e}")
        return {}


def save_users(users: Dict[str, Any]):
    """Save users to file"""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_user(email: str) -> Optional[Dict[str, Any]]:
    users = get_users()
    return users.get(email)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = get_user(email)
    if user is None:
        raise credentials_exception
    return user
