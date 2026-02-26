from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings
from app.database import user_repo
from app.models import UserResponse, UserInDB

# PASSWORD HASHING
# Bcrypt context cho password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# JWT TOKEN
def create_access_token(user_id: int, email: str, role: str = "user") -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": str(user_id),  # Subject - user ID
        "email": email,
        "role": role,
        "exp": expire,  # Expiration time
        "iat": datetime.utcnow(),  # Issued at
    }
    
    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return token


def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        return None



# FASTAPI DEPENDENCIES

# Security scheme - Bearer token
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserResponse:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token không hợp lệ hoặc đã hết hạn",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Decode token
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        raise credentials_exception
    
    # Lấy user_id từ payload
    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    # Lấy user từ database
    user = user_repo.get_user_by_id(int(user_id))
    
    if user is None:
        raise credentials_exception
    
    # Kiểm tra user còn active không
    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tài khoản đã bị vô hiệu hóa",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Convert to UserResponse
    return UserResponse(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        is_active=user["is_active"],
        last_login_at=user.get("last_login_at"),
        created_at=user["created_at"]
    )


async def get_current_active_admin(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Yêu cầu quyền admin"
        )
    return current_user



# HELPER FUNCTIONS

def authenticate_user(email: str, password: str) -> Optional[dict]:
    user = user_repo.get_user_by_email(email)
    
    if user is None:
        return None
    
    if not user.get("is_active", False):
        return None
    
    if not verify_password(password, user["password_hash"]):
        return None
    
    return user
