"""
Authentication Module - JWT-based Authentication

Features:
    - Password hashing với bcrypt
    - JWT token creation và validation
    - FastAPI dependency để protect endpoints
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings
from app.database import user_repo
from app.models import UserResponse, UserInDB


# ================================================================
# PASSWORD HASHING
# ================================================================

# Bcrypt context cho password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash password sử dụng bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password với hash đã lưu
    
    Args:
        plain_password: Password người dùng nhập
        hashed_password: Hash đã lưu trong database
        
    Returns:
        True nếu password đúng
    """
    return pwd_context.verify(plain_password, hashed_password)


# ================================================================
# JWT TOKEN
# ================================================================

def create_access_token(user_id: int, email: str, role: str = "user") -> str:
    """
    Tạo JWT access token
    
    Args:
        user_id: ID của user
        email: Email của user
        role: Role của user (user/admin)
        
    Returns:
        JWT token string
    """
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
    """
    Decode và validate JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Payload dict nếu valid, None nếu invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        return None


# ================================================================
# FASTAPI DEPENDENCIES
# ================================================================

# Security scheme - Bearer token
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserResponse:
    """
    FastAPI dependency để lấy current user từ JWT token
    
    Sử dụng:
        @app.get("/protected")
        async def protected_route(current_user: UserResponse = Depends(get_current_user)):
            return {"user": current_user}
    
    Raises:
        HTTPException 401 nếu token invalid hoặc user không tồn tại
    """
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
    """
    FastAPI dependency để yêu cầu admin role
    
    Sử dụng:
        @app.get("/admin-only")
        async def admin_route(admin: UserResponse = Depends(get_current_active_admin)):
            return {"admin": admin}
    
    Raises:
        HTTPException 403 nếu user không phải admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Yêu cầu quyền admin"
        )
    return current_user


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def authenticate_user(email: str, password: str) -> Optional[dict]:
    """
    Xác thực user bằng email và password
    
    Args:
        email: Email đăng nhập
        password: Password
        
    Returns:
        User dict nếu xác thực thành công, None nếu thất bại
    """
    user = user_repo.get_user_by_email(email)
    
    if user is None:
        return None
    
    if not user.get("is_active", False):
        return None
    
    if not verify_password(password, user["password_hash"]):
        return None
    
    return user
