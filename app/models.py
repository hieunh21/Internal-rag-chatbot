"""
Pydantic Models cho RAG ChatBot API
Định nghĩa cấu trúc dữ liệu cho Request, Response, Logging
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


# ================================================================
# API REQUEST MODELS
# ================================================================

class ChatRequest(BaseModel):
    """
    Request body cho endpoint /chat
    
    Attributes:
        question: Câu hỏi của user (1-500 ký tự)
        session_id: ID để tracking conversation (default: "default")
    """
    question: str = Field(
        ...,  # Required field
        min_length=1,
        max_length=500,
        description="Câu hỏi của người dùng"
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        description="Session ID để tracking conversation"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Thời gian thử việc tối đa là bao lâu?",
                "session_id": "user_123"
            }
        }


# ================================================================
# SOURCE / CITATION MODELS
# ================================================================

class Source(BaseModel):
    """
    Một nguồn trích dẫn (citation)
    
    Attributes:
        source: Tên file nguồn
        chunk_id: ID của chunk trong file
        score: Điểm similarity (0-1)
        excerpt: Đoạn trích ngắn (preview)
        full_content: Nội dung đầy đủ của chunk
        page: Số trang trong PDF (nếu có)
    """
    source: str = Field(
        ...,
        description="Tên file tài liệu nguồn"
    )
    chunk_id: int = Field(
        ...,
        ge=0,
        description="ID của chunk"
    )
    score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Điểm tương đồng (0-1)"
    )
    excerpt: str = Field(
        default="",
        max_length=200,
        description="Đoạn trích ngắn (preview)"
    )
    full_content: str = Field(
        default="",
        description="Nội dung đầy đủ của chunk"
    )
    page: Optional[int] = Field(
        default=None,
        description="Số trang trong PDF"
    )
    
    def get_display_name(self) -> str:
        """Tên hiển thị đẹp hơn"""
        name = self.source.replace("_", " ").replace(".pdf", "").replace(".md", "")
        return name
    
    def get_location_info(self) -> str:
        """Thông tin vị trí"""
        parts = []
        if self.page:
            parts.append(f"Trang {self.page}")
        parts.append(f"Chunk #{self.chunk_id}")
        return " • ".join(parts)
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "01_lao_dong_hop_dong.md",
                "chunk_id": 0,
                "score": 0.85,
                "excerpt": "Thời gian thử việc tối đa 60 ngày đối với..."
            }
        }


# ================================================================
# METADATA MODELS
# ================================================================

class Metadata(BaseModel):
    """
    Metadata về quá trình xử lý request
    
    Attributes:
        model: Tên model LLM được sử dụng
        latency_ms: Thời gian xử lý (milliseconds)
        top_k: Số lượng documents retrieved
        sources_count: Số nguồn được sử dụng trong câu trả lời
        timestamp: Thời điểm xử lý
    """
    model: str = Field(
        ...,
        description="Tên model LLM"
    )
    latency_ms: float = Field(
        ...,
        ge=0,
        description="Thời gian xử lý (ms)"
    )
    top_k: int = Field(
        ...,
        ge=1,
        description="Số documents retrieved"
    )
    sources_count: int = Field(
        ...,
        ge=0,
        description="Số nguồn sử dụng"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Thời điểm xử lý"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "mistralai/mistral-7b-instruct:free",
                "latency_ms": 1234.56,
                "top_k": 3,
                "sources_count": 2,
                "timestamp": "2025-01-05T10:30:00"
            }
        }


# ================================================================
# API RESPONSE MODELS
# ================================================================

class ChatResponse(BaseModel):
    """
    Response body cho endpoint /chat
    
    Attributes:
        answer: Câu trả lời từ chatbot
        sources: Danh sách nguồn trích dẫn
        meta: Metadata về quá trình xử lý
        session_id: Session ID của conversation
        is_grounded: True nếu câu trả lời dựa trên tài liệu
    """
    answer: str = Field(
        ...,
        description="Câu trả lời từ chatbot"
    )
    sources: List[Source] = Field(
        default=[],
        description="Danh sách nguồn trích dẫn"
    )
    meta: Metadata = Field(
        ...,
        description="Metadata xử lý"
    )
    session_id: str = Field(
        ...,
        description="Session ID"
    )
    is_grounded: bool = Field(
        default=True,
        description="True nếu trả lời dựa trên tài liệu"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Thời gian thử việc tối đa là 60 ngày theo quy định của công ty.",
                "sources": [
                    {
                        "source": "01_lao_dong_hop_dong.md",
                        "chunk_id": 0,
                        "score": 0.85,
                        "excerpt": "Thời gian thử việc tối đa 60 ngày..."
                    }
                ],
                "meta": {
                    "model": "mistralai/mistral-7b-instruct:free",
                    "latency_ms": 1234.56,
                    "top_k": 3,
                    "sources_count": 1,
                    "timestamp": "2025-01-05T10:30:00"
                },
                "session_id": "user_123",
                "is_grounded": True
            }
        }


# ================================================================
# CONVERSATION MEMORY MODELS
# ================================================================

class Message(BaseModel):
    """
    Một message trong conversation history
    
    Attributes:
        role: "user" hoặc "assistant"
        content: Nội dung message
        timestamp: Thời điểm gửi
        sources: Danh sách nguồn trích dẫn (cho assistant)
        latency: Thời gian xử lý (ms)
        is_grounded: Có nguồn hỗ trợ không
    """
    role: str = Field(
        ...,
        pattern="^(user|assistant)$",
        description="Role: 'user' hoặc 'assistant'"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Nội dung message"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Thời điểm gửi"
    )
    # Optional fields for assistant messages
    sources: Optional[List[dict]] = Field(
        default=None,
        description="Danh sách nguồn trích dẫn"
    )
    latency: Optional[float] = Field(
        default=None,
        description="Thời gian xử lý (ms)"
    )
    is_grounded: Optional[bool] = Field(
        default=None,
        description="Có nguồn hỗ trợ không"
    )


class Conversation(BaseModel):
    """
    Một conversation session
    
    Attributes:
        session_id: ID của session
        messages: Danh sách messages
        created_at: Thời điểm tạo session
        updated_at: Thời điểm cập nhật cuối
    """
    session_id: str = Field(
        ...,
        description="Session ID"
    )
    messages: List[Message] = Field(
        default=[],
        description="Lịch sử messages"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Thời điểm tạo"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Thời điểm cập nhật cuối"
    )


# ================================================================
# LOGGING MODELS
# ================================================================

class ChatLog(BaseModel):
    """
    Log entry cho mỗi request
    Lưu vào file logs/chat_logs.jsonl
    
    Attributes:
        session_id: Session ID
        question: Câu hỏi của user
        answer: Câu trả lời
        sources: Nguồn trích dẫn
        latency_ms: Thời gian xử lý
        is_grounded: Có dựa trên tài liệu không
        timestamp: Thời điểm
        error: Thông tin lỗi (nếu có)
    """
    session_id: str
    question: str
    answer: str
    sources: List[Source] = []
    latency_ms: float
    is_grounded: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_123",
                "question": "Thời gian thử việc?",
                "answer": "Thời gian thử việc tối đa 60 ngày.",
                "sources": [],
                "latency_ms": 1234.56,
                "is_grounded": True,
                "timestamp": "2025-01-05T10:30:00",
                "error": None
            }
        }


# ================================================================
# HEALTH CHECK MODELS
# ================================================================

class HealthResponse(BaseModel):
    """Response cho endpoint /health"""
    status: str = Field(
        default="healthy",
        description="Trạng thái hệ thống"
    )
    qdrant_connected: bool = Field(
        ...,
        description="Qdrant có kết nối không"
    )
    db_connected: bool = Field(
        ...,
        description="MySQL có kết nối không"
    )
    embedding_model: str = Field(
        ...,
        description="Tên embedding model"
    )
    vectors_count: int = Field(
        default=0,
        ge=0,
        description="Số vectors trong Qdrant"
    )


class StatsResponse(BaseModel):
    """Response cho endpoint /stats"""
    total_queries: int = Field(
        ...,
        ge=0,
        description="Tổng số queries"
    )
    grounded_queries: int = Field(
        ...,
        ge=0,
        description="Số queries có nguồn"
    )
    grounded_rate: float = Field(
        ...,
        ge=0,
        le=100,
        description="Tỷ lệ grounded (%)"
    )
    avg_latency_ms: float = Field(
        ...,
        ge=0,
        description="Latency trung bình (ms)"
    )
    active_sessions: int = Field(
        ...,
        ge=0,
        description="Số sessions active"
    )


# ================================================================
# ERROR MODELS
# ================================================================

class ErrorResponse(BaseModel):
    """Response khi có lỗi"""
    error: str = Field(
        ...,
        description="Mô tả lỗi"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Chi tiết lỗi"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Thời điểm xảy ra lỗi"
    )


# ================================================================
# USER / AUTH MODELS
# ================================================================

class UserCreate(BaseModel):
    """Request body cho đăng ký user mới"""
    email: EmailStr = Field(
        ...,
        description="Email đăng nhập"
    )
    password: str = Field(
        ...,
        min_length=6,
        max_length=100,
        description="Mật khẩu (tối thiểu 6 ký tự)"
    )
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Họ tên đầy đủ"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "nguyen.van.a@abccorp.vn",
                "password": "securepassword123",
                "full_name": "Nguyễn Văn A"
            }
        }


class UserLogin(BaseModel):
    """Request body cho đăng nhập"""
    email: EmailStr = Field(
        ...,
        description="Email đăng nhập"
    )
    password: str = Field(
        ...,
        description="Mật khẩu"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "nguyen.van.a@abccorp.vn",
                "password": "securepassword123"
            }
        }


class UserResponse(BaseModel):
    """Response chứa thông tin user (không có password)"""
    id: int = Field(..., description="User ID")
    email: str = Field(..., description="Email")
    full_name: str = Field(..., description="Họ tên")
    role: str = Field(default="user", description="Role: user/admin")
    is_active: bool = Field(default=True, description="Trạng thái tài khoản")
    last_login_at: Optional[datetime] = Field(default=None, description="Lần đăng nhập cuối")
    created_at: datetime = Field(..., description="Ngày tạo tài khoản")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "email": "nguyen.van.a@abccorp.vn",
                "full_name": "Nguyễn Văn A",
                "role": "user",
                "is_active": True,
                "last_login_at": "2025-01-19T10:30:00",
                "created_at": "2025-01-01T08:00:00"
            }
        }


class UserInDB(UserResponse):
    """User model với password hash (dùng nội bộ)"""
    password_hash: str = Field(..., description="Password đã hash")


class TokenResponse(BaseModel):
    """Response chứa JWT token sau khi login"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Loại token")
    user: UserResponse = Field(..., description="Thông tin user")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "user": {
                    "id": 1,
                    "email": "nguyen.van.a@abccorp.vn",
                    "full_name": "Nguyễn Văn A",
                    "role": "user",
                    "is_active": True,
                    "created_at": "2025-01-01T08:00:00"
                }
            }
        }


class PasswordChange(BaseModel):
    """Request body cho đổi mật khẩu"""
    current_password: str = Field(..., description="Mật khẩu hiện tại")
    new_password: str = Field(
        ...,
        min_length=6,
        max_length=100,
        description="Mật khẩu mới (tối thiểu 6 ký tự)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "current_password": "oldpassword123",
                "new_password": "newpassword456"
            }
        }


class UserUpdate(BaseModel):
    """Request body cho cập nhật thông tin user"""
    full_name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=100,
        description="Họ tên mới"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "full_name": "Nguyễn Văn B"
            }
        }

