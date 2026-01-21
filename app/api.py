"""
FastAPI Application - API Layer cho RAG ChatBot

Endpoints:
    - GET  /health     : Health check
    - POST /chat       : Main chat endpoint
    - GET  /stats      : Thống kê hệ thống
    - DELETE /session/{session_id} : Xóa session
    - GET  /sessions   : Liệt kê sessions
    
Auth Endpoints:
    - POST /auth/register : Đăng ký tài khoản
    - POST /auth/login    : Đăng nhập
    - GET  /auth/me       : Lấy thông tin user
    - PUT  /auth/me       : Cập nhật profile
    - POST /auth/change-password : Đổi mật khẩu
"""

import os
import json
import traceback
from datetime import datetime
from typing import Optional
from app.database import db, user_repo

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import (
    ChatRequest,
    ChatResponse,
    Source,
    Metadata,
    ChatLog,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
    # Auth models
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
    TokenResponse,
    PasswordChange
)
from app.rag_engine import rag_engine
from app.memory import memory
from app.config import settings, ensure_directories
from app.auth import (
    hash_password,
    verify_password,
    create_access_token,
    authenticate_user,
    get_current_user,
    get_current_active_admin
)


# ================================================================
# INITIALIZE APP
# ================================================================

app = FastAPI(
    title="RAG ChatBot API",
    description="""
    Hệ thống chatbot thông minh trả lời câu hỏi về quy định, chính sách công ty.
    """,
    version="2.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc
)

# ================================================================
# MIDDLEWARE
# ================================================================

# CORS - cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # ["*"] trong development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# STARTUP EVENT
# ================================================================

@app.on_event("startup")
async def startup_event():
    """Chạy khi server khởi động"""
    print("=" * 60)
    print("Starting RAG ChatBot API...")
    print("=" * 60)
    
    # Tạo thư mục cần thiết
    ensure_directories()
    
    # Verify RAG engine đã sẵn sàng
    health = rag_engine.health_check()
    print(f"  ✓ RAG Engine: {'Ready' if health.get('db_connected', False) else 'Not Ready'}")
    print(f"  ✓ Qdrant: {health.get('vectors_count', 0)} vectors")
    print(f"  ✓ API: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"  ✓ Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("=" * 60)

# ================================================================
# LOGGING HELPER
# ================================================================

def log_chat(
    session_id: str,
    question: str,
    answer: str,
    sources: list,
    latency_ms: float,
    is_grounded: bool,
    error: str = None
) -> None:
    """Ghi log vào file JSONL"""
    try:
        log_entry = ChatLog(
            session_id=session_id,
            question=question,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            is_grounded=is_grounded,
            timestamp=datetime.now(),
            error=error
        )
        
        # Đảm bảo thư mục logs tồn tại
        log_dir = os.path.dirname(settings.LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Append vào file JSONL
        with open(settings.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry.model_dump_json() + "\n")
            
    except Exception as e:
        print(f"Failed to write log: {e}")

# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API info"""
    return {
        "name": "RAG ChatBot API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint
    
    Kiểm tra trạng thái của:
    - Vector Database
    - Memory system
    - Overall status
    """
    rag_health = rag_engine.health_check()
    
    return HealthResponse(
        status="healthy" if rag_health.get("db_connected", False) else "unhealthy",
        qdrant_connected=rag_health.get("db_connected", False),
        db_connected=True,  # MySQL connection (vì đã khởi tạo thành công)
        embedding_model=rag_health.get("embedding_model", "unknown"),
        vectors_count=rag_health.get("vectors_count", 0)
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Main chat endpoint (Yêu cầu đăng nhập)
    
    Xử lý câu hỏi của user và trả về:
    - Câu trả lời dựa trên tài liệu
    - Nguồn trích dẫn (citations)
    - Metadata (latency, model, etc.)
    
    **Yêu cầu:** Bearer token trong header
    
    **Parameters:**
    - `question`: Câu hỏi của user (1-500 ký tự)
    - `session_id`: ID để tracking conversation (optional)
    
    **Response:**
    - `answer`: Câu trả lời
    - `sources`: Danh sách nguồn trích dẫn
    - `meta`: Thông tin xử lý
    - `is_grounded`: True nếu trả lời dựa trên tài liệu
    """
    try:
        # Tạo session_id gắn với user nếu không có
        session_id = request.session_id
        if session_id == "default":
            session_id = f"user_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 1. Lấy conversation history
        history = memory.get_history(session_id)
        
        # 2. Gọi RAG Engine
        answer, sources, is_grounded, latency_ms = rag_engine.ask(
            question=request.question,
            history=history
        )
        
        # 3. Cập nhật memory (lưu vào MySQL)
        memory.add_message(session_id, "user", request.question)
        memory.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=[s.model_dump() for s in sources] if sources else None,
            latency=latency_ms,
            is_grounded=is_grounded
        )
        
        # 4. Build response
        response = ChatResponse(
            answer=answer,
            sources=sources,
            meta=Metadata(
                model=settings.MODEL_NAME,
                latency_ms=round(latency_ms, 2),
                top_k=settings.TOP_K,
                sources_count=len(sources),
                timestamp=datetime.now()
            ),
            session_id=session_id,
            is_grounded=is_grounded
        )
        
        # 5. Log request - đã tắt (dữ liệu đã lưu trong MySQL)
        # log_chat(
        #     session_id=session_id,
        #     question=request.question,
        #     answer=answer,
        #     sources=sources,
        #     latency_ms=latency_ms,
        #     is_grounded=is_grounded
        # )
        
        return response
        
    except Exception as e:
        # Log error - đã tắt
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        # log_chat(
        #     session_id=session_id,
        #     question=request.question,
        #     answer="",
        #     sources=[],
        #     latency_ms=0,
        #     is_grounded=False,
        #     error=error_msg
        # )
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats(current_user: UserResponse = Depends(get_current_active_admin)):
    """
    Thống kê hệ thống (Chỉ Admin)
    
    **Yêu cầu:** Bearer token với role=admin
    
    Trả về:
    - Tổng số queries
    - Tỷ lệ grounded
    - Latency trung bình
    - Số sessions active
    """
    # Đọc log file
    if not os.path.exists(settings.LOG_FILE):
        return StatsResponse(
            total_queries=0,
            grounded_queries=0,
            grounded_rate=0.0,
            avg_latency_ms=0.0,
            active_sessions=memory.get_active_sessions()
        )
    
    try:
        with open(settings.LOG_FILE, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")
    
    if not logs:
        return StatsResponse(
            total_queries=0,
            grounded_queries=0,
            grounded_rate=0.0,
            avg_latency_ms=0.0,
            active_sessions=memory.get_active_sessions()
        )
    
    # Tính toán stats
    total = len(logs)
    grounded = sum(1 for log in logs if log.get("is_grounded", False))
    avg_latency = sum(log.get("latency_ms", 0) for log in logs) / total
    
    return StatsResponse(
        total_queries=total,
        grounded_queries=grounded,
        grounded_rate=round(grounded / total * 100, 2),
        avg_latency_ms=round(avg_latency, 2),
        active_sessions=memory.get_active_sessions()
    )


@app.delete("/session/{session_id}", tags=["Session"])
async def clear_session(
    session_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Xóa conversation history của một session (Yêu cầu đăng nhập)
    
    **Yêu cầu:** Bearer token trong header
    
    **Parameters:**
    - `session_id`: ID của session cần xóa
    """
    # Kiểm tra session có thuộc về user không (dựa trên prefix user_id)
    if not session_id.startswith(f"user_{current_user.id}_") and current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Bạn không có quyền xóa session này"
        )
    
    if memory.clear_session(session_id):
        return {
            "message": f"Session '{session_id}' cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )


@app.get("/sessions", tags=["Session"])
async def list_sessions(current_user: UserResponse = Depends(get_current_user)):
    """
    Liệt kê sessions của user hiện tại (Yêu cầu đăng nhập)
    
    **Yêu cầu:** Bearer token trong header
    
    Admin có thể xem tất cả sessions.
    """
    all_sessions = memory.get_session_summaries()
    
    # Filter sessions theo user (dựa trên prefix user_id trong session_id)
    if current_user.role == "admin":
        # Admin xem tất cả
        user_sessions = all_sessions
    else:
        # User thường chỉ xem sessions của mình
        user_sessions = [
            s for s in all_sessions 
            if s.get("session_id", "").startswith(f"user_{current_user.id}_")
        ]
    
    return {
        "total_sessions": len(user_sessions),
        "sessions": user_sessions
    }


@app.get("/session/{session_id}/history", tags=["Session"])
async def get_session_history(
    session_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Lấy lịch sử hội thoại của một session (Yêu cầu đăng nhập)
    
    **Yêu cầu:** Bearer token trong header
    """
    # Kiểm tra quyền truy cập
    if not session_id.startswith(f"user_{current_user.id}_") and current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Bạn không có quyền xem session này"
        )
    
    if not memory.session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )
    
    messages = memory.get_messages(session_id)
    
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "sources": msg.sources if msg.sources else [],
                "latency": msg.latency,
                "is_grounded": msg.is_grounded
            }
            for msg in messages
        ]
    }


@app.post("/reload-index", tags=["System"])
async def reload_index(current_user: UserResponse = Depends(get_current_active_admin)):
    """
    Reload Qdrant index (sau khi update) - Chỉ Admin
    
    **Yêu cầu:** Bearer token với role=admin
    """
    try:
        rag_engine.reload_db()
        return {
            "message": "Index reloaded successfully",
            "reloaded_by": current_user.email,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# AUTH ENDPOINTS
# ================================================================

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(user_data: UserCreate):
    """
    Đăng ký tài khoản mới
    
    **Parameters:**
    - `email`: Email đăng nhập (unique)
    - `password`: Mật khẩu (tối thiểu 6 ký tự)
    - `full_name`: Họ tên đầy đủ
    
    **Returns:**
    - JWT access token
    - Thông tin user
    """
    # Kiểm tra email đã tồn tại
    if user_repo.email_exists(user_data.email):
        raise HTTPException(
            status_code=400,
            detail="Email đã được sử dụng"
        )
    
    # Hash password và tạo user
    hashed_password = hash_password(user_data.password)
    user_id = user_repo.create_user(
        email=user_data.email,
        password_hash=hashed_password,
        full_name=user_data.full_name,
        role="user"
    )
    
    if user_id is None:
        raise HTTPException(
            status_code=500,
            detail="Không thể tạo tài khoản"
        )
    
    # Lấy user vừa tạo
    user = user_repo.get_user_by_id(user_id)
    
    # Tạo access token
    access_token = create_access_token(
        user_id=user["id"],
        email=user["email"],
        role=user["role"]
    )
    
    # Cập nhật last_login
    user_repo.update_last_login(user_id)
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            full_name=user["full_name"],
            role=user["role"],
            is_active=user["is_active"],
            last_login_at=user.get("last_login_at"),
            created_at=user["created_at"]
        )
    )


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(credentials: UserLogin):
    """
    Đăng nhập và lấy JWT token
    
    **Parameters:**
    - `email`: Email đăng nhập
    - `password`: Mật khẩu
    
    **Returns:**
    - JWT access token (có hiệu lực 24 giờ)
    - Thông tin user
    """
    # Xác thực user
    user = authenticate_user(credentials.email, credentials.password)
    
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Email hoặc mật khẩu không đúng"
        )
    
    # Tạo access token
    access_token = create_access_token(
        user_id=user["id"],
        email=user["email"],
        role=user["role"]
    )
    
    # Cập nhật last_login
    user_repo.update_last_login(user["id"])
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            full_name=user["full_name"],
            role=user["role"],
            is_active=user["is_active"],
            last_login_at=user.get("last_login_at"),
            created_at=user["created_at"]
        )
    )


@app.get("/auth/me", response_model=UserResponse, tags=["Auth"])
async def get_me(current_user: UserResponse = Depends(get_current_user)):
    """
    Lấy thông tin user hiện tại
    
    **Yêu cầu:** Bearer token trong header
    
    **Returns:**
    - Thông tin user đã đăng nhập
    """
    return current_user


@app.put("/auth/me", response_model=UserResponse, tags=["Auth"])
async def update_me(
    update_data: UserUpdate,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Cập nhật thông tin profile
    
    **Yêu cầu:** Bearer token trong header
    
    **Parameters:**
    - `full_name`: Họ tên mới (optional)
    
    **Returns:**
    - Thông tin user đã cập nhật
    """
    # Chuẩn bị dữ liệu cập nhật
    update_fields = {}
    if update_data.full_name is not None:
        update_fields["full_name"] = update_data.full_name
    
    if not update_fields:
        raise HTTPException(
            status_code=400,
            detail="Không có thông tin nào để cập nhật"
        )
    
    # Cập nhật user
    success = user_repo.update_user(current_user.id, **update_fields)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Không thể cập nhật thông tin"
        )
    
    # Lấy user đã cập nhật
    updated_user = user_repo.get_user_by_id(current_user.id)
    
    return UserResponse(
        id=updated_user["id"],
        email=updated_user["email"],
        full_name=updated_user["full_name"],
        role=updated_user["role"],
        is_active=updated_user["is_active"],
        last_login_at=updated_user.get("last_login_at"),
        created_at=updated_user["created_at"]
    )


@app.post("/auth/change-password", tags=["Auth"])
async def change_password(
    password_data: PasswordChange,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Đổi mật khẩu
    
    **Yêu cầu:** Bearer token trong header
    
    **Parameters:**
    - `current_password`: Mật khẩu hiện tại
    - `new_password`: Mật khẩu mới (tối thiểu 6 ký tự)
    
    **Returns:**
    - Thông báo thành công
    """
    # Lấy user từ database (cần password_hash)
    user = user_repo.get_user_by_id(current_user.id)
    
    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User không tồn tại"
        )
    
    # Verify mật khẩu hiện tại
    if not verify_password(password_data.current_password, user["password_hash"]):
        raise HTTPException(
            status_code=400,
            detail="Mật khẩu hiện tại không đúng"
        )
    
    # Hash mật khẩu mới
    new_password_hash = hash_password(password_data.new_password)
    
    # Cập nhật password
    success = user_repo.update_user(current_user.id, password_hash=new_password_hash)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Không thể đổi mật khẩu"
        )
    
    return {
        "message": "Đổi mật khẩu thành công",
        "timestamp": datetime.now().isoformat()
    }


# ================================================================
# ERROR HANDLERS
# ================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail
        ).model_dump(mode="json")
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump(mode="json")
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,  # Auto-reload khi code thay đổi
        log_level="info"
    )