from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """
    Configuration Management cho RAG System
    Load từ .env file hoặc environment variables
    """
    
    # ==================== PATHS ====================
    DATA_DIR: str = "data/legal_kb"
    LOG_FILE: str = "logs/chat_logs.jsonl"
    
    # ==================== QDRANT SETTINGS ====================
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "abc_corp_docs"
    
    # ==================== MYSQL SETTINGS ====================
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str = "rag_chatbot"
    MYSQL_USER: str = "root" 
    MYSQL_PASSWORD: Optional[str] = None 
    
    # ==================== RAG PARAMETERS ====================
    # Số lượng documents retrieve (tăng để lấy nhiều context hơn)
    TOP_K: int = 5
    
    # Ngưỡng similarity (0-1). Dưới ngưỡng này = không liên quan
    SIMILARITY_THRESHOLD: float = 0.25
    
    # Chunking parameters (tăng để có context đầy đủ hơn)
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    
    # ==================== EMBEDDING MODEL ====================
    # BGE-M3: Multilingual model tốt nhất cho Tiếng Việt (BAAI)
    # Hỗ trợ 100+ ngôn ngữ, SOTA performance
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024  # BGE-M3 có 1024 dimensions
    
    # ==================== LLM SETTINGS ====================
    # API Key (load từ .env)
    OPENROUTER_API_KEY: Optional[str] = None
    
    # Model name (free tier)
    MODEL_NAME: str = "mistralai/devstral-2512:free"
    
    # Generation parameters
    TEMPERATURE: float = 0.2  # 0 = deterministic, 1 = creative
    MAX_TOKENS: int = 500
    
    # API endpoint
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # ==================== CONVERSATION MEMORY ====================
    # Số lượng lượt hội thoại giữ lại (1 lượt = user + assistant)
    MAX_HISTORY_TURNS: int = 5
    
    # ==================== API SETTINGS ====================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # CORS settings
    CORS_ORIGINS: list = ["*"]  # Production nên chỉ định cụ thể
    
    # ==================== JWT / AUTH SETTINGS ====================
    # # Secret key cho JWT

    JWT_SECRET_KEY: Optional[str] = None
    # Algorithm mã hóa JWT
    JWT_ALGORITHM: str = "HS256"
    
    # Thời gian hết hạn token (phút)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 giờ
    
    # ==================== LOGGING ====================
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # ==================== EVALUATION ====================
    GOLDEN_SET_PATH: str = "eval/golden_set.json"
    EVAL_RESULTS_PATH: str = "eval/results.json"
    
    class Config:
        # Load từ file .env trong root directory
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        # Extra fields sẽ bị ignore (không raise error)
        extra = "ignore"

# ==================== SINGLETON INSTANCE ====================
settings = Settings()

# ==================== HELPER FUNCTIONS ====================
def ensure_directories():
    """Tạo các thư mục cần thiết nếu chưa tồn tại"""
    directories = [
        settings.DATA_DIR,
        os.path.dirname(settings.LOG_FILE),
        os.path.dirname(settings.GOLDEN_SET_PATH),
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

def print_config():
    """In ra cấu hình hiện tại (để debug)"""
    print("=" * 60)
    print("CURRENT CONFIGURATION")
    print("=" * 60)
    
    print("\n PATHS:")
    print(f"  Data Directory: {settings.DATA_DIR}")
    print(f"  Log File: {settings.LOG_FILE}")
    
    print("\n QDRANT:")
    print(f"  URL: {settings.QDRANT_URL}")
    print(f"  Collection: {settings.QDRANT_COLLECTION_NAME}")
    
    print("\n RAG PARAMETERS:")
    print(f"  Top K: {settings.TOP_K}")
    print(f"  Similarity Threshold: {settings.SIMILARITY_THRESHOLD}")
    print(f"  Chunk Size: {settings.CHUNK_SIZE}")
    print(f"  Chunk Overlap: {settings.CHUNK_OVERLAP}")
    
    print("\n LLM SETTINGS:")
    print(f"  Model: {settings.MODEL_NAME}")
    print(f"  Temperature: {settings.TEMPERATURE}")
    print(f"  Max Tokens: {settings.MAX_TOKENS}")
    api_key_status = " SET" if settings.OPENROUTER_API_KEY else "❌ NOT SET"
    print(f"  API Key: {api_key_status}")
    
    print("\n MEMORY:")
    print(f"  Max History Turns: {settings.MAX_HISTORY_TURNS}")
    
    print("\n API:")
    print(f"  Host: {settings.API_HOST}")
    print(f"  Port: {settings.API_PORT}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Test configuration
    ensure_directories()
    print_config()