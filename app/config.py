from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # ==================== PATHS ====================
    DATA_DIR: str = "data/legal_kb"
    LOG_FILE: str = "logs/chat_logs.jsonl"
    
    # ==================== QDRANT SETTINGS ====================
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "abc_corp_docs"
    
    # ==================== POSTGRESQL SETTINGS (from .env) ====================
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DATABASE: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    
    # ==================== RAG PARAMETERS ====================
    # Số lượng documents retrieve
    TOP_K: int = 5
    
    # Ngưỡng similarity (0-1).
    SIMILARITY_THRESHOLD: float = 0.25
    
    # Chunking parameters 
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    
    # ==================== EMBEDDING MODEL ====================
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024 
    
    # ==================== LLM SETTINGS ====================
    OPENROUTER_API_KEY: Optional[str] = None
    # Model name 
    MODEL_NAME: str = "z-ai/glm-4.5-air:free"
    
    # Generation parameters
    TEMPERATURE: float = 0.2 
    MAX_TOKENS: int = 500
    
    # API endpoint
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # ==================== RERANKER SETTINGS ====================
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_CANDIDATES: int = 15
    RERANKER_TOP_K: int = 5
    # Threshold trước rerank 
    PRE_RERANK_THRESHOLD: float = 0.1
    # Threshold sau rerank
    POST_RERANK_THRESHOLD: float = 0.0

    # ==================== CONVERSATION MEMORY ====================
    MAX_HISTORY_TURNS: int = 5
    
    # ==================== API SETTINGS ====================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # CORS settings
    CORS_ORIGINS: list = ["*"]  
    
    # ==================== JWT / AUTH SETTINGS ====================
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    
    # Thời gian hết hạn token (phút)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 giờ
    
    # ==================== LOGGING ====================
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # ==================== EVALUATION ====================
    GOLDEN_SET_PATH: str = "eval/golden_set.json"
    EVAL_RESULTS_PATH: str = "eval/results.json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
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
    ensure_directories()
    print_config()