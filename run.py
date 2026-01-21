"""
RAG ChatBot - Main Entry Point

Script khởi động all-in-one cho RAG ChatBot.

Usage:
    python run.py                  # Khởi động Streamlit (default)
    python run.py --mode streamlit # Khởi động Streamlit UI
    python run.py --mode api       # Khởi động API server
    python run.py --mode ingest    # Chạy ingest documents
    python run.py --check          # Kiểm tra hệ thống
"""

import os
import sys
import argparse
import subprocess

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def check_system():
    """Kiểm tra hệ thống trước khi chạy"""
    print("Checking system requirements...")
    print("-" * 50)
    
    issues = []
    
    # 1. Check Python version
    py_version = sys.version_info
    print(f"  Python: {py_version.major}.{py_version.minor}.{py_version.micro}", end=" ")
    if py_version >= (3, 9):
        print("✅")
    else:
        print("❌ (requires >= 3.9)")
        issues.append("Python >= 3.9 required")
    
    # 2. Check required packages
    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS (faiss-cpu)"),
        ("pydantic", "Pydantic"),
        ("requests", "Requests"),
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"  {name}: ✅")
        except ImportError:
            print(f"  {name}: ❌")
            issues.append(f"{name} not installed")
    
    # 3. Check directories
    from app.config import settings
    
    dirs_to_check = [
        (settings.DATA_DIR, "Data directory"),
    ]
    
    print()
    for dir_path, name in dirs_to_check:
        exists = os.path.exists(dir_path)
        print(f"  {name} ({dir_path}): {'✅' if exists else '⚠️ Not found'}")
    
    # 4. Check Qdrant connection
    print("\n🔍 Checking Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.QDRANT_URL)
        collection_info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
        print(f"  Qdrant: ✅ Connected ({collection_info.points_count} vectors)")
    except Exception as e:
        print(f"  Qdrant: ⚠️ Not connected - {str(e)}")
        issues.append("Qdrant not running or collection not created - run 'python -m app.ingest' first")
    
    # 5. Check MySQL connection
    print("\n🔍 Checking MySQL connection...")
    try:
        from app.database import db
        db.test_connection()
        print(f"  MySQL: ✅ Connected")
    except Exception as e:
        print(f"  MySQL: ⚠️ Not connected - {str(e)}")
        issues.append("MySQL not running or credentials incorrect")
    
    # 6. Check .env file
    env_file = os.path.join(PROJECT_ROOT, ".env")
    print(f"  .env file: {'✅' if os.path.exists(env_file) else '⚠️ Not found (using defaults)'}")
    
    # 7. Check API key
    print()
    if settings.OPENROUTER_API_KEY:
        print(f"  API Key: ✅ Set ({settings.OPENROUTER_API_KEY[:15]}...)")
    else:
        print("  API Key: ⚠️ Not set (check .env file)")
    
    # Summary
    print()
    print("-" * 50)
    if issues:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ All checks passed!")
        return True


def run_streamlit():
    """Khởi động Streamlit app"""
    print("🎨 Starting Streamlit app...")
    print("-" * 50)
    print("  URL: http://localhost:8501")
    print("  Qdrant: http://localhost:6333/dashboard")
    print("-" * 50)
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/streamlit_app.py",
        "--server.port=8501",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false"
    ])


def run_api():
    """Khởi động API server"""
    print("🚀 Starting API server...")
    print("-" * 50)
    
    from app.config import settings
    
    print(f"  Host: {settings.API_HOST}")
    print(f"  Port: {settings.API_PORT}")
    print(f"  Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("-" * 50)
    
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )
def run_ingest():
    """Chạy ingest documents"""
    print("Running document ingestion...")
    print("-" * 50)
    
    from app.ingest import build_index
    build_index()


def run_update():
    """Chạy incremental update"""
    print("Running incremental update...")
    print("-" * 50)
    
    subprocess.run([sys.executable, "scripts/update_index.py"])


def main():
    parser = argparse.ArgumentParser(
        description="RAG ChatBot - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                  # Start Streamlit UI (default)
  python run.py --mode streamlit # Start Streamlit UI
  python run.py --mode api       # Start API server
  python run.py --mode ingest    # Ingest documents
  python run.py --check          # System check
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["streamlit", "api", "ingest", "update"],
        default="streamlit",
        help="Run mode (default: streamlit)"
    )
    
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check system requirements"
    )
    
    parser.add_argument(
        "--host",
        default=None,
        help="API host (overrides config)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="API port (overrides config)"
    )
    
    args = parser.parse_args()
    
    
    # Override config if specified
    if args.host or args.port:
        from app.config import settings
        if args.host:
            settings.API_HOST = args.host
        if args.port:
            settings.API_PORT = args.port
    
    # System check
    if args.check:
        check_system()
        return
    
    # Run based on mode
    if args.mode == "streamlit":
        run_streamlit()
    
    elif args.mode == "api":
        # Quick check before starting
        if not check_system():
            print("\nFix issues above before starting the server")
            return
        print()
        run_api()
        
    elif args.mode == "ingest":
        run_ingest()
        
    elif args.mode == "update":
        run_update()


if __name__ == "__main__":
    main()
