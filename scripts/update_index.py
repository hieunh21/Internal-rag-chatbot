"""
Incremental Index Update Script

Features:
    - Detect files mới/thay đổi/xóa
    - Chỉ update phần cần thiết (không rebuild toàn bộ)
    - Backup index cũ trước khi update
    - Rollback nếu có lỗi
    
Usage:
    python scripts/update_index.py              # Check và update
    python scripts/update_index.py --force      # Force rebuild toàn bộ
    python scripts/update_index.py --dry-run    # Chỉ check, không update
"""

import os
import sys
import json
import hashlib
import shutil
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple

# Thêm parent directory vào path để import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.ingest import LocalEmbedding
from app.config import settings


# ================================================================
# CONSTANTS
# ================================================================

MANIFEST_FILE = os.path.join(settings.DB_DIR, "manifest.json")
BACKUP_DIR = os.path.join(settings.DB_DIR, "backup")


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def calculate_file_hash(filepath: str) -> str:
    """Tính MD5 hash của file để detect thay đổi"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_current_files(data_dir: str) -> Dict[str, str]:
    """
    Scan thư mục và trả về dict {filename: hash}
    """
    files = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(data_dir, filename)
            files[filename] = calculate_file_hash(filepath)
    return files


def load_manifest() -> Dict:
    """Load manifest từ file"""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}, "last_update": None, "version": 1}


def save_manifest(manifest: Dict) -> None:
    """Lưu manifest ra file"""
    manifest["last_update"] = datetime.now().isoformat()
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def detect_changes(
    current_files: Dict[str, str],
    manifest: Dict
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    So sánh files hiện tại với manifest
    
    Returns:
        (new_files, modified_files, deleted_files)
    """
    old_files = manifest.get("files", {})
    
    current_set = set(current_files.keys())
    old_set = set(old_files.keys())
    
    # Files mới (có trong current, không có trong old)
    new_files = current_set - old_set
    
    # Files đã xóa (có trong old, không có trong current)
    deleted_files = old_set - current_set
    
    # Files đã sửa (hash khác)
    modified_files = set()
    for filename in current_set & old_set:
        if current_files[filename] != old_files[filename]:
            modified_files.add(filename)
    
    return new_files, modified_files, deleted_files


# ================================================================
# DOCUMENT PROCESSING
# ================================================================

def load_and_split_file(filepath: str) -> List[Document]:
    """Load và split một file thành chunks"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "]
    )
    
    chunks = splitter.split_text(content)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": filepath,
                "chunk_id": i,
                "filename": os.path.basename(filepath)
            }
        )
        documents.append(doc)
    
    return documents


# ================================================================
# INDEX OPERATIONS
# ================================================================

def backup_index() -> bool:
    """Backup index hiện tại"""
    if not os.path.exists(settings.DB_DIR):
        return False
    
    # Tạo backup directory
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Copy files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, timestamp)
    os.makedirs(backup_path, exist_ok=True)
    
    for filename in ["index.faiss", "index.pkl", "manifest.json"]:
        src = os.path.join(settings.DB_DIR, filename)
        if os.path.exists(src):
            shutil.copy2(src, backup_path)
    
    print(f"  ✓ Backed up to: {backup_path}")
    return True


def restore_from_backup(backup_path: str) -> bool:
    """Restore index từ backup"""
    if not os.path.exists(backup_path):
        return False
    
    for filename in ["index.faiss", "index.pkl", "manifest.json"]:
        src = os.path.join(backup_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, settings.DB_DIR)
    
    print(f"  ✓ Restored from: {backup_path}")
    return True


def rebuild_full_index(data_dir: str, embeddings) -> FAISS:
    """Rebuild toàn bộ index từ đầu"""
    print("\n🔨 Rebuilding full index...")
    
    all_documents = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(data_dir, filename)
            docs = load_and_split_file(filepath)
            all_documents.extend(docs)
            print(f"  ✓ {filename}: {len(docs)} chunks")
    
    print(f"\n📊 Total: {len(all_documents)} chunks")
    
    # Tạo FAISS index
    print("  Creating FAISS index...")
    db = FAISS.from_documents(all_documents, embeddings)
    
    return db


def incremental_update(
    db: FAISS,
    new_files: Set[str],
    modified_files: Set[str],
    deleted_files: Set[str],
    data_dir: str,
    embeddings
) -> FAISS:
    """
    Update index một cách incremental
    
    Note: FAISS không hỗ trợ delete documents trực tiếp,
    nên với modified/deleted files, ta cần rebuild.
    """
    # Nếu có files bị xóa hoặc sửa, cần rebuild
    if modified_files or deleted_files:
        print("\n⚠️ Modified/deleted files detected - need full rebuild")
        print(f"   Modified: {modified_files}")
        print(f"   Deleted: {deleted_files}")
        return rebuild_full_index(data_dir, embeddings)
    
    # Nếu chỉ có files mới, có thể add thêm
    if new_files:
        print(f"\n➕ Adding {len(new_files)} new files...")
        
        new_documents = []
        for filename in new_files:
            filepath = os.path.join(data_dir, filename)
            docs = load_and_split_file(filepath)
            new_documents.extend(docs)
            print(f"  ✓ {filename}: {len(docs)} chunks")
        
        print(f"  Adding {len(new_documents)} new chunks to index...")
        db.add_documents(new_documents)
        
        return db
    
    # Không có gì thay đổi
    print("\n✅ No changes detected")
    return db


# ================================================================
# MAIN FUNCTION
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Incremental Index Update for RAG ChatBot"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebuild toàn bộ index"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Chỉ check changes, không update"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Không backup trước khi update"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔄 INCREMENTAL INDEX UPDATE")
    print("=" * 60)
    print(f"Data directory: {settings.DATA_DIR}")
    print(f"Vector DB: {settings.DB_DIR}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE UPDATE'}")
    print("-" * 60)
    
    # ============ STEP 1: Scan current files ============
    print("\n📂 Scanning data directory...")
    
    if not os.path.exists(settings.DATA_DIR):
        print(f"❌ Data directory not found: {settings.DATA_DIR}")
        sys.exit(1)
    
    current_files = get_current_files(settings.DATA_DIR)
    print(f"  Found {len(current_files)} files")
    
    # ============ STEP 2: Load manifest ============
    print("\n📋 Loading manifest...")
    manifest = load_manifest()
    
    if manifest.get("last_update"):
        print(f"  Last update: {manifest['last_update']}")
    else:
        print("  No previous manifest found")
    
    # ============ STEP 3: Detect changes ============
    print("\n🔍 Detecting changes...")
    
    new_files, modified_files, deleted_files = detect_changes(
        current_files, manifest
    )
    
    print(f"  New files: {len(new_files)}")
    for f in new_files:
        print(f"    + {f}")
    
    print(f"  Modified files: {len(modified_files)}")
    for f in modified_files:
        print(f"    ~ {f}")
    
    print(f"  Deleted files: {len(deleted_files)}")
    for f in deleted_files:
        print(f"    - {f}")
    
    # ============ STEP 4: Check if update needed ============
    total_changes = len(new_files) + len(modified_files) + len(deleted_files)
    
    if not args.force and total_changes == 0:
        print("\n✅ No changes detected. Index is up to date!")
        return
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes will be made")
        print(f"   Would update: {total_changes} files")
        return
    
    # ============ STEP 5: Backup ============
    if not args.no_backup and os.path.exists(settings.DB_DIR):
        print("\n💾 Backing up current index...")
        backup_index()
    
    # ============ STEP 6: Initialize embeddings ============
    print("\n🧠 Loading embedding model...")
    embeddings = LocalEmbedding()
    
    # ============ STEP 7: Update index ============
    try:
        if args.force or not os.path.exists(settings.DB_DIR):
            # Force rebuild hoặc chưa có index
            db = rebuild_full_index(settings.DATA_DIR, embeddings)
        else:
            # Load existing index và update
            print("\n📂 Loading existing index...")
            db = FAISS.load_local(
                settings.DB_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            db = incremental_update(
                db,
                new_files,
                modified_files,
                deleted_files,
                settings.DATA_DIR,
                embeddings
            )
        
        # ============ STEP 8: Save index ============
        print("\n💾 Saving updated index...")
        os.makedirs(settings.DB_DIR, exist_ok=True)
        db.save_local(settings.DB_DIR)
        print(f"  ✓ Saved to: {settings.DB_DIR}")
        
        # ============ STEP 9: Update manifest ============
        print("\n📋 Updating manifest...")
        manifest["files"] = current_files
        save_manifest(manifest)
        print(f"  ✓ Saved manifest")
        
        # ============ SUMMARY ============
        print("\n" + "=" * 60)
        print("✅ INDEX UPDATE COMPLETED!")
        print("=" * 60)
        print(f"  Total files: {len(current_files)}")
        print(f"  Changes: {total_changes}")
        print(f"  Timestamp: {datetime.now().isoformat()}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n🔄 Attempting rollback...")
        
        # Tìm backup gần nhất
        if os.path.exists(BACKUP_DIR):
            backups = sorted(os.listdir(BACKUP_DIR), reverse=True)
            if backups:
                restore_from_backup(os.path.join(BACKUP_DIR, backups[0]))
                print("  ✓ Rollback completed")
            else:
                print("  ⚠️ No backup available")
        
        sys.exit(1)


# ================================================================
# UTILITY COMMANDS
# ================================================================

def list_backups():
    """Liệt kê các backups"""
    if not os.path.exists(BACKUP_DIR):
        print("No backups found")
        return
    
    backups = sorted(os.listdir(BACKUP_DIR), reverse=True)
    print(f"Found {len(backups)} backups:")
    for backup in backups:
        path = os.path.join(BACKUP_DIR, backup)
        size = sum(
            os.path.getsize(os.path.join(path, f))
            for f in os.listdir(path)
        ) / 1024  # KB
        print(f"  - {backup} ({size:.1f} KB)")


def clean_backups(keep: int = 5):
    """Xóa backups cũ, giữ lại N bản gần nhất"""
    if not os.path.exists(BACKUP_DIR):
        return
    
    backups = sorted(os.listdir(BACKUP_DIR), reverse=True)
    
    if len(backups) <= keep:
        print(f"Only {len(backups)} backups, keeping all")
        return
    
    to_delete = backups[keep:]
    for backup in to_delete:
        path = os.path.join(BACKUP_DIR, backup)
        shutil.rmtree(path)
        print(f"  Deleted: {backup}")
    
    print(f"Cleaned {len(to_delete)} old backups")


if __name__ == "__main__":
    main()