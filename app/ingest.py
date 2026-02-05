"""
Document Ingestion - Qdrant Vector Database
Hỗ trợ PDF (bao gồm bảng), Markdown, Text
"""

import os
import uuid
from pathlib import Path
import pdfplumber
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import settings

# ================================================================
# EMBEDDING MODEL (Local - FREE)
# ================================================================

class LocalEmbedding(Embeddings):
    """
    Local Embedding sử dụng SentenceTransformer
    Model được cấu hình trong config.py
    """
    def __init__(self):
        print(f" Loading embedding: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = settings.EMBEDDING_DIM
    
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()


# ================================================================
# PDF EXTRACTION WITH TABLE SUPPORT
# ================================================================

def format_table(table: list, table_num: int) -> str:
    """
    Format bảng thành text có cấu trúc dễ đọc cho LLM
    """
    if not table or not table[0]:
        return ""
    
    lines = [f"\n[Bảng {table_num}]"]
    headers = table[0]
    
    for row_idx, row in enumerate(table):
        if row_idx == 0:
            # Header row
            header_text = " | ".join(str(cell or "").strip() for cell in row)
            lines.append(f"Cột: {header_text}")
        else:
            # Data row - kết hợp header để dễ hiểu
            row_parts = []
            for col_idx, cell in enumerate(row):
                cell_value = str(cell or "").strip()
                if cell_value and col_idx < len(headers) and headers[col_idx]:
                    header_name = str(headers[col_idx]).strip()
                    row_parts.append(f"{header_name}: {cell_value}")
                elif cell_value:
                    row_parts.append(cell_value)
            if row_parts:
                lines.append("• " + ", ".join(row_parts))
    
    return "\n".join(lines)


def extract_pdf_with_tables(pdf_path: str) -> list:
    """
    Extract PDF bao gồm bảng với PDFPlumber
    """
    docs = []
    filename = Path(pdf_path).name
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_content = []
            
            # 1. Extract text thường
            text = page.extract_text()
            if text:
                page_content.append(text)
            
            # 2. Extract bảng (tables) với format đẹp
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables, 1):
                if table and len(table) > 1:
                    table_text = format_table(table, table_idx)
                    if table_text:
                        page_content.append(table_text)
            
            # Gộp content
            full_content = "\n\n".join(page_content)
            
            if full_content.strip():
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "source": filename,
                        "file_type": "pdf",
                        "page": page_num
                    }
                )
                docs.append(doc)
    
    return docs


# ================================================================
# DOCUMENT LOADING
# ================================================================

def load_documents(data_dir: str):
    """
    Load tất cả documents: .pdf, .md, .txt
    """
    docs = []
    
    print(f"Source: {data_dir}")
    print()
    
    for file in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, file)
        
        # Bỏ qua thư mục
        if os.path.isdir(path):
            continue
        
        try:
            # ==================== PDF (với table support) ====================
            if file.endswith('.pdf'):
                loaded_docs = extract_pdf_with_tables(path)
                docs.extend(loaded_docs)
                print(f" {file} ({len(loaded_docs)} pages, tables extracted)")
            
            # ==================== MARKDOWN & TEXT ====================
            elif file.endswith(('.md', '.txt')):
                loader = TextLoader(path, encoding="utf-8")
                loaded_docs = loader.load()
                
                for doc in loaded_docs:
                    doc.metadata["source"] = file
                    doc.metadata["file_type"] = "markdown" if file.endswith('.md') else "text"
                
                docs.extend(loaded_docs)
                print(f" {file}")
            
            else:
                print(f" Skipped: {file} (unsupported format)")
                
        except Exception as e:
            print(f"Error: {file} - {e}")
    
    return docs


# ================================================================
# QDRANT INDEX BUILDING
# ================================================================
def build_index():
    # 1.Load data
    docs = load_documents(settings.DATA_DIR)
    if not docs:
        print("\n No documents found")
        return
    print(f"\n Total loaded: {len(docs)} documents")
    # 2. Chunking
    print("\n Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.CHUNK_SIZE,
        chunk_overlap = settings.CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""] #ngắt 
    )
    chunks = splitter.split_documents(docs)
    # Thêm chunk_id vào metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    # 3. Create embeddings
    print("\n Creating embeddings...")
    embeddings = LocalEmbedding()
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    # 4. Connect to Qdrant
    print("\n Connecting to Qdrant...")
    client = QdrantClient(url = settings.QDRANT_URL)
    # 5. Xóa cũ nếu có
    collection_name = settings.QDRANT_COLLECTION_NAME
    # Xóa collection cũ nếu đã tồn tại
    try:
        client.delete_collection(collection_name)
        print(f" Xóa collection cũ: {collection_name}")
    except:
        pass
    #Tạo collection mới
    client.create_collection(
        collection_name = collection_name,
        vectors_config = VectorParams(
            size = embeddings.dimension,
            distance = Distance.COSINE
        )
    )
    print(f" Tạo collection: {collection_name}")

    # 6.Upload vectors
    print("\n Uploading to Qdrant...")
    points = [] #trong Qdrant là Point = [id, vector, payload]
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point = PointStruct(
            id = str(uuid.uuid4()),
            vector = vector,
            payload = {
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "file_type": chunk.metadata.get("file_type", "unknown"),
                "page": chunk.metadata.get("page", 0), 
                "chunk_id": chunk.metadata.get("chunk_id", i)
            }
        )
        points.append(point)
    
    # Upload theo batch
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name = collection_name,
            points = batch
        )
        print(f" → Uploaded {min(i + batch_size, len(points))}/{len(points)} points")
    # 7. Verify
    collection_info = client.get_collection(collection_name)
    print("\n" + "=" * 60)
    print(" QDRANT INDEX BUILT SUCCESSFULLY!")
    print(f" Collection: {collection_name}")
    print(f" Total vectors: {collection_info.points_count}")
    print(f" Qdrant URL: {settings.QDRANT_URL}")
    print(f" Dashboard: http://localhost:6333/dashboard")
    print("=" * 60)



# def build_index():
#     """
#     Build Qdrant index từ tất cả documents
#     """
#     print("=" * 60)
#     print("   DOCUMENT INGESTION PIPELINE")
#     print("   Using: Qdrant Vector Database (Docker)")
#     print("   PDF table extraction enabled (pdfplumber)")
#     print("=" * 60)
    
#     # 1. Load documents
#     print("\n Loading documents...")
#     docs = load_documents(settings.DATA_DIR)
    
#     if not docs:
#         print("\n No documents found!")
#         return
    
#     print(f"\n Total loaded: {len(docs)} documents")
    
#     # 2. Chunking
#     print("\n Chunking...")
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=settings.CHUNK_SIZE,
#         chunk_overlap=settings.CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
#     )
#     chunks = splitter.split_documents(docs)
    
#     # Thêm chunk_id vào metadata
#     for i, chunk in enumerate(chunks):
#         chunk.metadata["chunk_id"] = i
    
#     print(f"  → {len(chunks)} chunks created")
    
#     # 3. Create embeddings
#     print("\n Creating embeddings...")
#     embeddings = LocalEmbedding()
#     texts = [chunk.page_content for chunk in chunks]
#     vectors = embeddings.embed_documents(texts)
    
#     # 4. Connect to Qdrant
#     print(f"\n🔌 Connecting to Qdrant at {settings.QDRANT_URL}...")
#     client = QdrantClient(url=settings.QDRANT_URL)
    
#     # 5. Recreate collection (xóa cũ nếu có)
#     collection_name = settings.QDRANT_COLLECTION_NAME
    
#     # Xóa collection cũ nếu tồn tại
#     try:
#         client.delete_collection(collection_name)
#         print(f"  Deleted existing collection: {collection_name}")
#     except:
#         pass
    
#     # Tạo collection mới
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(
#             size=embeddings.dimension,  # 384 for all-MiniLM-L6-v2
#             distance=Distance.COSINE
#         )
#     )
#     print(f"   Created collection: {collection_name}")
    
#     # 6. Upload vectors
#     print("\n Uploading to Qdrant...")
    
#     points = []
#     for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
#         point = PointStruct(
#             id=str(uuid.uuid4()),  # UUID cho mỗi point
#             vector=vector,
#             payload={
#                 "content": chunk.page_content,
#                 "source": chunk.metadata.get("source", "unknown"),
#                 "file_type": chunk.metadata.get("file_type", "unknown"),
#                 "page": chunk.metadata.get("page", 0),
#                 "chunk_id": chunk.metadata.get("chunk_id", i)
#             }
#         )
#         points.append(point)
    
#     # Upload theo batch (tối ưu performance)
#     batch_size = 100
#     for i in range(0, len(points), batch_size):
#         batch = points[i:i + batch_size]
#         client.upsert(
#             collection_name=collection_name,
#             points=batch
#         )
#         print(f"  → Uploaded {min(i + batch_size, len(points))}/{len(points)} points")
    
#     # 7. Verify
#     collection_info = client.get_collection(collection_name)
    
#     print("\n" + "=" * 60)
#     print(" QDRANT INDEX BUILT SUCCESSFULLY!")
#     print(f" Collection: {collection_name}")
#     print(f" Total vectors: {collection_info.points_count}")
#     print(f" Qdrant URL: {settings.QDRANT_URL}")
#     print(f" Dashboard: http://localhost:6333/dashboard")
#     print("=" * 60)


if __name__ == "__main__":
    build_index()
