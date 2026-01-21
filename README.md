# RAG ChatBot - Internal Knowledge Base

Hệ thống Chatbot thông minh sử dụng kỹ thuật RAG (Retrieval-Augmented Generation) để trả lời câu hỏi dựa trên tài liệu nội bộ doanh nghiệp.

## Mô tả

Dự án xây dựng một chatbot có khả năng:
- Tìm kiếm và truy xuất thông tin từ tài liệu nội bộ (PDF)
- Trả lời câu hỏi bằng tiếng Việt với ngữ cảnh chính xác
- Lưu trữ lịch sử hội thoại theo từng người dùng
- Xác thực người dùng qua JWT

## Lưu ý về dữ liệu

Dữ liệu trong thư mục `data/` chỉ là **dữ liệu giả lập** phục vụ mục đích demo và phát triển. Các tài liệu PDF mô phỏng quy trình, chính sách của một công ty giả định (ABC Corp) và không phản ánh bất kỳ tổ chức thực tế nào.

## Kiến trúc hệ thống

```
+-------------+     +-------------+     +-------------+
|  Streamlit  | --> |   FastAPI   | --> |    LLM      |
|     UI      |     |   Backend   |     | (OpenRouter)|
+-------------+     +-------------+     +-------------+
                          |
            +-------------+-------------+
            |                           |
      +-----v-----+              +------v------+
      |   Qdrant  |              |    MySQL    |
      | VectorDB  |              |  (Memory)   |
      +-----------+              +-------------+
```

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Backend API | FastAPI |
| Frontend | Streamlit |
| Vector Database | Qdrant |
| Relational Database | MySQL |
| Embedding Model | BAAI/bge-m3 |
| LLM | Mistral (qua OpenRouter) |
| Authentication | JWT |

## Cài đặt

### Yêu cầu

- Python 3.10+
- MySQL Server
- Docker (cho Qdrant)

### Bước 1: Clone và cài đặt dependencies

```bash
git clone <repository-url>
cd RAG-ChatBot

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### Bước 2: Khởi động Qdrant

```bash
docker-compose up -d
```

### Bước 3: Cấu hình environment

Tạo file `.env` trong thư mục gốc:

```env
# OpenRouter API
OPENROUTER_API_KEY=your_openrouter_api_key

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=rag_chatbot
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password

# JWT Secret (thay doi trong production)
JWT_SECRET_KEY=your_super_secret_key_here
```

### Bước 4: Tạo database MySQL

```sql
CREATE DATABASE rag_chatbot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### Bước 5: Nạp dữ liệu vào Vector DB

```bash
python run.py --mode ingest
```

## Chạy ứng dụng

### Chạy API Server

```bash
python run.py --mode api
```

API sẽ chạy tại: http://localhost:8000

### Chạy Streamlit UI

```bash
python run.py --mode streamlit
```

UI sẽ chạy tại: http://localhost:8501

## API Endpoints

| Method | Endpoint | Mô tả | Auth |
|--------|----------|-------|------|
| POST | /auth/register | Đăng ký tài khoản | - |
| POST | /auth/login | Đăng nhập | - |
| GET | /auth/me | Thông tin user hiện tại | JWT |
| POST | /chat | Gửi câu hỏi | JWT |
| GET | /sessions | Danh sách phiên chat | JWT |
| GET | /session/{id}/history | Lịch sử chat | JWT |
| GET | /health | Kiểm tra trạng thái | - |
| GET | /stats | Thống kê hệ thống | Admin |

## Cấu trúc thư mục

```
RAG-ChatBot/
├── app/
│   ├── api.py           # FastAPI endpoints
│   ├── auth.py          # Authentication logic
│   ├── config.py        # Configuration management
│   ├── database.py      # MySQL repositories
│   ├── ingest.py        # Document ingestion
│   ├── llm.py           # LLM interaction
│   ├── memory.py        # Conversation memory
│   ├── models.py        # Pydantic models
│   ├── rag_engine.py    # RAG core logic
│   └── streamlit_app.py # Streamlit UI
├── data/
│   └── legal_kb/        # PDF documents (demo data)
├── scripts/
│   └── update_index.py  # Index management
├── docker-compose.yml   # Qdrant container
├── requirements.txt
├── run.py               # Entry point
└── .env                 # Environment variables (not in git)
```

