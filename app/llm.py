# app/llm.py
# ingest.py  → xây trí nhớ
# rag.py     → tìm đoạn liên quan
# llm.py     → viết câu trả lời
import httpx
from openai import OpenAI

from app.config import settings

# Tạo httpx client không có proxies
http_client = httpx.Client(
    timeout=httpx.Timeout(60.0, connect=10.0)
)

client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
    http_client=http_client
)

def call_llm(prompt: str) -> str:
    """
    Gọi LLM với prompt đã được xây dựng sẵn
    
    Args:
        prompt: Prompt đầy đủ (đã bao gồm context, question, instructions)
        
    Returns:
        Câu trả lời từ LLM
    """
    try:
        response = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        # Lấy response và loại bỏ special tokens
        answer = response.choices[0].message.content
        # Strip các special tokens
        answer = answer.replace("[/s]", "").replace("</s>", "").replace("[/INST]", "").replace("[INST]", "").strip()
        return answer
        
    except Exception as e:
        return f"Lỗi khi gọi LLM: {str(e)}"


# Legacy function for backward compatibility with app.py and rag.py
def call_llm_legacy(context: str, question: str) -> str:
    """
    Hàm cũ để tương thích với code cũ (app.py, rag.py)
    """
    prompt = f"""
Bạn là chatbot nội bộ doanh nghiệp.
Chỉ trả lời dựa trên tài liệu sau.
Nếu không có thông tin, hãy nói "Không tìm thấy trong tài liệu".

TÀI LIỆU:
{context}

CÂU HỎI:
{question}
"""
    return call_llm(prompt)
