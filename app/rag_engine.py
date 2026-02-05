"""
RAG Engine - Core của hệ thống Retrieval-Augmented Generation
Using Qdrant Vector Database

Features:
    - Retrieve documents với similarity scores
    - Threshold filtering (lọc kết quả không liên quan)
    - Citation formatting (trích dẫn nguồn)
    - Integration với Conversation Memory
    - Fallback handling (khi không tìm thấy nguồn)
"""

import os
import time
from typing import List, Tuple, Optional

from qdrant_client import QdrantClient
from langchain.schema import Document

from app.ingest import LocalEmbedding
from app.models import Source
from app.config import settings
from app.llm import call_llm


class RAGEngine:
    """
    RAG Engine với Qdrant và citations
    
    Workflow:
        1. Retrieve: Tìm documents liên quan từ Qdrant
        2. Filter: Lọc theo ngưỡng similarity
        3. Build Context: Ghép documents thành context
        4. Generate: Gọi LLM với context + history
        5. Format: Trả về answer + sources
    """
    
    def __init__(self):
        """Initialize RAG Engine với Qdrant"""
        print("Initializing RAG Engine...")
        
        # Embedding model (same as ingest)
        self.embeddings = LocalEmbedding()
        
        # Qdrant client
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._connect_db()
        
        print("RAG Engine ready!")
    
    def _connect_db(self) -> None:
        """Kết nối đến Qdrant server"""
        try:
            self.client = QdrantClient(url=settings.QDRANT_URL)
            
            # Verify collection exists
            collection_info = self.client.get_collection(self.collection_name)
            print(f"  ✓ Connected to Qdrant at '{settings.QDRANT_URL}'")
            print(f"  ✓ Collection: {self.collection_name} ({collection_info.points_count} vectors)")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Qdrant at '{settings.QDRANT_URL}'. "
                f"Please ensure Qdrant is running and run 'python -m app.ingest' first!\n"
                f"Error: {e}"
            )
    
    def reload_db(self) -> None:
        """Reconnect to Qdrant (sau khi update index)"""
        print("Reconnecting to Qdrant...")
        self._connect_db()
        print("Reconnected!")
    
    # ================================================================
    # RETRIEVAL METHODS
    # ================================================================
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = None
    ) -> List[Tuple]:
        """
        Retrieve documents từ Qdrant với similarity scores
        
        Args:
            query: Câu hỏi của user
            k: Số documents cần retrieve (default: TOP_K từ config)
            
        Returns:
            List[(Document, similarity_score)]
            Score từ 0-1, càng cao càng giống (COSINE similarity)
        """
        k = k or settings.TOP_K
        
        # Embed query
        query_vector = self.embeddings.embed_query(query)
        
        # Search in Qdrant (query_points for newer versions)
        from qdrant_client.models import NamedVector
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            with_payload=True
        ).points
        
        # Convert Qdrant results to (Document, score) format
        results_with_similarity = []
        for hit in results:
            # Tạo Document từ payload
            doc = Document(
                page_content=hit.payload.get("content", ""),
                metadata={
                    "source": hit.payload.get("source", "unknown"),
                    "file_type": hit.payload.get("file_type", "unknown"),
                    "page": hit.payload.get("page", 0),
                    "chunk_id": hit.payload.get("chunk_id", 0)
                }
            )
            # Qdrant COSINE score đã là 0-1
            similarity = hit.score
            results_with_similarity.append((doc, similarity))
        
        return results_with_similarity
    
    def filter_by_threshold(self, results: List[Tuple]) -> List[Tuple]:
        """
        Lọc kết quả theo ngưỡng similarity
        
        Args:
            results: List[(Document, score)]
            
        Returns:
            List[(Document, score)] đã lọc
        """
        threshold = settings.SIMILARITY_THRESHOLD
        filtered = [(doc, score) for doc, score in results if score >= threshold]
        return filtered
    
    # ================================================================
    # CITATION FORMATTING
    # ================================================================
    
    def format_sources(self, results: List[Tuple]) -> List[Source]:
        """
        Chuyển results thành Source objects (citations)
        
        Args:
            results: List[(Document, score)]
            
        Returns:
            List[Source] với đầy đủ thông tin trích dẫn
        """
        sources = []
        
        for idx, (doc, score) in enumerate(results):
            # Lấy tên file từ metadata
            source_path = doc.metadata.get("source", "unknown")
            source_file = os.path.basename(source_path)
            
            # Nội dung đầy đủ
            full_content = doc.page_content
            
            # Tạo excerpt (2-3 câu đầu làm preview)
            sentences = full_content.split('. ')
            excerpt = '. '.join(sentences[:2])
            if len(sentences) > 2:
                excerpt += '...'
            if len(excerpt) > 200:
                excerpt = excerpt[:197] + '...'
            
            # Lấy page number từ metadata
            page = doc.metadata.get("page")
            chunk_id = doc.metadata.get("chunk_id", idx)
            
            # Tạo Source object
            sources.append(Source(
                source=source_file,
                chunk_id=chunk_id,
                score=round(score, 4),
                excerpt=excerpt,
                full_content=full_content,
                page=page
            ))
        
        return sources
    
    # ================================================================
    # CONTEXT BUILDING
    # ================================================================
    
    def build_context(self, results: List[Tuple]) -> str:
        """
        Ghép các documents thành context string cho LLM
        
        Args:
            results: List[(Document, score)]
            
        Returns:
            String chứa context đã format
        """
        if not results:
            return ""
        
        context_parts = []
        
        for idx, (doc, score) in enumerate(results, 1):
            # Lấy tên file
            source_path = doc.metadata.get("source", "unknown")
            source_file = os.path.basename(source_path)
            
            # Format mỗi document
            context_parts.append(
                f"[Nguồn {idx}: {source_file}]\n"
                f"(Độ liên quan: {score:.1%})\n"
                f"{doc.page_content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    # ================================================================
    # PROMPT BUILDING
    # ================================================================
    
    def build_prompt(
        self,
        question: str,
        context: str,
        history: str = ""
    ) -> str:
        """
        Xây dựng prompt cho LLM
        
        Args:
            question: Câu hỏi của user
            context: Context từ retrieved documents
            history: Lịch sử hội thoại (optional)
            
        Returns:
            Prompt string hoàn chỉnh
        """
        # Base system instruction
        system_instruction = """Bạn là trợ lý AI của Công ty ABC (ABC Corp), chuyên trả lời các câu hỏi về quy định, chính sách công ty.

Nguyên tắc trả lời:
1. CHỈ trả lời dựa trên thông tin trong tài liệu được cung cấp
2. Trả lời ngắn gọn, súc tích, dễ hiểu
3. Nếu thông tin không có trong tài liệu, nói rõ "Thông tin này không có trong tài liệu nội bộ"
4. Không bịa đặt thông tin
5. Có thể tham chiếu ngữ cảnh hội thoại trước đó nếu liên quan"""

        # Build prompt dựa trên có history hay không
        if history:
            prompt = f"""{system_instruction}

--- LỊCH SỬ HỘI THOẠI ---
{history}

--- TÀI LIỆU THAM KHẢO ---
{context}

--- CÂU HỎI MỚI ---
{question}

Trả lời:"""
        else:
            prompt = f"""{system_instruction}

--- TÀI LIỆU THAM KHẢO ---
{context}

--- CÂU HỎI ---
{question}

Trả lời:"""
        
        return prompt
    
    def build_fallback_prompt(self, question: str, history: str = "") -> str:
        """
        Prompt khi không tìm thấy nguồn liên quan
        Cho phép LLM trả lời chung chung hoặc từ chối lịch sự
        """
        if history:
            return f"""Bạn là trợ lý AI của Công ty ABC (ABC Corp).

Lịch sử hội thoại:
{history}

Câu hỏi: {question}

Lưu ý: Không tìm thấy thông tin liên quan trong tài liệu nội bộ.
Hãy trả lời lịch sự, gợi ý người dùng liên hệ bộ phận phù hợp (HR, IT, Legal).
Nếu câu hỏi hoàn toàn không liên quan đến công việc, từ chối lịch sự.

Trả lời:"""
        else:
            return f"""Bạn là trợ lý AI của Công ty ABC (ABC Corp).

Câu hỏi: {question}

Lưu ý: Không tìm thấy thông tin liên quan trong tài liệu nội bộ.
Hãy trả lời lịch sự, gợi ý người dùng liên hệ bộ phận phù hợp.

Trả lời:"""
    
    # ================================================================
    # MAIN ASK METHOD
    # ================================================================
    
    def ask(
        self,
        question: str,
        history: str = "",
        use_fallback: bool = True
    ) -> Tuple[str, List[Source], bool, float]:
        """
        Main RAG method - xử lý câu hỏi và trả về answer với sources
        
        Args:
            question: Câu hỏi của user
            history: Lịch sử hội thoại (từ memory)
            use_fallback: Có dùng fallback khi không tìm thấy nguồn
            
        Returns:
            Tuple gồm:
            - answer: Câu trả lời từ LLM
            - sources: List[Source] citations
            - is_grounded: True nếu có nguồn hỗ trợ
            - latency_ms: Thời gian xử lý (milliseconds)
        """
        start_time = time.time()
        
        # ============ STEP 1: RETRIEVE ============
        results = self.retrieve_with_scores(question)
        
        # ============ STEP 2: FILTER BY THRESHOLD ============
        filtered_results = self.filter_by_threshold(results)
        
        # ============ STEP 3: CHECK IF GROUNDED ============
        is_grounded = len(filtered_results) > 0
        
        # ============ STEP 4: GENERATE ANSWER ============
        if is_grounded:
            # Có nguồn → build context và generate
            context = self.build_context(filtered_results)
            prompt = self.build_prompt(question, context, history)
            
            try:
                answer = call_llm(prompt)
            except Exception as e:
                answer = f"Xin lỗi, đã có lỗi khi xử lý: {str(e)}"
            
            # Format sources
            sources = self.format_sources(filtered_results)
            
        else:
            # Không có nguồn phù hợp
            if use_fallback:
                # Dùng fallback prompt
                prompt = self.build_fallback_prompt(question, history)
                try:
                    answer = call_llm(prompt)
                except Exception as e:
                    answer = (
                        "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu nội bộ. "
                        "Vui lòng liên hệ bộ phận HR hoặc Legal để được hỗ trợ."
                    )
            else:
                # Không dùng LLM, trả về message cứng
                answer = (
                    "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu nội bộ. "
                    "Vui lòng liên hệ:\n"
                    "- HR: hr@abccorp.vn\n"
                    "- Legal: legal@abccorp.vn\n"
                    "- IT Support: it@abccorp.vn"
                )
            
            sources = []
        
        # ============ STEP 5: CALCULATE LATENCY ============
        latency_ms = (time.time() - start_time) * 1000
        
        return answer, sources, is_grounded, latency_ms
    
    # ================================================================
    # UTILITY METHODS
    # ================================================================
    
    def get_similar_questions(
        self, 
        question: str, 
        k: int = 3
    ) -> List[str]:
        """
        Tìm các đoạn tài liệu tương tự (để suggest)
        
        Useful cho "Did you mean...?" feature
        """
        results = self.retrieve_with_scores(question, k=k)
        
        suggestions = []
        for doc, score in results:
            if score >= 0.2:  # Chỉ lấy nếu có phần nào liên quan
                # Lấy 50 ký tự đầu làm suggestion
                suggestion = doc.page_content[:50] + "..."
                suggestions.append(suggestion)
        
        return suggestions
    
    def health_check(self) -> dict:
        """Kiểm tra trạng thái của RAG Engine"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            points_count = collection_info.points_count
        except:
            points_count = 0
            
        return {
            "db_connected": self.client is not None,
            "qdrant_url": settings.QDRANT_URL,
            "collection": self.collection_name,
            "vectors_count": points_count,
            "embedding_model": settings.EMBEDDING_MODEL,
            "top_k": settings.TOP_K,
            "threshold": settings.SIMILARITY_THRESHOLD
        }


# ================================================================
# SINGLETON INSTANCE
# ================================================================

# Tạo instance duy nhất để dùng trong toàn bộ application
# Lazy loading - chỉ tạo khi import lần đầu
rag_engine = RAGEngine()