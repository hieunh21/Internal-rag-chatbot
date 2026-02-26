import os
import time
from typing import List, Tuple, Optional

from qdrant_client import QdrantClient
from langchain.schema import Document
from sentence_transformers import CrossEncoder

from app.ingest import LocalEmbedding
from app.models import Source
from app.config import settings
from app.llm import call_llm


class RAGEngine:
    def __init__(self):
        print("Initializing RAG Engine...")
        
        # Embedding model (same as ingest)
        self.embeddings = LocalEmbedding()
        
        # Qdrant client
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._connect_db()
        
        # Reranker model (optional)
        self.reranker = None
        if settings.USE_RERANKER:
            self._load_reranker()
        
        print("RAG Engine ready!")
    
    def _load_reranker(self) -> None:
        try:
            print(f"  Loading reranker: {settings.RERANKER_MODEL}...")
            self.reranker = CrossEncoder(
                settings.RERANKER_MODEL,
                max_length=512
            )
            print(f"Reranker loaded: {settings.RERANKER_MODEL}")
        except Exception as e:
            print(f"Reranker load failed: {e}. Continuing without reranking.")
            self.reranker = None
    
    def _connect_db(self) -> None:
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
        print("Reconnecting to Qdrant...")
        self._connect_db()
        print("Reconnected!")
    

    # RETRIEVAL METHODS
    def rerank(self, query: str, results: List[Tuple]) -> List[Tuple]:
        if not self.reranker or not results:
            return results
        
        # Tạo pairs (query, passage) cho CrossEncoder
        pairs = [(query, doc.page_content) for doc, _ in results]
        
        # Predict rerank scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Gắn rerank score vào results
        reranked = [
            (doc, float(rerank_score))
            for (doc, _), rerank_score in zip(results, rerank_scores)
        ]
        
        # Sắp xếp theo rerank score giảm dần
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy top K và filter theo POST_RERANK_THRESHOLD
        top_k = reranked[:settings.RERANKER_TOP_K]
        filtered = [(doc, score) for doc, score in top_k if score >= settings.POST_RERANK_THRESHOLD]
        return filtered

    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = None
    ) -> List[Tuple]:
        if self.reranker:
            k = k or settings.RERANKER_CANDIDATES
        else:
            k = k or settings.TOP_K
        
        # Embed query
        query_vector = self.embeddings.embed_query(query)
        
        # Search in Qdrant
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
    
    def filter_by_threshold(self, results: List[Tuple], pre_rerank: bool = False) -> List[Tuple]:
        if pre_rerank:
            # Trước rerank: chỉ loại rác rõ ràng, ngưỡng thấp
            threshold = settings.PRE_RERANK_THRESHOLD
        else:
            # Không có reranker: dùng threshold chuẩn
            threshold = settings.SIMILARITY_THRESHOLD
        filtered = [(doc, score) for doc, score in results if score >= threshold]
        return filtered
    # CITATION FORMATTING
    def format_sources(self, results: List[Tuple]) -> List[Source]:
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
    
    # CONTEXT BUILDING
    def build_context(self, results: List[Tuple]) -> str:
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
    

    # PROMPT BUILDING
    def build_prompt(
        self,
        question: str,
        context: str,
        history: str = ""
    ) -> str:
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
    
    def ask(
        self,
        question: str,
        history: str = "",
        use_fallback: bool = True
    ) -> Tuple[str, List[Source], bool, float]:
        start_time = time.time()
        
        # ============ STEP 1: RETRIEVE ============
        results = self.retrieve_with_scores(question)
        
        # ============ STEP 2: FILTER ============
        if self.reranker:
            # Có reranker: dùng ngưỡng thấp (0.1) để không loại nhầm candidates
            filtered_results = self.filter_by_threshold(results, pre_rerank=True)
        else:
            # Không có reranker: dùng ngưỡng chuẩn (0.25)
            filtered_results = self.filter_by_threshold(results, pre_rerank=False)
        
        # ============ STEP 3: RERANK (CrossEncoder scoring + post-rerank filter) ============
        if self.reranker and filtered_results:
            filtered_results = self.rerank(question, filtered_results)
        
        # ============ STEP 4: CHECK IF GROUNDED ============
        is_grounded = len(filtered_results) > 0
        
        # ============ STEP 5: GENERATE ANSWER ============
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
        
        # ============ STEP 6: CALCULATE LATENCY ============
        latency_ms = (time.time() - start_time) * 1000
        
        return answer, sources, is_grounded, latency_ms
    
    # UTILITY METHODS
    def get_similar_questions(
        self, 
        question: str, 
        k: int = 3
    ) -> List[str]:
        results = self.retrieve_with_scores(question, k=k)
        
        suggestions = []
        for doc, score in results:
            if score >= 0.2:  # Chỉ lấy nếu có phần nào liên quan
                # Lấy 50 ký tự đầu làm suggestion
                suggestion = doc.page_content[:50] + "..."
                suggestions.append(suggestion)
        
        return suggestions
    
    def health_check(self) -> dict:
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
            "threshold": settings.SIMILARITY_THRESHOLD,
            "reranker": settings.RERANKER_MODEL if settings.USE_RERANKER else None
        }

# SINGLETON INSTANCE
# Tạo instance duy nhất để dùng trong toàn bộ application
# Lazy loading - chỉ tạo khi import lần đầu
rag_engine = RAGEngine()