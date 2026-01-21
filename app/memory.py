"""
Conversation Memory Management
Quản lý lịch sử hội thoại theo session_id - MySQL Backend
"""

from typing import Optional, List
from datetime import datetime

from app.models import Message
from app.config import settings


class ConversationMemory:
    """
    Quản lý memory cho nhiều conversation sessions
    Backend: MySQL Database
    
    Features:
        - Persistent storage (MySQL)
        - Session management (create, clear, list)
        - Export history dạng text cho prompt
    """
    
    def __init__(self):
        """Initialize memory với MySQL backend"""
        # Lazy import để tránh circular dependency
        from app.database import message_repo, session_repo
        self._message_repo = message_repo
        self._session_repo = session_repo
        
        print(" ConversationMemory initialized (MySQL backend)")
    
    # ================================================================
    # CORE METHODS
    # ================================================================
    
    def add_message(self, session_id: str, role: str, content: str,
                    sources: list = None, latency: float = None, 
                    is_grounded: bool = None) -> None:
        """
        Thêm message vào session (lưu vào MySQL)
        
        Args:
            session_id: ID của session
            role: "user" hoặc "assistant"
            content: Nội dung message
            sources: Danh sách nguồn trích dẫn (cho assistant)
            latency: Thời gian xử lý ms (cho assistant)
            is_grounded: Có nguồn hỗ trợ không (cho assistant)
        """
        self._message_repo.add_message(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
            latency=latency,
            is_grounded=is_grounded
        )
    
    def get_history(self, session_id: str) -> str:
        """
        Lấy lịch sử hội thoại dạng text (để ghép vào prompt)
        
        Args:
            session_id: ID của session
            
        Returns:
            String chứa lịch sử hội thoại format:
            "User: câu hỏi 1
             Assistant: trả lời 1
             ..."
        """
        return self._message_repo.get_history_text(
            session_id=session_id,
            max_turns=settings.MAX_HISTORY_TURNS
        )
    
    def get_messages(self, session_id: str) -> List[Message]:
        """
        Lấy danh sách messages của session
        
        Args:
            session_id: ID của session
            
        Returns:
            List[Message] hoặc [] nếu session không tồn tại
        """
        messages_data = self._message_repo.get_messages(session_id)
        
        # Convert to Message objects
        messages = []
        for msg in messages_data:
            messages.append(Message(
                role=msg['role'],
                content=msg['content'],
                timestamp=msg['timestamp'],
                sources=msg.get('sources'),
                latency=msg.get('latency'),
                is_grounded=msg.get('is_grounded')
            ))
        
        return messages
    
    # ================================================================
    # SESSION MANAGEMENT
    # ================================================================
    
    def clear_session(self, session_id: str) -> bool:
        """
        Xóa một session và tất cả messages
        
        Args:
            session_id: ID của session cần xóa
            
        Returns:
            True nếu xóa thành công
        """
        return self._session_repo.delete_session(session_id)
    
    def session_exists(self, session_id: str) -> bool:
        """Kiểm tra session có tồn tại không"""
        return self._session_repo.get_session(session_id) is not None
    
    def list_sessions(self) -> List[str]:
        """Lấy danh sách tất cả session IDs"""
        sessions = self._session_repo.list_sessions()
        return [s['session_id'] for s in sessions]
    
    # ================================================================
    # STATISTICS
    # ================================================================
    
    def get_session_stats(self, session_id: str) -> Optional[dict]:
        """
        Lấy thống kê của một session
        """
        session = self._session_repo.get_session(session_id)
        if not session:
            return None
        
        message_count = self._message_repo.count_messages(session_id)
        
        return {
            "session_id": session_id,
            "message_count": message_count,
            "created_at": session['created_at'].isoformat() if session['created_at'] else None,
            "updated_at": session['updated_at'].isoformat() if session['updated_at'] else None
        }
    
    # ================================================================
    # FOR STREAMLIT SIDEBAR
    # ================================================================
    
    def get_session_summaries(self) -> List[dict]:
        """
        Lấy danh sách tóm tắt các sessions (để hiển thị trong sidebar)
        
        Returns:
            List[{session_id, title, message_count, updated_at}]
        """
        sessions = self._session_repo.list_sessions(limit=20)
        
        summaries = []
        for sess in sessions:
            first_question = sess.get('first_question') or ""
            if len(first_question) > 50:
                first_question = first_question[:50] + "..."
            
            summaries.append({
                "session_id": sess['session_id'],
                "title": first_question or "New Chat",
                "message_count": sess.get('message_count', 0),
                "updated_at": sess['updated_at']
            })
        
        return summaries
    
    # ================================================================
    # COMPATIBILITY METHODS (for existing code)
    # ================================================================
    
    def save_to_disk(self) -> bool:
        """
        Compatibility method - MySQL tự động lưu
        """
        # MySQL đã tự động persist, không cần làm gì
        return True


# ================================================================
# SINGLETON INSTANCE
# ================================================================

memory = ConversationMemory()
