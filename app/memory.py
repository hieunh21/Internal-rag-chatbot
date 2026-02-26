from typing import Optional, List
from datetime import datetime

from app.models import Message
from app.config import settings


class ConversationMemory:
    def __init__(self):
        # Lazy import để tránh circular dependency
        from app.database import message_repo, session_repo
        self._message_repo = message_repo
        self._session_repo = session_repo
        
        print(" ConversationMemory initialized (MySQL backend)")

    # CORE METHODS
    def add_message(self, session_id: str, role: str, content: str,
                    sources: list = None, latency: float = None, 
                    is_grounded: bool = None) -> None:
        self._message_repo.add_message(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
            latency=latency,
            is_grounded=is_grounded
        )
    
    def get_history(self, session_id: str) -> str:
        return self._message_repo.get_history_text(
            session_id=session_id,
            max_turns=settings.MAX_HISTORY_TURNS
        )
    
    def get_messages(self, session_id: str) -> List[Message]:
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
    
    
    # SESSION MANAGEMENT
    def clear_session(self, session_id: str) -> bool:
        return self._session_repo.delete_session(session_id)
    
    def session_exists(self, session_id: str) -> bool:
        return self._session_repo.get_session(session_id) is not None
    
    def list_sessions(self) -> List[str]:
        sessions = self._session_repo.list_sessions()
        return [s['session_id'] for s in sessions]
    # STATISTICS
    def get_session_stats(self, session_id: str) -> Optional[dict]:
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
    
    def get_active_sessions(self) -> int:
        try:
            sessions = self._session_repo.list_sessions(limit=1000)
            return len(sessions)
        except Exception:
            return 0

    # FOR STREAMLIT SIDEBAR
    def get_session_summaries(self) -> List[dict]:
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
    
    # COMPATIBILITY METHODS
    
    def save_to_disk(self) -> bool:
        # MySQL đã tự động persist, không cần làm gì
        return True

# SINGLETON INSTANCE
memory = ConversationMemory()
