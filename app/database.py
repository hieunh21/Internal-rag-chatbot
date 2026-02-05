"""
PostgreSQL Database Module
Quản lý kết nối và operations với PostgreSQL
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import json
from datetime import datetime

from app.config import settings


class Database:
    """
    PostgreSQL Database Manager
    
    Features:
        - Connection management
        - Auto-reconnect
        - Context manager for transactions
    """
    
    def __init__(self):
        """Initialize database connection"""
        self._connection = None
        self._connect()
        self._create_tables()
        print(" PostgreSQL Database initialized")
    
    def _connect(self) -> None:
        """Tạo kết nối đến PostgreSQL"""
        try:
            self._connection = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD or "",
                dbname=settings.POSTGRES_DATABASE,
                cursor_factory=RealDictCursor
            )
            self._connection.autocommit = True
            print(f"  ✓ Connected to PostgreSQL at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
        except psycopg2.OperationalError as e:
            print(f"  ✗ PostgreSQL connection failed: {e}")
            raise e
    
    def _create_tables(self) -> None:
        """Tạo các bảng cần thiết"""
        
        # Bảng sessions
        create_sessions = """
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_session_id ON sessions(session_id);
        CREATE INDEX IF NOT EXISTS idx_updated_at ON sessions(updated_at);
        """
        
        # Bảng messages
        create_messages = """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            sources JSONB,
            latency FLOAT,
            is_grounded BOOLEAN,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_msg_session_id ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_msg_timestamp ON messages(timestamp);
        """
        
        # Bảng users
        create_users = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(255),
            role VARCHAR(20) DEFAULT 'user',
            is_active BOOLEAN DEFAULT TRUE,
            last_login_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_user_email ON users(email);
        """
        
        # Function và Trigger để tự động cập nhật updated_at
        create_update_trigger = """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_sessions_updated_at') THEN
                CREATE TRIGGER update_sessions_updated_at
                    BEFORE UPDATE ON sessions
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_users_updated_at') THEN
                CREATE TRIGGER update_users_updated_at
                    BEFORE UPDATE ON users
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            END IF;
        END;
        $$;
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(create_sessions)
            cursor.execute(create_messages)
            cursor.execute(create_users)
            cursor.execute(create_update_trigger)
        
        print("  ✓ Tables created/verified")
    
    @contextmanager
    def get_cursor(self):
        """Context manager để lấy cursor an toàn"""
        # Kiểm tra và reconnect nếu cần
        if self._connection is None or self._connection.closed:
            self._connect()
        
        cursor = self._connection.cursor()
        try:
            yield cursor
            self._connection.commit()
        except Exception as e:
            self._connection.rollback()
            raise e
        finally:
            cursor.close()
    
    def close(self) -> None:
        """Đóng kết nối"""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def test_connection(self) -> bool:
        """Test PostgreSQL connection"""
        try:
            if self._connection is None or self._connection.closed:
                self._connect()
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        except:
            return False


# ================================================================
# SESSION OPERATIONS
# ================================================================

class SessionRepository:
    """Repository pattern cho Session operations"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_session(self, session_id: str) -> bool:
        """Tạo session mới"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    "INSERT INTO sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (session_id,)
                )
            return True
        except psycopg2.IntegrityError:
            # Session đã tồn tại
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Lấy thông tin session"""
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM sessions WHERE session_id = %s",
                (session_id,)
            )
            return cursor.fetchone()
    
    def update_session(self, session_id: str) -> None:
        """Cập nhật timestamp của session"""
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
                (session_id,)
            )
    
    def delete_session(self, session_id: str) -> bool:
        """Xóa session (cascade xóa messages)"""
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM sessions WHERE session_id = %s",
                (session_id,)
            )
            return cursor.rowcount > 0
    
    def list_sessions(self, limit: int = 20) -> List[Dict]:
        """Lấy danh sách sessions mới nhất"""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    s.session_id,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.id) as message_count,
                    (SELECT content FROM messages 
                     WHERE session_id = s.session_id AND role = 'user' 
                     ORDER BY timestamp LIMIT 1) as first_question
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id, s.created_at, s.updated_at
                ORDER BY s.updated_at DESC
                LIMIT %s
            """, (limit,))
            return cursor.fetchall()


# ================================================================
# MESSAGE OPERATIONS
# ================================================================

class MessageRepository:
    """Repository pattern cho Message operations"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def add_message(self, session_id: str, role: str, content: str,
                   sources: List[Dict] = None, latency: float = None,
                   is_grounded: bool = None) -> int:
        """Thêm message mới"""
        with self.db.get_cursor() as cursor:
            # Đảm bảo session tồn tại (INSERT ... ON CONFLICT DO NOTHING)
            cursor.execute(
                "INSERT INTO sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (session_id,)
            )
            
            # Thêm message
            cursor.execute("""
                INSERT INTO messages 
                (session_id, role, content, sources, latency, is_grounded)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                session_id,
                role,
                content,
                json.dumps(sources, ensure_ascii=False) if sources else None,
                latency,
                is_grounded
            ))
            
            result = cursor.fetchone()
            message_id = result['id'] if result else 0
            
            # Cập nhật session timestamp
            cursor.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
                (session_id,)
            )
            
            return message_id
    
    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        """Lấy messages của session"""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM messages 
                WHERE session_id = %s 
                ORDER BY timestamp ASC
                LIMIT %s
            """, (session_id, limit))
            
            messages = cursor.fetchall()
            
            # Parse JSON sources (psycopg2 tự động parse JSONB)
            for msg in messages:
                if msg['sources'] and isinstance(msg['sources'], str):
                    msg['sources'] = json.loads(msg['sources'])
            
            return messages
    
    def get_history_text(self, session_id: str, max_turns: int = 5) -> str:
        """Lấy lịch sử dạng text cho prompt"""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT role, content FROM messages 
                WHERE session_id = %s 
                ORDER BY timestamp DESC
                LIMIT %s
            """, (session_id, max_turns * 2))
            
            messages = list(cursor.fetchall())
            messages.reverse()  # Đảo lại thứ tự
            
            lines = []
            for msg in messages:
                prefix = "User" if msg['role'] == "user" else "Assistant"
                lines.append(f"{prefix}: {msg['content']}")
            
            return "\n".join(lines)
    
    def delete_messages(self, session_id: str) -> int:
        """Xóa tất cả messages của session"""
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM messages WHERE session_id = %s",
                (session_id,)
            )
            return cursor.rowcount
    
    def count_messages(self, session_id: str) -> int:
        """Đếm số messages trong session"""
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM messages WHERE session_id = %s",
                (session_id,)
            )
            result = cursor.fetchone()
            return result['count'] if result else 0


# ================================================================
# USER OPERATIONS
# ================================================================

class UserRepository:
    """Repository pattern cho User operations"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_user(self, email: str, password_hash: str, full_name: str,
                    role: str = 'user') -> Optional[int]:
        """Tạo user mới, trả về user_id nếu thành công"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO users (email, password_hash, full_name, role)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (email, password_hash, full_name, role))
                result = cursor.fetchone()
                return result['id'] if result else None
        except psycopg2.IntegrityError:
            # Email đã tồn tại
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Lấy user theo ID"""
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM users WHERE id = %s AND is_active = TRUE",
                (user_id,)
            )
            return cursor.fetchone()
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Lấy user theo email (dùng cho login)"""
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM users WHERE email = %s",
                (email,)
            )
            return cursor.fetchone()
    
    def update_user(self, user_id: int, **kwargs) -> bool:
        """Cập nhật thông tin user"""
        if not kwargs:
            return False
        
        # Build dynamic UPDATE query
        set_clauses = []
        values = []
        
        allowed_fields = ['full_name', 'role', 'is_active', 'password_hash']
        for field, value in kwargs.items():
            if field in allowed_fields:
                set_clauses.append(f"{field} = %s")
                values.append(value)
        
        if not set_clauses:
            return False
        
        values.append(user_id)
        
        with self.db.get_cursor() as cursor:
            cursor.execute(f"""
                UPDATE users SET {', '.join(set_clauses)}
                WHERE id = %s
            """, tuple(values))
            return cursor.rowcount > 0
    
    def update_last_login(self, user_id: int) -> None:
        """Cập nhật thời gian đăng nhập cuối"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (user_id,)
                )
        except Exception as e:
            # Bỏ qua nếu có lỗi
            pass
    
    def delete_user(self, user_id: int) -> bool:
        """Xóa user (soft delete - set is_active = FALSE)"""
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "UPDATE users SET is_active = FALSE WHERE id = %s",
                (user_id,)
            )
            return cursor.rowcount > 0
    
    def list_users(self, limit: int = 50, include_inactive: bool = False) -> List[Dict]:
        """Lấy danh sách users"""
        with self.db.get_cursor() as cursor:
            if include_inactive:
                cursor.execute("""
                    SELECT id, email, full_name, role, is_active, 
                           last_login_at, created_at, updated_at
                    FROM users ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT id, email, full_name, role, is_active,
                           last_login_at, created_at, updated_at
                    FROM users WHERE is_active = TRUE
                    ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            return cursor.fetchall()
    
    def email_exists(self, email: str) -> bool:
        """Kiểm tra email đã tồn tại chưa"""
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM users WHERE email = %s LIMIT 1",
                (email,)
            )
            return cursor.fetchone() is not None


# ================================================================
# SINGLETON INSTANCES
# ================================================================

# Tạo instances
db = Database()
session_repo = SessionRepository(db)
message_repo = MessageRepository(db)
user_repo = UserRepository(db)
