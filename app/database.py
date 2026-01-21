"""
MySQL Database Module
Quản lý kết nối và operations với MySQL
"""

import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import json
from datetime import datetime

from app.config import settings

class Database:
    """
    MySQL Database Manager
    
    Features:
        - Connection pooling (đơn giản)
        - Auto-reconnect
        - Context manager for transactions
    """
    
    def __init__(self):
        """Initialize database connection"""
        self._connection = None
        self._connect()
        self._create_tables()
        print(" MySQL Database initialized")
    
    def _connect(self) -> None:
        """Tạo kết nối đến MySQL"""
        try:
            self._connection = pymysql.connect(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD or "",
                database=settings.MYSQL_DATABASE,
                charset='utf8mb4',
                cursorclass=DictCursor,
                autocommit=True
            )
            print(f"  ✓ Connected to MySQL at {settings.MYSQL_HOST}:{settings.MYSQL_PORT}")
        except pymysql.err.OperationalError as e:
            # Database không tồn tại, tạo mới
            if e.args[0] == 1049:
                self._create_database()
                self._connect()
            else:
                raise e
    
    def _create_database(self) -> None:
        """Tạo database nếu chưa tồn tại"""
        conn = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD or "",
            charset='utf8mb4'
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS {settings.MYSQL_DATABASE} "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            print(f"  ✓ Created database: {settings.MYSQL_DATABASE}")
        finally:
            conn.close()
    
    def _create_tables(self) -> None:
        """Tạo các bảng cần thiết"""
        
        # Bảng sessions
        create_sessions = """
        CREATE TABLE IF NOT EXISTS sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100) UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_session_id (session_id),
            INDEX idx_updated_at (updated_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        # Bảng messages
        create_messages = """
        CREATE TABLE IF NOT EXISTS messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL,
            role ENUM('user', 'assistant') NOT NULL,
            content TEXT NOT NULL,
            sources JSON,
            latency FLOAT,
            is_grounded BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session_id (session_id),
            INDEX idx_timestamp (timestamp),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(create_sessions)
            cursor.execute(create_messages)
        
        print("  ✓ Tables created/verified")
    
    @contextmanager
    def get_cursor(self):
        """Context manager để lấy cursor an toàn"""
        # Kiểm tra và reconnect nếu cần
        try:
            self._connection.ping(reconnect=True)
        except:
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
        """Test MySQL connection"""
        try:
            self._connection.ping(reconnect=True)
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
                    "INSERT INTO sessions (session_id) VALUES (%s)",
                    (session_id,)
                )
            return True
        except pymysql.err.IntegrityError:
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
                "UPDATE sessions SET updated_at = NOW() WHERE session_id = %s",
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
                GROUP BY s.session_id
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
            # Đảm bảo session tồn tại
            cursor.execute(
                "INSERT IGNORE INTO sessions (session_id) VALUES (%s)",
                (session_id,)
            )
            
            # Thêm message
            cursor.execute("""
                INSERT INTO messages 
                (session_id, role, content, sources, latency, is_grounded)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                session_id,
                role,
                content,
                json.dumps(sources, ensure_ascii=False) if sources else None,
                latency,
                is_grounded
            ))
            
            # Cập nhật session timestamp
            cursor.execute(
                "UPDATE sessions SET updated_at = NOW() WHERE session_id = %s",
                (session_id,)
            )
            
            return cursor.lastrowid
    
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
            
            # Parse JSON sources
            for msg in messages:
                if msg['sources']:
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
                """, (email, password_hash, full_name, role))
                return cursor.lastrowid
        except pymysql.err.IntegrityError:
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
        """Cập nhật thời gian đăng nhập cuối (safe - bỏ qua nếu cột không tồn tại)"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    "UPDATE users SET last_login_at = NOW() WHERE id = %s",
                    (user_id,)
                )
        except pymysql.err.OperationalError as e:
            # Bỏ qua nếu cột last_login_at chưa tồn tại
            if "Unknown column" in str(e):
                pass
            else:
                raise
    
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
            # Query không dùng last_login_at để tránh lỗi nếu cột chưa tồn tại
            if include_inactive:
                cursor.execute("""
                    SELECT id, email, full_name, role, is_active, 
                           created_at, updated_at
                    FROM users ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT id, email, full_name, role, is_active,
                           created_at, updated_at
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

