"""
RAG ChatBot - Streamlit Interface

Giao diện web đẹp cho RAG ChatBot sử dụng Streamlit.
Tích hợp Authentication (Login/Register).

Usage:
    streamlit run app/streamlit_app.py
    hoặc
    python run.py --mode streamlit
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.config import settings

# API Base URL
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="RAG ChatBot - ABC Corp",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    
    /* Source citation - FIX: dark text on light background */
    .source-box {
        background-color: #fff3e0;
        padding: 0.75rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        border-left: 3px solid #ff9800;
        color: #333333 !important;
    }
    
    .source-box strong {
        color: #e65100 !important;
    }
    
    .source-box em {
        color: #555555 !important;
    }
    
    /* Metrics cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Header */
    .header-container {
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve chat input */
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    
    /* Fix expander text color */
    .streamlit-expanderContent {
        color: #333333 !important;
    }
    
    .streamlit-expanderContent p,
    .streamlit-expanderContent div,
    .streamlit-expanderContent span {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================

def init_session_state():
    """Initialize session state variables"""
    # Auth state
    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"  # login or register
    
    # Chat state
    if "session_id" not in st.session_state:
        st.session_state.session_id = None  # Sẽ được tạo sau khi đăng nhập
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    
    if "grounded_queries" not in st.session_state:
        st.session_state.grounded_queries = 0
    
    if "total_latency" not in st.session_state:
        st.session_state.total_latency = 0
    
    # Dev mode toggle (ẩn mặc định)
    if "dev_mode" not in st.session_state:
        st.session_state.dev_mode = False

init_session_state()


# ================================================================
# AUTH HELPER FUNCTIONS
# ================================================================

def get_auth_headers() -> Dict[str, str]:
    """Get authorization headers với JWT token"""
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


def api_request(method: str, endpoint: str, json_data: dict = None, 
                require_auth: bool = True, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """
    Helper function để gọi API với authentication.
    
    Returns: Response JSON hoặc None nếu lỗi
    """
    url = f"{API_URL}{endpoint}"
    headers = get_auth_headers() if require_auth else {}
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, json=json_data, headers=headers, timeout=timeout)
        elif method.upper() == "PUT":
            response = requests.put(url, json=json_data, headers=headers, timeout=timeout)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers, timeout=timeout)
        else:
            return None
        
        # Handle 401 - Token expired/invalid
        if response.status_code == 401:
            st.session_state.access_token = None
            st.session_state.current_user = None
            st.error("Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.")
            st.rerun()
            return None
        
        # Handle other errors
        if response.status_code >= 400:
            error_detail = response.json().get("detail", f"Lỗi {response.status_code}")
            st.error(f" {error_detail}")
            return None
        
        return response.json()
        
    except requests.exceptions.ConnectionError:
        st.error("Không thể kết nối đến API server. Vui lòng đảm bảo FastAPI đang chạy.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timeout. Vui lòng thử lại.")
        return None
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        return None


def login(email: str, password: str) -> bool:
    """Đăng nhập user"""
    try:
        response = requests.post(
            f"{API_URL}/auth/login",
            json={"email": email, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.current_user = data["user"]
            # Tạo session_id cho user
            user_id = data["user"]["id"]
            st.session_state.session_id = f"user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return True
        else:
            error_msg = response.json().get("detail", "Đăng nhập thất bại")
            st.error(f"{error_msg}")
            return False
            
    except requests.exceptions.ConnectionError:
        st.error(" Không thể kết nối đến API server")
        return False
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        return False


def register(email: str, password: str, full_name: str) -> bool:
    """Đăng ký user mới"""
    try:
        response = requests.post(
            f"{API_URL}/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": full_name
            },
            timeout=10
        )
        
        if response.status_code in [200, 201]:
            data = response.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.current_user = data["user"]
            # Tạo session_id cho user
            user_id = data["user"]["id"]
            st.session_state.session_id = f"user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.success("Đăng ký thành công!")
            return True
        else:
            error_msg = response.json().get("detail", "Đăng ký thất bại")
            st.error(f" {error_msg}")
            return False
            
    except requests.exceptions.ConnectionError:
        st.error("Không thể kết nối đến API server")
        return False
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        return False


def logout():
    """Đăng xuất"""
    st.session_state.access_token = None
    st.session_state.current_user = None
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.total_queries = 0
    st.session_state.grounded_queries = 0
    st.session_state.total_latency = 0


def is_authenticated() -> bool:
    """Kiểm tra user đã đăng nhập chưa"""
    return st.session_state.access_token is not None and st.session_state.current_user is not None


# ================================================================
# AUTH UI
# ================================================================

def render_login_page():
    """Render trang đăng nhập"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>ABC Corp RAG ChatBot</h1>
        <p style="color: #666;">Đăng nhập để sử dụng hệ thống hỏi đáp thông minh</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Đăng nhập")
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="email@example.com")
            password = st.text_input("Mật khẩu", type="password", placeholder="••••••••")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submitted = st.form_submit_button("Đăng nhập", use_container_width=True, type="primary")
            with col_btn2:
                if st.form_submit_button("Chưa có tài khoản?", use_container_width=True):
                    st.session_state.auth_page = "register"
                    st.rerun()
            
            if submitted:
                if not email or not password:
                    st.error("Vui lòng nhập đầy đủ thông tin")
                elif login(email, password):
                    st.success("Đăng nhập thành công!")
                    time.sleep(0.5)
                    st.rerun()


def render_register_page():
    """Render trang đăng ký"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>ABC Corp RAG ChatBot</h1>
        <p style="color: #666;">Tạo tài khoản mới</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Đăng ký")
        
        with st.form("register_form"):
            full_name = st.text_input("Họ và tên", placeholder="Nguyễn Văn A")
            email = st.text_input("Email", placeholder="email@example.com")
            password = st.text_input("Mật khẩu", type="password", placeholder="Tối thiểu 6 ký tự")
            password_confirm = st.text_input("Xác nhận mật khẩu", type="password", placeholder="••••••••")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submitted = st.form_submit_button("Đăng ký", use_container_width=True, type="primary")
            with col_btn2:
                if st.form_submit_button("Đã có tài khoản?", use_container_width=True):
                    st.session_state.auth_page = "login"
                    st.rerun()
            
            if submitted:
                if not full_name or not email or not password:
                    st.error("Vui lòng nhập đầy đủ thông tin")
                elif len(password) < 6:
                    st.error("Mật khẩu phải có ít nhất 6 ký tự")
                elif password != password_confirm:
                    st.error("Mật khẩu xác nhận không khớp")
                elif register(email, password, full_name):
                    time.sleep(0.5)
                    st.rerun()


# ================================================================
# SIDEBAR
# ================================================================

def render_sidebar():
    """Render sidebar với settings và stats"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60?text=ABC+Corp", width=200)
        st.title("RAG ChatBot")
        st.caption("Hệ thống hỏi đáp thông minh - Công ty ABC")
        
        st.divider()
        
        # ==================== USER INFO ====================
        if is_authenticated():
            user = st.session_state.current_user
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin-bottom: 1rem;
            ">
                <div style="font-size: 0.9rem; opacity: 0.9;">Xin chào,</div>
                <div style="font-size: 1.1rem; font-weight: bold;">{user.get('full_name', user.get('email'))}</div>
                <div style="font-size: 0.8rem; opacity: 0.8;">{user.get('email')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Đăng xuất", use_container_width=True):
                logout()
                st.rerun()
            
            st.divider()
        
        # ==================== CHAT HISTORY ====================
        st.subheader("Chat History")
        
        # Lấy danh sách sessions từ API (với auth)
        sessions_data = api_request("GET", "/sessions", require_auth=True, timeout=5)
        
        if sessions_data:
            sessions = sessions_data.get("sessions", [])
            
            if sessions:
                for sess in sessions[:10]:  # Hiển thị 10 sessions gần nhất
                    # Highlight session hiện tại
                    is_current = sess["session_id"] == st.session_state.session_id
                    icon = "💬" if is_current else "📝"
                    
                    # Button để chuyển session
                    btn_label = f"{icon} {sess['title']}"
                    if st.button(btn_label, key=f"sess_{sess['session_id']}", 
                               use_container_width=True,
                               type="primary" if is_current else "secondary"):
                        if not is_current:
                            # Chuyển sang session đã chọn
                            st.session_state.session_id = sess["session_id"]
                            
                            # Load messages từ API
                            history_data = api_request(
                                "GET", 
                                f"/session/{sess['session_id']}/history",
                                require_auth=True,
                                timeout=5
                            )
                            
                            if history_data:
                                st.session_state.messages = history_data.get("messages", [])
                            else:
                                st.session_state.messages = []
                            
                            st.rerun()
            else:
                st.caption("Chưa có lịch sử chat")
        
        st.divider()
        
        # ==================== SESSION INFO ====================
        # Chỉ hiển thị khi Dev Mode bật
        if st.session_state.dev_mode:
            st.subheader("Current Session")
            st.text(f"ID: {st.session_state.session_id[-15:]}")
            st.text(f"Messages: {len(st.session_state.messages)}")
            st.divider()
        
        # Statistics - Ẩn mặc định
        if st.session_state.dev_mode:
            st.subheader("Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Queries", st.session_state.total_queries)
            with col2:
                grounded_rate = 0
                if st.session_state.total_queries > 0:
                    grounded_rate = (st.session_state.grounded_queries / st.session_state.total_queries) * 100
                st.metric("Grounded", f"{grounded_rate:.0f}%")
            
            avg_latency = 0
            if st.session_state.total_queries > 0:
                avg_latency = st.session_state.total_latency / st.session_state.total_queries
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")
            
            st.divider()
        
        # Settings
        st.subheader("Settings")
        
        show_sources = st.checkbox("Show Sources", value=True)
        # Show Confidence Scores - Chỉ hiện trong Dev Mode
        if st.session_state.dev_mode:
            show_scores = st.checkbox("Show Confidence Scores", value=True)
            show_latency = st.checkbox("Show Latency", value=True)
        else:
            show_scores = False
            show_latency = False
        
        st.divider()
        
        # Actions
        st.subheader("Actions")
        
        if st.button("Clear Chat", use_container_width=True):
            if st.session_state.session_id:
                result = api_request(
                    "DELETE", 
                    f"/session/{st.session_state.session_id}",
                    require_auth=True,
                    timeout=5
                )
                if result:
                    st.session_state.messages = []
                    st.rerun()
                else:
                    # Vẫn clear local nếu API fail
                    st.session_state.messages = []
                    st.rerun()
        
        if st.button("New Session", use_container_width=True):
            # Tạo session mới với user_id
            if is_authenticated():
                user_id = st.session_state.current_user["id"]
                st.session_state.session_id = f"user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                st.session_state.session_id = f"guest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.session_state.grounded_queries = 0
            st.session_state.total_latency = 0
            st.rerun()
        
        st.divider()
        
        # System Health - Ẩn mặc định
        if st.session_state.dev_mode:
            st.subheader("System Health")
            
            try:
                api_url = os.getenv("API_URL", "http://localhost:8000")
                health_response = requests.get(f"{api_url}/health", timeout=5)
                
                if health_response.status_code == 200:
                    health = health_response.json()
                    
                    if health["qdrant_connected"]:
                        st.success(f"Qdrant: Connected ({health.get('vectors_count', 'N/A')} vectors)")
                    else:
                        st.error("Qdrant: Not Connected")
                    
                    if health["db_connected"]:
                        st.success("MySQL: Connected")
                    else:
                        st.error("MySQL: Not Connected")
                    
                    st.info(f"Model: {health.get('embedding_model', 'N/A')}")
                    st.info(f"Status: {health.get('status', 'unknown')}")
                else:
                    st.error(f"API Error: {health_response.status_code}")
            except:
                st.error("Cannot connect to API server")
            
            st.divider()
        
        # Dev Mode Toggle (ở cuối sidebar)
        st.markdown("---")
        dev_mode = st.checkbox("🔧 Developer Mode", value=st.session_state.dev_mode)
        if dev_mode != st.session_state.dev_mode:
            st.session_state.dev_mode = dev_mode
            st.rerun()
        
        return show_sources, show_scores, show_latency


# ================================================================
# CHAT INTERFACE
# ================================================================

def render_message(role: str, content: str, sources: list = None, 
                   latency: float = None, is_grounded: bool = True,
                   show_sources: bool = True, show_scores: bool = True,
                   show_latency: bool = True):
    """Render a chat message"""
    
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            
            # Show grounded status
            if not is_grounded:
                st.warning("Câu trả lời không dựa trên tài liệu nội bộ")
            
            # Show latency
            if show_latency and latency:
                st.caption(f"{latency:.0f}ms")
        
        # Show sources OUTSIDE chat_message to allow expanders
        if show_sources and sources:
            with st.expander(f"📚 Nguồn tham khảo ({len(sources)})", expanded=False):
                for i, src in enumerate(sources):
                    # Lấy thông tin
                    source_name = src.get('source', 'Unknown')
                    score = src.get('score', 0)
                    full_content = src.get('full_content', '') or src.get('excerpt', '')
                    page = src.get('page')
                    chunk_id = src.get('chunk_id', i)
                    
                    # Tên hiển thị đẹp
                    display_name = source_name.replace("_", " ").replace(".pdf", "").replace(".md", "")
                    
                    # Thông tin vị trí
                    location_parts = []
                    if page:
                        location_parts.append(f"Trang {page}")
                    location_parts.append(f"Chunk #{chunk_id}")
                    location_info = " • ".join(location_parts)
                    
                    # Score text
                    score_text = f" (Score: {score:.2f})" if show_scores else ""
                    
                    # Header
                    st.markdown(f"**{display_name}**{score_text}")
                    st.caption(location_info)
                    
                    # Nội dung đầy đủ trong container cuộn được
                    if full_content:
                        # Escape HTML characters
                        safe_content = full_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        st.markdown(f"""
                        <div style="
                            background: #f8f9fa;
                            border: 1px solid #e9ecef;
                            border-radius: 8px;
                            padding: 12px;
                            margin: 8px 0 16px 0;
                            max-height: 250px;
                            overflow-y: auto;
                            font-size: 0.9em;
                            line-height: 1.6;
                            white-space: pre-wrap;
                            word-wrap: break-word;
                        ">{safe_content}</div>
                        """, unsafe_allow_html=True)
                    
                    st.divider()


def process_query(question: str):
    """Process user query and get response via FastAPI (với authentication)"""
    
    try:
        # Call FastAPI endpoint với auth headers
        headers = get_auth_headers()
        
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "question": question,
                "session_id": st.session_state.session_id or "default"
            },
            headers=headers,
            timeout=60
        )
        
        # Handle 401 - Token expired
        if response.status_code == 401:
            st.session_state.access_token = None
            st.session_state.current_user = None
            st.error("Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.")
            st.rerun()
            return "Phiên đăng nhập đã hết hạn.", [], False, 0
        
        # Check response
        if response.status_code != 200:
            error_msg = response.json().get("detail", f"Lỗi {response.status_code}")
            st.error(f"{error_msg}")
            return "Xin lỗi, có lỗi khi xử lý yêu cầu của bạn.", [], False, 0
        
        data = response.json()
        
        # Extract data
        answer = data["answer"]
        sources_dict = data["sources"]
        is_grounded = data["is_grounded"]
        latency_ms = data["meta"]["latency_ms"]
        
        # Cập nhật session_id từ response (server có thể tạo mới)
        if data.get("session_id"):
            st.session_state.session_id = data["session_id"]
        
        # Update stats
        st.session_state.total_queries += 1
        st.session_state.total_latency += latency_ms
        if is_grounded:
            st.session_state.grounded_queries += 1
        
        return answer, sources_dict, is_grounded, latency_ms
        
    except requests.exceptions.ConnectionError:
        st.error("Không thể kết nối đến API server. Vui lòng đảm bảo FastAPI đang chạy.")
        return "Lỗi kết nối đến server.", [], False, 0
    except requests.exceptions.Timeout:
        st.error("Request timeout. Vui lòng thử lại.")
        return "Timeout khi xử lý câu hỏi.", [], False, 0
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        return f"Có lỗi xảy ra: {str(e)}", [], False, 0


# ================================================================
# MAIN APP
# ================================================================

def main():
    """Main application"""
    
    # Check authentication
    if not is_authenticated():
        # Show login or register page
        if st.session_state.auth_page == "register":
            render_register_page()
        else:
            render_login_page()
        return
    
    # User đã đăng nhập - render chat interface
    
    # Render sidebar and get settings
    show_sources, show_scores, show_latency = render_sidebar()
    
    # Header
    user_name = st.session_state.current_user.get("full_name", "User")
    st.markdown(f"""
    <div class="header-container">
        <h1>ABC Corp RAG ChatBot</h1>
        <p>Xin chào <strong>{user_name}</strong>! Hỏi đáp về chính sách nhân sự, quy trình nghiệp vụ, IT & bảo mật</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = [
        ("Nghỉ phép năm", "Nhân viên chính thức được nghỉ phép năm bao nhiêu ngày?"),
        ("Giờ làm việc", "Giờ làm việc chuẩn của công ty là gì?"),
        ("Mật khẩu", "Chính sách mật khẩu của công ty là gì?"),
        ("Onboarding", "Quy trình onboarding nhân viên mới như thế nào?"),
    ]
    
    for col, (label, question) in zip([col1, col2, col3, col4], quick_questions):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.pending_question = question
    
    st.divider()
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for msg in st.session_state.messages:
            render_message(
                role=msg["role"],
                content=msg["content"],
                sources=msg.get("sources"),
                latency=msg.get("latency"),
                is_grounded=msg.get("is_grounded", True),
                show_sources=show_sources,
                show_scores=show_scores,
                show_latency=show_latency
            )
    
    # Check for pending quick question
    if hasattr(st.session_state, 'pending_question') and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        # Process and add response
        with st.spinner("Đang xử lý..."):
            answer, sources, is_grounded, latency = process_query(question)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "latency": latency,
            "is_grounded": is_grounded
        })
        
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message immediately
        with chat_container:
            render_message("user", prompt)
        
        # Process query
        with st.spinner("Đang xử lý..."):
            answer, sources, is_grounded, latency = process_query(prompt)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "latency": latency,
            "is_grounded": is_grounded
        })
        
        # Rerun to display the new message
        st.rerun()
    
    # Footer
    st.divider()
    st.caption("RAG ChatBot")


def about_page():
    """About page"""
    st.title("About RAG ChatBot")
    
    st.markdown("""
    ## ABC Corp RAG ChatBot
    
    Hệ thống chatbot thông minh sử dụng **Retrieval-Augmented Generation (RAG)** 
    để trả lời câu hỏi về chính sách nhân sự, quy trình nghiệp vụ, IT & bảo mật.
    """)


# ================================================================
# RUN APP
# ================================================================

if __name__ == "__main__":
    # Simple navigation using query params
    page = st.query_params.get("page", "chat")
    
    if page == "about":
        about_page()
    else:
        main()
