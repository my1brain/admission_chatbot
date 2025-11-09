import streamlit as st
import docx  # Make sure this is at the top of your file
import time
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from typing import Optional, List, Mapping, Any
import os
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel
from streamlit_mic_recorder import mic_recorder, speech_to_text

# Simple session auth helpers
def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None
    if 'username' not in st.session_state:
        st.session_state.username = None

def authenticate_user(username: str, password: str, user_type: str) -> bool:
    demo_users = {
        'student': {'demo_student': 'student123', 'john_doe': 'password'},
        'admin': {'admin': 'admin123', 'university_admin': 'secure456'}
    }
    return demo_users.get(user_type, {}).get(username) == password




# def login_page():
#     st.markdown("""
#     <style>
    

#     .welcome-title { font-size: 32px; font-weight: 700; color:#1a365d; margin-bottom: 8px; letter-spacing:-0.3px; }
#     .user-type-selector { display:flex; background:#f7fafc; border-radius:16px; padding:4px; margin: 12px 0 20px 0; }
#     .user-type-option { flex:1; padding: 10px 14px; text-align:center; border-radius:8px; cursor:pointer; transition: all .2s ease; font-size:14px; font-weight:600; color:#4a5568; border:1px solid transparent; }
#     .user-type-option.active { background:#ffffff; color:#4299e1; border-color:#e2e8f0; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
#     .form-label { display:block; color:#4a5568; font-size:14px; font-weight:600; margin: 10px 0 6px 0; }
#     .login-button { width:100%; padding:14px; background:#4299e1 !important; color:#fff !important; border:none !important; border-radius:12px !important; font-size:16px !important; font-weight:700 !important; cursor:pointer; transition: all .2s ease; margin: 18px 0; }
#     .login-button:hover { background:#3182ce !important; transform: translateY(-1px); box-shadow: 0 8px 22px rgba(66,153,225,.28); }
#     .signup-link { color:#718096; font-size:14px; }
#     .signup-link a { color:#4299e1; text-decoration:none; font-weight:600; }
#     .signup-link a:hover { text-decoration:underline; }
#     .faq-item { display:flex; align-items:flex-start; gap:10px; margin: 8px 0 12px 0; }
#     .faq-icon { width:20px; height:20px; background:#4299e1; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#fff; font-size:12px; font-weight:800; }
#     .faq-content h4 { color:#1a365d; font-size:14px; font-weight:700; margin:0 0 4px 0; }
#     .faq-content p { color:#718096; font-size:13px; line-height:1.4; margin:0; }

#     .blob { position:absolute; width:320px; height:320px; background: radial-gradient(circle, rgba(102,126,234,.3) 0%, rgba(118,75,162,.1) 70%); border-radius: 50% 40% 60% 30%; animation: morph 8s infinite ease-in-out; top: 10%; left: 10%; }
#     @keyframes morph { 0%,100% { border-radius:50% 40% 60% 30%; transform: rotate(0deg) scale(1);} 25% { border-radius:30% 60% 40% 50%; transform: rotate(90deg) scale(1.1);} 50% { border-radius:60% 30% 50% 40%; transform: rotate(180deg) scale(.92);} 75% { border-radius:40% 50% 30% 60%; transform: rotate(270deg) scale(1.05);} }
#     .fluid-container { perspective: 1000px; width: 700px; height: 400px; display:flex; align-items:center; justify-content:center; margin: 0 auto; }
#     .cube { width: 200px; height: 400px; position: relative; transform-style: preserve-3d; animation: rotate 18s infinite linear; }
#     .cube-face { position:absolute; width:200px; height:200px; border-radius: 20px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); opacity:.85; filter: drop-shadow(0 10px 24px rgba(0,0,0,.15)); }
#     .front { transform: rotateY(0deg) translateZ(100px);} .back { transform: rotateY(180deg) translateZ(100px);} .right { transform: rotateY(90deg) translateZ(100px);} .left { transform: rotateY(-90deg) translateZ(100px);} .top { transform: rotateX(90deg) translateZ(100px);} .bottom { transform: rotateX(-90deg) translateZ(100px);} 
#     @keyframes rotate { 0% { transform: rotateX(0deg) rotateY(0deg);} 100% { transform: rotateX(360deg) rotateY(360deg);} }
#     .particle { position:absolute; width:4px; height:4px; background:#4299e1; border-radius:50%; animation: float 6s infinite ease-in-out; }
#     @keyframes float { 0%,100% { transform: translateY(0px) scale(1); opacity:.7;} 50% { transform: translateY(-18px) scale(1.2); opacity:1; } }
#     </style>
#     """, unsafe_allow_html=True)

#     left, right = st.columns([1, 1])

#     with left:
#         st.markdown('<h1 class="welcome-title">Welcome to University Portal</h1>', unsafe_allow_html=True)
#         if 'login_user_type' not in st.session_state:
#             st.session_state.login_user_type = 'student'
#         btn_cols = st.columns(2)
#         with btn_cols[0]:
#             if st.button("🎓 Student", use_container_width=True):
#                 st.session_state.login_user_type = 'student'
#         with btn_cols[1]:
#             if st.button("⚙️ Admin", use_container_width=True):
#                 st.session_state.login_user_type = 'admin'
#         st.markdown(f"""
#         <div class=\"user-type-selector\">
#             <div class=\"user-type-option {'active' if st.session_state.login_user_type=='student' else ''}\">🎓 Student</div>
#             <div class=\"user-type-option {'active' if st.session_state.login_user_type=='admin' else ''}\">⚙️ Admin</div>
#         </div>
#         """, unsafe_allow_html=True)

#         with st.form("login_form"):
#             st.markdown('<label class="form-label">Email</label>', unsafe_allow_html=True)
#             username = st.text_input("", key="login_email", placeholder="student@university.edu" if st.session_state.login_user_type=='student' else "admin@university.edu")
#             st.markdown('<label class="form-label">Password</label>', unsafe_allow_html=True)
#             password = st.text_input("", key="login_password", placeholder="••••••••", type="password")
#             submit = st.form_submit_button("Login", use_container_width=True)
#             if submit:
#                 user_type = st.session_state.login_user_type
#                 if username and password:
#                     with st.spinner("🔄 Authenticating credentials..."):
#                         time.sleep(1.0)
#                     if authenticate_user(username, password, user_type):
#                         st.session_state.authenticated = True
#                         st.session_state.user_type = user_type
#                         st.session_state.username = username
#                         st.success("✅ Login successful! Redirecting...")
#                         time.sleep(0.6)
#                         st.rerun()
#                     else:
#                         st.error("❌ Invalid credentials. Please try again.")
#                 else:
#                     st.warning("⚠️ Please fill in all fields.")

#         st.markdown("""
#         <div class=\"faq-item\">
#             <div class=\"faq-icon\">?</div>
#             <div class=\"faq-content\">
#                 <h4>Need help accessing your account?</h4>
#                 <p>Contact support or use the forgot password option.</p>
#             </div>
#         </div>
#         <div class=\"signup-link\">Don't have an account? <a href=\"#\">Sign Up!</a></div>
#         """, unsafe_allow_html=True)

#     with right:
#         # Try to load a logo to overlay on cube faces
#         logo_data_uri = None
#         try:
#             base_dir = os.path.dirname(__file__)
#             for name in [
#                 "logo.png", "logo.jpg", "logo.jpeg",
#                 "mit_adt_logo.png", "mit_adt_logo.jpg", "mit_adt_logo.jpeg", "mit_adt_logo.svg"
#             ]:
#                 p = os.path.join(base_dir, name)
#                 if os.path.exists(p):
#                     import base64
#                     with open(p, "rb") as f:
#                         b64 = base64.b64encode(f.read()).decode()
#                     mime = "image/svg+xml" if p.endswith(".svg") else ("image/png" if p.endswith(".png") else "image/jpeg")
#                     logo_data_uri = f"data:{mime};base64,{b64}"
#                     break
#         except Exception:
#             logo_data_uri = None

#         st.markdown('<div class="right-panel">', unsafe_allow_html=True)
#         st.markdown('<div class="blob"></div>', unsafe_allow_html=True)
#         img_tag = f"<img src='{logo_data_uri}' style=\"width:100%;height:100%;object-fit:cover;border-radius:20px;opacity:.95;\"/>" if logo_data_uri else ""
#         st.markdown(f"""
#         <style>
#         .cube-face img {{ pointer-events:none; }}
#         </style>
#         <div class=\"fluid-container\">
#             <div class=\"cube\">
#                 <div class=\"cube-face front\">{img_tag}</div>
#                 <div class=\"cube-face back\">{img_tag}</div>
#                 <div class=\"cube-face right\">{img_tag}</div>
#                 <div class=\"cube-face left\">{img_tag}</div>
#                 <div class=\"cube-face top\">{img_tag}</div>
#                 <div class=\"cube-face bottom\">{img_tag}</div>
#             </div>
#         </div>
#         <div class=\"particle\" style=\"top:20%;left:30%;animation-delay:0s;\"></div>
#         <div class=\"particle\" style=\"top:40%;left:70%;animation-delay:1s;\"></div>
#         <div class=\"particle\" style=\"top:60%;left:20%;animation-delay:2s;\"></div>
#         <div class=\"particle\" style=\"top:80%;left:80%;animation-delay:3s;\"></div>
#         <div class=\"particle\" style=\"top:30%;left:50%;animation-delay:4s;\"></div>
#         """, unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)


def login_page():
    st.markdown("""
    <style>
    

    .welcome-title { font-size: 32px; font-weight: 700; color:#1a365d; margin-bottom: 8px; letter-spacing:-0.3px; }
    .user-type-selector { display:flex; background:#f7fafc; border-radius:16px; padding:4px; margin: 12px 0 20px 0; }
    .user-type-option { flex:1; padding: 10px 14px; text-align:center; border-radius:8px; cursor:pointer; transition: all .2s ease; font-size:14px; font-weight:600; color:#4a5568; border:1px solid transparent; }
    .user-type-option.active { background:#ffffff; color:#4299e1; border-color:#e2e8f0; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
    .form-label { display:block; color:#4a5568; font-size:14px; font-weight:600; margin: 10px 0 6px 0; }
    .login-button { width:100%; padding:14px; background:#4299e1 !important; color:#fff !important; border:none !important; border-radius:12px !important; font-size:16px !important; font-weight:700 !important; cursor:pointer; transition: all .2s ease; margin: 18px 0; }
    .login-button:hover { background:#3182ce !important; transform: translateY(-1px); box-shadow: 0 8px 22px rgba(66,153,225,.28); }
    .signup-link { color:#718096; font-size:14px; }
    .signup-link a { color:#4299e1; text-decoration:none; font-weight:600; }
    .signup-link a:hover { text-decoration:underline; }
    .faq-item { display:flex; align-items:flex-start; gap:10px; margin: 8px 0 12px 0; }
    .faq-icon { width:20px; height:20px; background:#4299e1; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#fff; font-size:12px; font-weight:800; }
    .faq-content h4 { color:#1a365d; font-size:14px; font-weight:700; margin:0 0 4px 0; }
    .faq-content p { color:#718096; font-size:13px; line-height:1.4; margin:0; }

    .blob { position:absolute; width:320px; height:320px; background: radial-gradient(circle, rgba(102,126,234,.3) 0%, rgba(118,75,162,.1) 70%); border-radius: 50% 40% 60% 30%; animation: morph 8s infinite ease-in-out; top: 10%; left: 10%; }
    @keyframes morph { 0%,100% { border-radius:50% 40% 60% 30%; transform: rotate(0deg) scale(1);} 25% { border-radius:30% 60% 40% 50%; transform: rotate(90deg) scale(1.1);} 50% { border-radius:60% 30% 50% 40%; transform: rotate(180deg) scale(.92);} 75% { border-radius:40% 50% 30% 60%; transform: rotate(270deg) scale(1.05);} }
    .fluid-container { perspective: 1000px; width: 700px; height: 400px; display:flex; align-items:center; justify-content:center; margin: 0 auto; }
    .cube { width: 200px; height: 400px; position: relative; transform-style: preserve-3d; animation: rotate 18s infinite linear; }
    .cube-face { position:absolute; width:200px; height:200px; border-radius: 20px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); opacity:.85; filter: drop-shadow(0 10px 24px rgba(0,0,0,.15)); }
    .front { transform: rotateY(0deg) translateZ(100px);} .back { transform: rotateY(180deg) translateZ(100px);} .right { transform: rotateY(90deg) translateZ(100px);} .left { transform: rotateY(-90deg) translateZ(100px);} .top { transform: rotateX(90deg) translateZ(100px);} .bottom { transform: rotateX(-90deg) translateZ(100px);} 
    @keyframes rotate { 0% { transform: rotateX(0deg) rotateY(0deg);} 100% { transform: rotateX(360deg) rotateY(360deg);} }
    .particle { position:absolute; width:4px; height:4px; background:#4299e1; border-radius:50%; animation: float 6s infinite ease-in-out; }
    @keyframes float { 0%,100% { transform: translateY(0px) scale(1); opacity:.7;} 50% { transform: translateY(-18px) scale(1.2); opacity:1; } }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<h1 class="welcome-title">Welcome to University Portal</h1>', unsafe_allow_html=True)
        if 'login_user_type' not in st.session_state:
            st.session_state.login_user_type = 'student'
        
        # Keep your exact styled panel and make it clickable
        st.markdown(f"""
        <div class=\"user-type-selector\">
            <div class=\"user-type-option {'active' if st.session_state.login_user_type=='student' else ''}\" 
                 onclick=\"document.getElementById('student-btn').click()\">🎓 Student</div>
            <div class=\"user-type-option {'active' if st.session_state.login_user_type=='admin' else ''}\" 
                 onclick=\"document.getElementById('admin-btn').click()\">⚙️ Admin</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hidden buttons for functionality only (completely invisible)
        if st.button("", key="student-btn", help="", type="primary"):
            st.session_state.login_user_type = 'student'
            st.rerun()
        if st.button("", key="admin-btn", help="", type="primary"):
            st.session_state.login_user_type = 'admin'
            st.rerun()
            
        # Hide the functional buttons completely
        st.markdown("""
        <style>
        button[kind="primary"] {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown('<label class="form-label">Email</label>', unsafe_allow_html=True)
            username = st.text_input("", key="login_email", placeholder="student@university.edu" if st.session_state.login_user_type=='student' else "admin@university.edu")
            st.markdown('<label class="form-label">Password</label>', unsafe_allow_html=True)
            password = st.text_input("", key="login_password", placeholder="••••••••", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            if submit:
                user_type = st.session_state.login_user_type
                if username and password:
                    with st.spinner("🔄 Authenticating credentials..."):
                        time.sleep(1.0)
                    if authenticate_user(username, password, user_type):
                        st.session_state.authenticated = True
                        st.session_state.user_type = user_type
                        st.session_state.username = username
                        st.success("✅ Login successful! Redirecting...")
                        time.sleep(0.6)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
                else:
                    st.warning("⚠️ Please fill in all fields.")

        st.markdown("""
        <div class=\"faq-item\">
            <div class=\"faq-icon\">?</div>
            <div class=\"faq-content\">
                <h4>Need help accessing your account?</h4>
                <p>Contact support or use the forgot password option.</p>
            </div>
        </div>
        <div class=\"signup-link\">Don't have an account? <a href=\"#\">Sign Up!</a></div>
        """, unsafe_allow_html=True)

    with right:
        # Try to load a logo to overlay on cube faces
        logo_data_uri = None
        try:
            base_dir = os.path.dirname(__file__)
            for name in [
                "logo.png", "logo.jpg", "logo.jpeg",
                "mit_adt_logo.png", "mit_adt_logo.jpg", "mit_adt_logo.jpeg", "mit_adt_logo.svg"
            ]:
                p = os.path.join(base_dir, name)
                if os.path.exists(p):
                    import base64
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    mime = "image/svg+xml" if p.endswith(".svg") else ("image/png" if p.endswith(".png") else "image/jpeg")
                    logo_data_uri = f"data:{mime};base64,{b64}"
                    break
        except Exception:
            logo_data_uri = None

        st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        st.markdown('<div class="blob"></div>', unsafe_allow_html=True)
        img_tag = f"<img src='{logo_data_uri}' style=\"width:100%;height:100%;object-fit:cover;border-radius:20px;opacity:.95;\"/>" if logo_data_uri else ""
        st.markdown(f"""
        <style>
        .cube-face img {{ pointer-events:none; }}
        </style>
        <div class=\"fluid-container\">
            <div class=\"cube\">
                <div class=\"cube-face front\">{img_tag}</div>
                <div class=\"cube-face back\">{img_tag}</div>
                <div class=\"cube-face right\">{img_tag}</div>
                <div class=\"cube-face left\">{img_tag}</div>
                <div class=\"cube-face top\">{img_tag}</div>
                <div class=\"cube-face bottom\">{img_tag}</div>
            </div>
        </div>
        <div class=\"particle\" style=\"top:20%;left:30%;animation-delay:0s;\"></div>
        <div class=\"particle\" style=\"top:40%;left:70%;animation-delay:1s;\"></div>
        <div class=\"particle\" style=\"top:60%;left:20%;animation-delay:2s;\"></div>
        <div class=\"particle\" style=\"top:80%;left:80%;animation-delay:3s;\"></div>
        <div class=\"particle\" style=\"top:30%;left:50%;animation-delay:4s;\"></div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)



# Logo helper
def _load_logo_data_uri() -> Optional[str]:
    try:
        base_dir = os.path.dirname(__file__)
        for name in ["logo.png", "logo.jpg", "logo.jpeg", "mit_adt_logo.png", "mit_adt_logo.jpg", "mit_adt_logo.jpeg", "mit_adt_logo.svg"]:
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                import base64
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                mime = "image/svg+xml" if p.endswith(".svg") else ("image/png" if p.endswith(".png") else "image/jpeg")
                return f"data:{mime};base64,{b64}"
    except Exception:
        pass
    return None

# Page config
st.set_page_config(
    page_title="MIT ADT Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#Custom styling for modern chatbot interface
st.markdown("""
    <style>
    :root {
        --mit-primary: #6E2D8C; /* MIT ADT purple tone */
        --mit-primary-light: #EDE6F4;
        --mit-gradient-start: #7A2FA1;
        --mit-gradient-end: #9536BE;
    }
    /* Main container */
    .main { 
        background: #F7F7FB; /* light mode */
        padding: 0;
        margin: 0;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(135deg, var(--mit-gradient-start) 0%, var(--mit-gradient-end) 100%);
        padding: 16px 20px;
        border-radius: 0 0 18px 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 18px;
        color: white;
    }
    
    .header h1 {
        color: #FFFFFF;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        text-align: left;
    }
    
    .header p {
        color: #7f8c8d;
        font-size: 1.1rem;
        text-align: center;
        margin: 10px 0 0 0;
        font-weight: 400;
    }
    
    /* Chat container */
    .chat-container {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px;
        margin: 20px auto;
        max-width: 900px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border: 1px solid #EFEFF5;
    }
    
    /* Chat messages */
    .stChatMessage {
        margin: 15px 0;
        border-radius: 18px;
        padding: 15px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: none;
    }
    
    /* User messages */
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, var(--mit-gradient-start) 0%, var(--mit-gradient-end) 100%);
        color: white;
        margin-left: 20%;
        border-radius: 18px 18px 5px 18px;
    }
    
    /* Assistant messages */
    .stChatMessage[data-testid="assistant"] {
        background: #FAFAFF;
        color: #2c3e50;
        margin-right: 20%;
        border-radius: 18px 18px 18px 5px;
        border-left: 4px solid var(--mit-primary);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 15px 20px;
        border: 2px solid #e9ecef;
        font-size: 16px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--mit-gradient-start) 0%, var(--mit-gradient-end) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(122, 47, 161, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(122, 47, 161, 0.35);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        text-align: center;
        color: #667eea;
    }
    
    /* Info/Error styling */
    .error {
        background: #FFECEC;
        color: #A33A3A;
        padding: 15px 20px;
        border-radius: 12px;
        margin: 10px 0;
        border: 1px solid #F7D6D6;
    }
    
    /* Status indicators */
    .status-online {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #2ecc71;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stToolbar"] {display:none;}
    </style>
""", unsafe_allow_html=True)

# Initialize session and gate content
init_session_state()

# If not authenticated, show login and stop further rendering
if not st.session_state.authenticated:
    login_page()
    st.stop()

# Header with logo
_logo_src = _load_logo_data_uri()
_logo_tag = f"<img src='{_logo_src}' style=\"height:48px;width:auto;display:block;\" alt='MIT ADT University logo'/>" if _logo_src else ""
st.markdown(f"""
    <div class="header">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
            <div style="display:flex;align-items:center;gap:12px;">
                {('<div class="logo-slot">' + _logo_tag + '</div>') if _logo_tag else ''}
                <div>
                    <h1>MIT ADT University • Assistant</h1>
                    <p style="margin:2px 0 0 0;color:#F2ECF8;">Multilingual: English • हिन्दी • मराठी</p>
                </div>
            </div>
            <p class="status" style="margin:0;color:#FFFFFF;"><span class="status-online"></span>Live • Ready to help</p>
        </div>
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    base_welcome = (
        "👋 **Welcome to MIT ADT University.**\n\n"
        "I’m your virtual assistant. I provide clear, professional guidance grounded in official information. "
        "Ask in English, Hindi, or Marathi — I’ll respond in the same language.\n\n"
        "I can help with:\n\n"
        "• Program details and eligibility\n"
        "• Fees, scholarships, and calculations\n"
        "• Hostel facilities and charges\n"
        "• Application timelines and process\n\n"
        "**How may I assist you today?**"
    )
    hi_welcome = (
        "👋 **MIT ADT University प्रवेश सहायक में आपका स्वागत है।**\n\n"
        "मैं आपका वर्चुअल एडमिशन असिस्टेंट हूँ। मैं आधिकारिक जानकारी के आधार पर स्पष्ट और पेशेवर मार्गदर्शन प्रदान करता/करती हूँ। "
        "आप अंग्रेज़ी, हिंदी या मराठी में पूछ सकते हैं — मैं उसी भाषा में उत्तर दूँगा/दूँगी।\n\n"
        "मैं इन विषयों में मदद कर सकता/सकती हूँ:\n\n"
        "• प्रोग्राम विवरण और पात्रता\n"
        "• फीस, छात्रवृत्ति और गणना\n"
        "• हॉस्टल सुविधाएँ और शुल्क\n"
        "• आवेदन समय-सीमा और प्रक्रिया\n\n"
        "**मैं आपकी किस प्रकार सहायता कर सकता/सकती हूँ?**"
    )
    mr_welcome = (
        "👋 **MIT ADT विद्यापीठ प्रवेश सहाय्यामध्ये आपले स्वागत आहे.**\n\n"
        "मी आपला व्हर्च्युअल अॅडमिशन असिस्टंट आहे. मी अधिकृत माहितीनुसार स्पष्ट आणि व्यावसायिक मार्गदर्शन देतो/देते. "
        "आपण English, हिंदी किंवा मराठीमध्ये विचारू शकता — मी त्याच भाषेत उत्तर देईन.\n\n"
        "मी या विषयांमध्ये मदत करू शकतो/शकते:\n\n"
        "• कार्यक्रम तपशील व पात्रता\n"
        "• फी, शिष्यवृत्ती व गणना\n"
        "• वसतिगृह सुविधा व शुल्क\n"
        "• अर्ज करण्याची वेळ व प्रक्रिया\n\n"
        "**मी आपली कशी मदत करू शकतो/शकते?**"
    )
    st.session_state.base_welcome = base_welcome
    st.session_state.welcome_texts = {"en": base_welcome, "hi": hi_welcome, "mr": mr_welcome}
    st.session_state.welcome_lang = "en"
    st.session_state.messages = [{"role": "assistant", "content": base_welcome}]
    st.session_state.stt_text = ""
    st.session_state.whisper_inited = False
    st.session_state.lang_pref = "auto"  # auto | en | hi | mr
    st.session_state.quick_prefill = ""

# Load and split handbook text


def load_text():
    base_dir = os.path.dirname(__file__)
    text_path = os.path.join(base_dir, "admission_fees.txt")
    if not os.path.exists(text_path):
        st.markdown("<p class='error'>⚠️ 'admission_fees.txt' not found in the project directory.</p>", unsafe_allow_html=True)
        return None
    try:
        with open(text_path, 'r', encoding='utf-8') as file:
            content = file.read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=5500, chunk_overlap=300)
        docs = splitter.split_documents([Document(page_content=content)])
        return docs
    except Exception as e:
        st.markdown(f"<p class='error'>Error loading fee text file: {str(e)}</p>", unsafe_allow_html=True)
        return None




# Language utilities

# Translation backends (IndicTrans2 primary, Google as fallback)
_hf_tokenizers: dict[str, AutoTokenizer] = {}
_hf_models: dict[str, AutoModelForSeq2SeqLM] = {}

# Map for available IndicTrans2 models on Hugging Face
INDICTRANS2_MODELS: dict[tuple[str, str], str] = {
    ("en", "hi"): "ai4bharat/indictrans2-en-hi",
    ("hi", "en"): "ai4bharat/indictrans2-hi-en",
    ("en", "mr"): "ai4bharat/indictrans2-en-mr",
    ("mr", "en"): "ai4bharat/indictrans2-mr-en",
}

def detect_language_or_default(text: str, default: str = "en") -> str:
    try:
        lang = detect(text)
        # Normalize to supported set
        return lang if lang in {"en", "hi", "mr"} else default
    except Exception:
        return default
# STT: Lazy-init Whisper
def get_whisper_model() -> WhisperModel | None:
    if st.session_state.get("whisper_inited", False):
        return st.session_state.get("whisper_model")
    # Try a few robust configurations (works on CPU and GPU)
    model_ids = ["small", "base"]
    compute_types = ["int8", "int8_float16", "float16", "float32"]
    last_error: Exception | None = None
    for m in model_ids:
        for ct in compute_types:
            try:
                wm = WhisperModel(m, device="auto", compute_type=ct)
                st.session_state.whisper_model = wm
                st.session_state.whisper_inited = True
                st.session_state.whisper_model_info = f"{m}/{ct}"
                return wm
            except Exception as e:
                last_error = e
                continue
    if last_error is not None:
        st.warning(f"Whisper init failed: {last_error}")
    return None

def _load_hf_model(model_id: str):
    if model_id not in _hf_tokenizers:
        _hf_tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
    if model_id not in _hf_models:
        _hf_models[model_id] = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return _hf_tokenizers[model_id], _hf_models[model_id]

def translate_indictrans2(text: str, src: str, dest: str) -> str:
    try:
        # Direct model available
        model_id = INDICTRANS2_MODELS.get((src, dest))
        if model_id is None:
            # Pivot via English if needed (e.g., hi <-> mr)
            if src != "en" and dest != "en":
                pivot = translate_indictrans2(text, src, "en")
                return translate_indictrans2(pivot, "en", dest)
            # No suitable path
            raise ValueError("No direct IndicTrans2 model for this language pair")

        tokenizer, model = _load_hf_model(model_id)

        # Some IndicTrans2 checkpoints accept natural text; some prefer prefix
        # We try plain text first; if decoding fails, fall back to prefixed prompt
        input_text = text
        try:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=512)
            out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            if out and out.strip():
                return out.strip()
        except Exception:
            pass

        # Fallback with explicit direction tag
        input_text = f"translate to {dest}: {text}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=512)
        out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return out.strip()
    except Exception:
        # Last resort: return original text so app does not break
        return text

def translate_text(text: str, dest: str, src: str | None = None) -> str:
    try:
        # Prefer IndicTrans2 for en/hi/mr
        if src is None:
            src = detect_language_or_default(text, default="en")
        if src in {"en", "hi", "mr"} and dest in {"en", "hi", "mr"} and src != dest:
            translated = translate_indictrans2(text, src=src, dest=dest)
            if translated and translated.strip() and translated != text:
                return translated
        # No supported translation path: return original
        return text
    except Exception:
        # If translation fails, fall back to original
        return text


# Vector store from documents
def create_vector_store(texts):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(texts, embeddings)
        return vector_store
    except Exception as e:
        st.markdown(f"<p class='error'>Error creating FAISS vector store: {str(e)}</p>", unsafe_allow_html=True)
        return None

# Initialize Ollama model (optimized for Mac M4)
def initialize_llm():
    try:
        # Using Ollama with Llama 3.1 8B - optimized for Mac M4
        return Ollama(
            model="llama3.1:8b",
            base_url="http://localhost:11434",
            temperature=0.1,  # Lower temperature for more consistent answers
            top_p=0.9,
            repeat_penalty=1.1
        )
    except Exception as e:
        st.markdown(f"<p class='error'>Error loading Ollama LLM: {str(e)}</p>", unsafe_allow_html=True)
        return None

# Setup RAG pipeline
def setup_rag_chain():
    try:
        texts = load_text()
        if texts is None:
            st.error("❌ Failed to load admission fees document. Please ensure 'admission_fees.txt' exists in the project directory.")
            return None
        
        vector_store = create_vector_store(texts)
        if vector_store is None:
            st.error("❌ Failed to create vector database. This might be due to insufficient memory or corrupted embeddings.")
            return None
            
        llm = initialize_llm()
        if llm is None:
            st.error("❌ Failed to initialize language model. Please ensure Ollama is running with llama3.1:8b model.")
            return None
    except Exception as e:
        st.error(f"❌ System initialization failed: {str(e)}")
        return None

    prompt_template = """You are a professional and empathetic University counselor. Use ONLY the context provided below to answer questions about admission fees, programs, and requirements. Be precise with numbers and calculations. Maintain a warm, reassuring tone. Respond in the user's language (English/Hindi/Marathi) based on the question.

Guidelines:
- Answer based ONLY on the provided context
- For fee calculations, show the math clearly
- If information is not in the context, say "I don't have that information in the available documents"
- Be helpful and professional
- Format your answers clearly with bullet points when listing multiple items
 - Keep responses concise and supportive; add empathetic phrasing where appropriate

Context:
{context}

Question:
{question}

Answer:"""


    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),  # More chunks for better context
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Init RAG
if "qa_chain" not in st.session_state:
    with st.spinner("🔄 Initializing chatbot..."):
        try:
            st.session_state.qa_chain = setup_rag_chain()
            if st.session_state.qa_chain is None:
                st.error("❌ Chatbot initialization failed. Please check the console for details and refresh the page.")
        except Exception as e:
            st.error(f"❌ Unexpected error during initialization: {str(e)}")
            st.session_state.qa_chain = None

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Sidebar: Fee calculator (lightweight helper for students)
with st.sidebar:
    if st.session_state.get("authenticated"):
        st.markdown(f"**Signed in as:** {st.session_state.get('username','')}  ")
        st.caption(f"Role: {st.session_state.get('user_type','')} • ✅ Authenticated")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_type = None
            st.session_state.username = None
            st.experimental_rerun()
    st.markdown("### 💰 Fee Calculator (Quick Estimate)")
    prog_fee = st.number_input("Tuition (per year)", min_value=0, step=1000, value=0)
    hostel_fee = st.number_input("Hostel (per year)", min_value=0, step=500, value=0)
    others_fee = st.number_input("Other fees (per year)", min_value=0, step=500, value=0)
    scholarship_pct = st.slider("Scholarship %", 0, 100, 0)
    gross = prog_fee + hostel_fee + others_fee
    net = int(round(gross * (100 - scholarship_pct) / 100))
    st.metric(label="Estimated Annual Total", value=f"₹ {net:,}", delta=f"Scholarship: {scholarship_pct}%")
    if st.button("Ask about this plan"):
        st.session_state.quick_prefill = (
            f"Please validate this fee estimate and provide breakdown: Tuition ₹{prog_fee}, Hostel ₹{hostel_fee}, Other ₹{others_fee}, Scholarship {scholarship_pct}%"
        )

# Language chips and Voice input controls
with st.container():
    cols = st.columns([1,1,2,2])
    with cols[0]:
        st.caption("🎙️ Voice Input")
        
        # Custom styled microphone button
        col1, col2, col3 = st.columns([1, 2, 1])
        # with col2:
            # Create a custom circular button with microphone icon
            # if st.button("🎤", key="mic_button", help="Click to start/stop recording", use_container_width=True, type="secondary"):
            #     st.session_state.mic_clicked = True
            #     st.rerun()
        
        # Use the original mic_recorder but with custom styling
        audio = mic_recorder(
            start_prompt="🎤 Start Recording", 
            stop_prompt="⏹️ Stop Recording", 
            just_once=True, 
            use_container_width=True,
            key="custom_mic"
        )
        
        if audio and audio.get("bytes"):
            with st.spinner("Transcribing audio..."):
                model = get_whisper_model()
                if model is not None:
                    # Save temp wav
                    tmp_path = os.path.join(os.path.dirname(__file__), "_tmp_query.wav")
                    with open(tmp_path, "wb") as f:
                        f.write(audio["bytes"])
                    # Let model auto-detect language; beam_size improves quality slightly
                    segments, _ = model.transcribe(tmp_path, language=None, beam_size=3)
                    transcript = " ".join([seg.text for seg in segments]).strip()
                    st.session_state.stt_text = transcript
                else:
                    st.warning("Whisper model not available. See warning above for details.")
    with cols[1]:
        st.caption("📝 Detected text")
        st.write(st.session_state.get("stt_text", ""))
    with cols[2]:
        st.caption("🌐 Language")
        chip_cols = st.columns(4)
        def chip(label, key, val):
            style = "background:#EDE6F4;color:#2c3e50;" if st.session_state.lang_pref!=val else "background:#7A2FA1;color:#fff;"
            if chip_cols[key].button(label, use_container_width=True):
                st.session_state.lang_pref = val
                # Update welcome paragraph immediately to chosen language (only the initial assistant message)
                target = "en" if val == "auto" else val
                if target in {"en","hi","mr"} and st.session_state.get("messages"):
                    current = st.session_state.get("welcome_lang", "en")
                    if target != current:
                        text_map = st.session_state.get("welcome_texts", {})
                        new_text = text_map.get(target, st.session_state.base_welcome)
                        if st.session_state.messages and st.session_state.messages[0]["role"] == "assistant":
                            st.session_state.messages[0]["content"] = new_text
                            st.session_state.welcome_lang = target
                            st.rerun()
        chip("Auto", 0, "auto")
        chip("EN", 1, "en")
        chip("हिन्दी", 2, "hi")
        chip("मराठी", 3, "mr")
    with cols[3]:
        st.caption("⚡ Quick questions")
        qcols = st.columns(2)
        if qcols[0].button("B.Tech Fees", use_container_width=True):
            st.session_state.quick_prefill = "What are the B.Tech CSE fees with hostel?"
        if qcols[1].button("Hostel", use_container_width=True):
            st.session_state.quick_prefill = "What are hostel charges and facilities?"
        if qcols[0].button("Scholarships", use_container_width=True):
            st.session_state.quick_prefill = "List available scholarships and eligibility."
        if qcols[1].button("Dates", use_container_width=True):
            st.session_state.quick_prefill = "What are the important admission dates?"

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])


# Prefill from quick actions or calculator
prefill_quick = st.session_state.get("quick_prefill", "")

# Handle user prompt (merge typed input with optional STT text)
prefill = st.session_state.get("stt_text", "") or prefill_quick
user_input = st.chat_input("💬 Ask in English / हिन्दी / मराठी...", key="chat_input")
if not user_input and prefill:
    user_input = prefill
    st.session_state.stt_text = ""
    st.session_state.quick_prefill = ""

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    if st.session_state.qa_chain:
        with st.spinner("🤖 Analyzing your question..."):
            try:
                # Detect language and translate to English for retrieval if needed
                # Respect explicit language selection
                if st.session_state.lang_pref != "auto":
                    user_lang = st.session_state.lang_pref
                else:
                    user_lang = detect_language_or_default(user_input, default="en")
                
                # Ensure we have a valid query
                if not user_input.strip():
                    raise ValueError("Please enter a valid question.")
                
                query_for_rag = user_input if user_lang == "en" else translate_text(user_input, dest="en")
                
                # Check if translation resulted in empty text
                if not query_for_rag.strip():
                    query_for_rag = user_input  # Fallback to original

                response_en = st.session_state.qa_chain.run(query_for_rag)
                
                # Validate response
                if not response_en or not response_en.strip():
                    raise ValueError("I couldn't generate a proper response. Please try rephrasing your question.")

                # Translate back to user's language if not English
                final_response = (
                    response_en if user_lang == "en" else translate_text(response_en, dest=user_lang)
                )
                
                # Ensure final response is valid
                if not final_response or not final_response.strip():
                    final_response = response_en  # Fallback to English response

                st.session_state.messages.append({"role": "assistant", "content": final_response})
                with st.chat_message("assistant", avatar="🎓"):
                    st.markdown(f"""
                    <div style="background:#FFFFFF;border:1px solid #EFEFF5;border-radius:12px;padding:14px;">
                        {final_response}
                        <div style="margin-top:10px; display:flex; gap:10px;">
                            <button onClick="navigator.clipboard.writeText(this.parentElement.parentElement.innerText)" style="padding:6px 10px;border-radius:8px;border:1px solid #E0E0EA;background:#FAFAFF;color:#2c3e50;cursor:pointer;">Copy</button>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Suggested follow-up questions
                    st.caption("Suggested next:")
                    follow_cols = st.columns(4)
                    followups = [
                        ("Detailed fee breakdown", "Please provide a detailed fee breakdown including tuition, exam, library, and other charges."),
                        ("Installment options", "Can I pay the fees in installments? What plans and due dates are available?"),
                        ("Scholarship eligibility", "List available scholarships with eligibility and how to apply."),
                        ("Hostel details", "Share hostel facilities, annual charges, and room types available."),
                    ]
                    for i, (lab, prompt) in enumerate(followups):
                        if follow_cols[i].button(lab, key=f"follow_{len(st.session_state.messages)}_{i}"):
                            st.session_state.quick_prefill = prompt
                            st.rerun()
            except Exception as e:
                error_msg = f"❌ **Sorry, I encountered an error while processing your request.**\n\nError details: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant", avatar="🎓"):
                    st.markdown(error_msg)
    else:
        err = "⚠️ **System Initialization Error**\n\nThe chatbot is not properly initialized. Please refresh the page or contact technical support."
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant", avatar="🎓"):
            st.markdown(err)

st.markdown('</div>', unsafe_allow_html=True)

# Floating microphone button
st.markdown("""
<div class="floating-mic" onclick="toggleFloatingMic()" id="floatingMic">
    🎤
</div>

<script>
function toggleFloatingMic() {
    // Trigger the hidden mic_recorder
    const micButton = document.querySelector('[data-testid="stMicRecorder"] button');
    if (micButton) {
        micButton.click();
    }
}
</script>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: rgba(255,255,255,0.9);">
    <p>🎓 <strong>MIT ADT University</strong> • Your Gateway to Excellence</p>
    <p style="font-size: 14px;">Powered by AI • Always here to help</p>
</div>
""", unsafe_allow_html=True)
