import streamlit as st
import docx  # Make sure this is at the top of your file
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from typing import Optional, List, Mapping, Any
import os

# Page config
st.set_page_config(
    page_title="Admission Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling for modern chatbot interface
st.markdown("""
    <style>
    /* Main container */
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
        margin: 0;
    }
    
    /* Header styling */
    .header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .header h1 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        margin: 20px auto;
        max-width: 800px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        border-radius: 18px 18px 5px 18px;
    }
    
    /* Assistant messages */
    .stChatMessage[data-testid="assistant"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #2c3e50;
        margin-right: 20%;
        border-radius: 18px 18px 18px 5px;
        border-left: 4px solid #667eea;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        text-align: center;
        color: #667eea;
    }
    
    /* Error styling */
    .error {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
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

# Header
st.markdown("""
    <div class="header">
        <h1>Admission Assistant</h1>
        <p><span class="status-online"></span>Live Assistant • Ready to help with admissions</p>
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "👋 **Welcome to University!**\n\nI'm your personal admission assistant, powered by advanced AI. I can help you with:\n\n🎓 **Program Information** - Fees, duration, requirements\n🏠 **Hostel & Accommodation** - Costs and facilities\n💰 **Fee Structure** - Detailed breakdowns and calculations\n📋 **Admission Process** - Requirements and procedures\n\n**What would you like to know?**"
        }
    ]

# Load and split handbook text


def load_text():
    text_path = "/Users/pranav/Documents/admission_chatbot/admission_fees.txt"
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
    texts = load_text()
    if texts is None:
        return None
    vector_store = create_vector_store(texts)
    if vector_store is None:
        return None
    llm = initialize_llm()
    if llm is None:
        return None

    prompt_template = """You are a professional University admission counselor. Use ONLY the context provided below to answer questions about admission fees, programs, and requirements. Be precise with numbers and calculations.

Guidelines:
- Answer based ONLY on the provided context
- For fee calculations, show the math clearly
- If information is not in the context, say "I don't have that information in the available documents"
- Be helpful and professional
- Format your answers clearly with bullet points when listing multiple items

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
        st.session_state.qa_chain = setup_rag_chain()

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# Quick action buttons
st.markdown("""
<div style="margin: 20px 0; text-align: center;">
    <h4 style="color: #7f8c8d; margin-bottom: 15px;">Quick Questions</h4>
    <div style="display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;">
        <button onclick="askQuestion('What are the B.Tech CSE fees?')" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 8px 16px; border-radius: 20px; cursor: pointer; font-size: 14px;">B.Tech CSE Fees</button>
        <button onclick="askQuestion('What are the hostel charges?')" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 8px 16px; border-radius: 20px; cursor: pointer; font-size: 14px;">Hostel Charges</button>
        <button onclick="askQuestion('MBA program details')" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 8px 16px; border-radius: 20px; cursor: pointer; font-size: 14px;">MBA Program</button>
        <button onclick="askQuestion('Architecture program fees')" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 8px 16px; border-radius: 20px; cursor: pointer; font-size: 14px;">Architecture Fees</button>
    </div>
</div>
""", unsafe_allow_html=True)

# Handle user prompt
if user_input := st.chat_input("💬 Ask me anything about university admissions..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    if st.session_state.qa_chain:
        with st.spinner("🤖 Analyzing your question..."):
            try:
                response = st.session_state.qa_chain.run(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant", avatar="🎓"):
                    st.markdown(response)
            except Exception as e:
                error_msg = f"❌ **Sorry, I encountered an error:**\n\n{str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant", avatar="🎓"):
                    st.markdown(error_msg)
    else:
        err = "⚠️ **System Initialization Error**\n\nThe chatbot is not properly initialized. Please refresh the page or contact technical support."
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant", avatar="🎓"):
            st.markdown(err)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: rgba(255,255,255,0.8);">
    <p>🎓 <strong>University</strong> • Your Gateway to Excellence</p>
    <p style="font-size: 14px;">Powered by My Brain • Always here to help</p>
</div>
""", unsafe_allow_html=True)
