# 🎓 Admission Assistant

A modern, AI-powered chatbot designed to help prospective students with University admission queries. Built with Streamlit and powered by LangChain with Ollama LLM integration.

## ✨ Features

- **Intelligent Q&A**: Powered by advanced RAG (Retrieval Augmented Generation) pipeline
- **Modern UI**: Beautiful, responsive interface with gradient designs
- **Comprehensive Data**: Covers all program fees, hostel charges, and admission requirements
- **Real-time Processing**: Fast responses using local Ollama LLM
- **Mobile Friendly**: Responsive design that works on all devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/admission_chatbot.git
   cd admission_chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the required model
   ollama pull llama3.1:8b
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
admission_chatbot/
├── app.py                 # Main Streamlit application
├── admission_fees.txt     # University fee data
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
├── pyth.py               # Test script for model validation
└── sentiment_app.py      # Additional sentiment analysis app
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **AI/ML**: LangChain, Ollama, FAISS
- **Embeddings**: HuggingFace Transformers
- **Vector Store**: FAISS
- **LLM**: Llama 3.1 8B (via Ollama)

## 📊 Data Coverage

The chatbot provides information about:

- **Programs**: B.Tech, M.Tech, MBA, Architecture, Law, Design, etc.
- **Fee Structure**: Detailed breakdowns for all programs
- **Hostel Information**: Accommodation costs and facilities
- **Admission Requirements**: Process and prerequisites
- **Special Programs**: NRI/OCI fee structures

## 🎯 Usage

1. **Ask Questions**: Type your query in the chat input
2. **Quick Actions**: Use the preset buttons for common questions
3. **Get Detailed Answers**: The AI provides comprehensive, context-aware responses

### Example Queries

- "What are the B.Tech CSE fees?"
- "Tell me about hostel charges"
- "What are the MBA program details?"
- "How much does Architecture cost?"

## 🔧 Configuration

### Model Configuration

The app uses Ollama with Llama 3.1 8B model. You can modify the model in `app.py`:

```python
def initialize_llm():
    return Ollama(
        model="llama3.1:8b",  # Change this to your preferred model
        base_url="http://localhost:11434",
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.1
    )
```

### Data Source

The chatbot uses `admission_fees.txt` as its knowledge base. Update this file to modify the information available to the chatbot.

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

For production deployment, consider using:
- **Streamlit Cloud**: Direct deployment from GitHub
- **Docker**: Containerized deployment
- **Cloud Platforms**: AWS, GCP, or Azure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request




## 🙏 Acknowledgments

- University for providing the admission data
- Streamlit team for the amazing framework
- LangChain for the RAG implementation
- Ollama for local LLM capabilities


