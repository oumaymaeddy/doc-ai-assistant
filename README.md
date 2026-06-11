#  Doc AI Assistant — RAG Application

A Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents and ask questions using natural language powered by a large language model.

##  Live Demo
👉 https://doc-ai-assistant-aniyhsuabmzz49fcr5tzt4.streamlit.app

## ✨ Features
- Upload any PDF document
- Ask questions in natural language via a chat interface
- AI answers grounded in document content (no hallucination)
- Powered by LLaMA 3.3 70B via Groq API
- Semantic search using FAISS vector store

## Tech Stack
| Layer | Technology |
|---|---|
| LLM | LLaMA 3.3 70B (Groq API) |
| Orchestration | LangChain |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| PDF Loader | PyPDF |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

##  How it works
1. Upload a PDF document
2. Document is split into chunks (RecursiveCharacterTextSplitter)
3. Chunks are embedded using HuggingFace sentence transformers
4. Embeddings stored in a FAISS vector store
5. User question is matched to relevant chunks (semantic search)
6. LLaMA 3.3 generates an answer grounded in retrieved context

## Run locally

### 1. Clone the repo
```bash
git clone https://github.com/oumaymaeddy/doc-ai-assistant
cd doc-ai-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:

doc-ai-assistant/
├── app.py              # Streamlit app
├── rag.py              # RAG pipeline (testing)
├── test.py             # LLM connection test
├── requirements.txt    # Dependencies
├── .env                # API keys (not committed)
├── .gitignore
└── README.md


## Environment Variables
| Variable | Description |
|---|---|
| GROQ_API_KEY | Groq API key (free at console.groq.com) |

##  Author
**Oumayma Oumouhou**
- GitHub: [@oumaymaeddy](https://github.com/oumaymaeddy)
- LinkedIn: https://www.linkedin.com/in/oumayma-oumouhou/
