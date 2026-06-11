import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os

load_dotenv()

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Doc AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Doc AI Assistant")
st.caption("Upload a PDF and ask questions about it using AI")

# ── Session state ─────────────────────────────────────────────────
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar — PDF Upload ──────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload your document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        with st.spinner("Processing document..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load + split
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            chunks = splitter.split_documents(pages)

            # Embeddings + vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # RAG chain
            llm = ChatGroq(
                model="llama-3.3-70b-versatile", temperature=0
            )
            retriever = vectorstore.as_retriever()
            prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}

If the answer is not in the context, say "I don't have enough information in the document to answer this question."
""")
            st.session_state.rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            os.unlink(tmp_path)

        st.success(f"✅ Ready! {len(pages)} pages · {len(chunks)} chunks")
        st.session_state.messages = []

# ── Chat interface ────────────────────────────────────────────────
if st.session_state.rag_chain is None:
    st.info("👈 Upload a PDF in the sidebar to get started")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    if question := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(question)
            st.write(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )