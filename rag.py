from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 2. Charge le PDF
loader = PyPDFLoader("document.pdf")
pages = loader.load()
print(f"✅ PDF chargé : {len(pages)} pages")

# 3. Découpe en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"✅ Chunks créés : {len(chunks)}")

# 4. Embeddings + Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ Vector store créé !")

# 5. RAG Chain
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Pose une question
question = "What is this document about?"
print(f"\n❓ Question: {question}")
answer = rag_chain.invoke(question)
print(f"💡 Réponse: {answer}")