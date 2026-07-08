# src/retrieval.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def setup_rag_system():
    # Load and split document
    loader = TextLoader("data/my_document.txt")
    documents = loader.load()

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_chunks = splitter.split_documents(documents)

    # Embeddings via Ollama's native API
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url="http://localhost:11434"
    )

    # Vector store + retriever
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

retriever = setup_rag_system()