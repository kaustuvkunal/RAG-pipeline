from langchain_community.vectorstores import FAISS
from src.archive.chunking import document_chunks
from src.archive.embedding import embeddings
from langchain_qdrant import QdrantVectorStore


vector_store = FAISS.from_documents(document_chunks, embeddings)

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)


# Create collection and index documents
vector_store = QdrantVectorStore.from_documents(
    documents=document_chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="my_docs",
)