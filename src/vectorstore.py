from typing import List
from langchain_core.documents import Document
from .logging_config import get_logger

logger = get_logger("vectorstore")

# src/vectorstore.py

def get_vectorstore(documents: List[Document], embedder, config):
    logger.info(f"▶️ [VECTORSTORE] type={config.vector_db_type} | docs={len(documents)}")
    
    if config.vector_db_type == "chroma":
        from langchain_chroma import Chroma
        
        # 1. Initialize the Vector Store Instance (without loading all docs at once)
        logger.info("⏳ [VECTORSTORE] Initializing ChromaDB instance...")
        vector_store = Chroma(
            collection_name=config.chroma_collection_name,
            persist_directory=config.chroma_persist_directory, 
            embedding_function=embedder  # We pass the embedder here to handle vectors later
        )

        # 2.  Ingest documents in small batches (e.g., 50 at a time)
        batch_size = 50 
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            chunk_end = min(i + batch_size, total_docs)
            logger.info(f"⏳ [VECTORSTORE] Ingesting batch {i // batch_size + 1} of {(total_docs - 1)//batch_size + 1}...")
            
            # Add the batch directly to the initialized store
            vector_store.add_documents(documents[i:chunk_end])

        logger.info("✅ [VECTORSTORE] Completed ingestion successfully.")
        # log vector DB location
        logger.info(" Chroma DB Vector DB path is -  {persist_directory}")
        return vector_store

    elif config.vector_db_type == "faiss":
        from langchain_community.vectorstores import FAISS
        return FAISS.from_documents(documents, embedder)
        
    elif config.vector_db_type == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        return QdrantVectorStore.from_documents(
            documents, embedder, url=config.qdrant_url, collection_name="rag_collection"
        )
        
    raise ValueError(f"❌ [VECTORSTORE] Unsupported type: {config.vector_db_type}")
