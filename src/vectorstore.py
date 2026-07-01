from typing import List
from langchain_core.documents import Document
from .logging_config import get_logger
import os 
logger = get_logger("vectorstore")

def get_vectorstore(documents: List[Document], embedder, config):
    logger.info(f"▶️ [VECTORSTORE] type={config.vector_db_type} | docs={len(documents)}")
    
    #ChromaDB initialization and document ingestion
    if config.vector_db_type == "chroma":
        from langchain_chroma import Chroma
        
        logger.info("⏳ [VECTORSTORE] Initializing ChromaDB instance...")
        vector_store = Chroma(
            collection_name=config.chroma_collection_name,
            persist_directory=config.chroma_persist_directory, 
            embedding_function=embedder  # We pass the embedder here to handle vectors later
        )

        # Ingest documents in small batches (e.g., 50 at a time)
        batch_size = int(os.getenv("VECTORSTORE_BATCH_SIZE", "50"))
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            chunk_end = min(i + batch_size, total_docs)
            logger.info(f"⏳ [VECTORSTORE] Ingesting batch {i // batch_size + 1} of {(total_docs - 1)//batch_size + 1}...")
            
            # Add the batch directly to the initialized store
            vector_store.add_documents(documents[i:chunk_end])

        logger.info("✅ [VECTORSTORE] Completed ingestion successfully.")
        # Log vector DB location
        logger.info(f"Chroma DB Vector DB path is - {config.chroma_persist_directory}")
    #
    elif config.vector_db_type == "faiss":
        from langchain_community.vectorstores import FAISS
        vector_store = FAISS.from_documents(documents, embedder)
        
    elif config.vector_db_type == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        # Initialize Qdrant Client
        logger.info("⏳ [VECTORSTORE] Initializing Qdrant Client...")
        qdrant_client = QdrantClient(url=config.qdrant_url )
        
        # Check if collection exists and validate dimensions
        logger.info("⏳ [VECTORSTORE] Checking if Qdrant Collection exists...")
        collection_name = config.qdrant_collection_name
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            logger.info(f"⏳ [VECTORSTORE] Collection '{collection_name}' already exists.")
            
            # Check if dimensions match - if not, recreate the collection
            existing_vector_params = collection_info.config.params.vectors
            
            if existing_vector_params and existing_vector_params.size != config.embedding_dim:
                logger.info(f"⚠️ [VECTORSTORE] Dimension mismatch: existing={existing_vector_params.size}, expected={config.embedding_dim}. Recreating collection...")
                qdrant_client.delete_collection(collection_name)
                # Create new collection with correct dimensions
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=config.embedding_dim, distance=Distance.COSINE),
                )
            else:
                logger.info(f"✅ [VECTORSTORE] Collection dimensions match: {config.embedding_dim}")
                
        except Exception as e:
            logger.info(f"⏳ [VECTORSTORE] Collection '{collection_name}' not found. Creating new collection...")
            # Create the collection
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=config.embedding_dim, distance=Distance.COSINE),
            )
                     
        vector_store = QdrantVectorStore( client=qdrant_client, 
                                         collection_name=config.qdrant_collection_name,
                                         embedding=embedder,
                                          )
        
        # Ingest documents in small batches (e.g., 50 at a time)
        batch_size = int(os.getenv("VECTORSTORE_BATCH_SIZE", "50"))
        total_docs = len(documents)

        for i in range(0, total_docs, batch_size):
            chunk_end = min(i + batch_size, total_docs)
            logger.info(f"⏳ [VECTORSTORE] Ingesting batch {i // batch_size + 1} of {(total_docs - 1)//batch_size + 1}...") 
            # Add the batch directly to the initialized store
            vector_store.add_documents(documents[i:chunk_end])
        logger.info("✅ [VECTORSTORE] Qdrant Vector Store successfully created and populated.") 
        
    return vector_store