from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from .logging_config import get_logger

logger = get_logger("retriever")

def get_retriever(vector_store, chunks: List[Document], config) -> EnsembleRetriever | ...:
    logger.info(f"▶️ [RETRIEVER] strategy={config.search_strategy} | top_k={config.k}")
    
    if config.search_strategy == "semantic":
        logger.info(" Retriver - search_strategy - semantic")
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": config.k})
    elif config.search_strategy == "keyword":
        return BM25Retriever.from_documents(chunks)
    elif config.search_strategy == "hybrid":
        sem = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": config.k, "score_threshold": 0.3})
        kw = BM25Retriever.from_documents(chunks)
        return EnsembleRetriever(retrievers=[sem, kw], weights=[0.6, 0.4])
        
    raise ValueError(f"❌ [RETRIEVER] Unsupported strategy: {config.search_strategy}")
