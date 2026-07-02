"""Typed configuration with .env support & explicit validation."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from pydantic import Field
from typing import Dict, Literal, Any
from .logging_config import get_logger
import os 

logger = get_logger("config")

class RAGConfig(BaseSettings):
    """Application settings loaded from .env with strict validation."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

   # Document Processing
    doc_path: str = os.getenv('DOC_PATH', '')
    chunk_strategy: Literal["recursive", "markdown", "semantic"] = os.getenv('CHUNK_STRATEGY', 'recursive')
    chunk_size: int = int(os.getenv('CHUNK_SIZE', 500))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', 50))

    # Vector Database
    vector_db_type: Literal["faiss", "qdrant", "chroma"] = os.getenv('VECTOR_DB_TYPE', 'faiss')


    # faiss 
    faiss_persist_directory : str = os.getenv('FAISS_PERSIST_DIRECTORY', './vectorstore/faiss')
    # Chroma 
    chroma_collection_name: str = os.getenv('CHROMA_COLLECTION_NAME', 'tesla_rag')
    chroma_persist_directory: str = os.getenv('CHROMA_PERSIST_DIRECTORY', './vectorstore/chroma_db')

    # qdant 
    qdrant_url: str = os.getenv('QDRANT_URL', 'http://localhost:6333')
    qdrant_collection_name: str =os.getenv('QDRANT_COLLECTION_NAME', 'Qdrant_default_collection')
    qdrant_persist_directory :str =os.getenv('QDRANT_PERSIST_DIR','./qdrant_db')

    # Embedding & LLM Backend
    embedder_type: Literal["ollama", "openai"] = os.getenv('EMBEDDER_TYPE', 'ollama')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'qwen3-embedding:0.6b')
    embedding_dim : int = int(os.getenv('EMBEDDING_DIM', 500))
    embedding_base_url: str = os.getenv('EMBEDDING_BASE_URL', 'http://localhost:11434')

    llm_type: Literal["ollama", "openai"] = os.getenv('LLM_TYPE', 'ollama')
    llm_model: str = os.getenv('LLM_MODEL', 'deepseek-r1:8b')
    llm_base_url: str = os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1')
    api_key: str = os.getenv('API_KEY', 'ollama')

    # Search Strategy
    search_strategy: Literal["semantic", "keyword", "hybrid"] = os.getenv('SEARCH_STRATEGY', 'semantic')
    k: int = int(os.getenv('K', 5))

    # deepeval
    # fetch from .env
    answer_relevancy_threshold: float = float(os.getenv('ANSWER_RELEVANCY_THRESHOLD', 0.8))
    faithfulness_threshold: float = float(os.getenv('FAITHFULNESS_THRESHOLD', 0.8))
    contextual_relevancy_threshold: float = float(os.getenv('CONTEXTUAL_RELEVANCY_THRESHOLD', 0.8))
    expected_output_needed : bool = os.getenv('EXPECTED_OUTPUT_NEEDED', 'false').lower() == 'true'   
    number_of_evolution :int = int(os.getenv('NUM_OF_EVOLUTIONS', 10))
    


    @model_validator(mode="after")
    def validate_required_fields(self):
        if not self.doc_path.strip():
            raise ValueError("DOC_PATH cannot be empty. Set it in .env to a valid file or directory path.")
        return self

def load_config() -> RAGConfig:
    """Load and log configuration state."""
    logger.info("▶️ [CONFIG] Loading RAG settings...")
    cfg = RAGConfig()
    logger.info(f"✅ Config loaded | doc_path={cfg.doc_path} | db={cfg.vector_db_type} | llm={cfg.llm_model}")
    return cfg
