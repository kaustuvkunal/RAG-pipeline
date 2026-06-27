"""Typed configuration with .env support & explicit validation."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from pydantic import Field
from typing import Literal
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

    # Chroma 
    chroma_collection_name: str = os.getenv('CHROMA_COLLECTION_NAME', 'tesla_rag')
    chroma_persist_directory: str = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')

    # qdant 
    qdrant_url: str = os.getenv('QDRANT_URL', 'http://localhost:6333')

    # Embedding & LLM Backend
    embedder_type: Literal["ollama", "openai"] = os.getenv('EMBEDDER_TYPE', 'ollama')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'qwen3-embedding:0.6b')
    embedding_base_url: str = os.getenv('EMBEDDING_BASE_URL', 'http://localhost:11434')

    llm_type: Literal["ollama", "openai"] = os.getenv('LLM_TYPE', 'ollama')
    llm_model: str = os.getenv('LLM_MODEL', 'deepseek-r1:8b')
    llm_base_url: str = os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1')
    api_key: str = os.getenv('API_KEY', 'ollama')

    search_strategy: Literal["semantic", "keyword", "hybrid"] = os.getenv('SEARCH_STRATEGY', 'semantic')
    k: int = int(os.getenv('K', 5))


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
