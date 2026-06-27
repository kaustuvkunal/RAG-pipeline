from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from .logging_config import get_logger, mask_sensitive
from typing import Union

logger = get_logger("embedding")

def get_embedding_model(config) -> Union[OllamaEmbeddings, OpenAIEmbeddings]:
    logger.info(f"▶️ [EMBEDDER] type={config.embedder_type} | model={config.embedding_model}")
    
    if config.embedder_type == "ollama":
        # Ensure Ollama client uses the correct URL without port mapping issues
        try:
            embeddings = OllamaEmbeddings(
                model=config.embedding_model, 
                base_url=config.embedding_base_url
            )
            # Test connection immediately to fail fast if Ollama is down or model missing
            embeddings.embed_documents(["test"])
            logger.info("✅ Ollama Embedder connected successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama Embedder: {e}")
            raise RuntimeError("Ollama is not running or model is missing. Run 'ollama pull' first.") from e
            
    elif config.embedder_type == "openai":
        logger.debug(f"   API key masked: {mask_sensitive(config.api_key)}")
        return OpenAIEmbeddings(model=config.embedding_model, openai_api_key=config.api_key)
        
    raise ValueError(f"❌ [EMBEDDER] Unsupported type: {config.embedder_type}")
