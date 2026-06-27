from langchain_openai import ChatOpenAI
from .logging_config import get_logger, mask_sensitive

logger = get_logger("llm")

def get_llm_client(config) -> ChatOpenAI:
    logger.info(f"▶️ [LLM] type={config.llm_type} | model={config.llm_model}")
    logger.debug(f"   API key masked: {mask_sensitive(config.api_key)}")
    
    # OpenAI-compatible wrapper routes to Ollama when base_url points to /v1 & api_key="ollama"
    return ChatOpenAI(
        model=config.llm_model,
        api_key=config.api_key,
        base_url=config.llm_base_url or "http://localhost:11434/v1"
    )
