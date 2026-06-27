from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

from .logging_config import get_logger

logger = get_logger("chunking")

def get_chunker(config) -> RecursiveCharacterTextSplitter | MarkdownHeaderTextSplitter:
    logger.info(f"▶️ [CHUNKING] Initializing strategy={config.chunk_strategy} | size={config.chunk_size} | overlap={config.chunk_overlap}")
    
    if config.chunk_strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    elif config.chunk_strategy == "markdown":
        headers_to_split_on = [("H1", "#"), ("H2", "##"), ("H3", "###")]
        return MarkdownHeaderTextSplitter(headers_to_split_on)
        
    raise ValueError(f"❌ [CHUNKING] Unsupported strategy: {config.chunk_strategy}")
