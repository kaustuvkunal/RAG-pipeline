import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ..vectorstore import get_vectorstore
from ..chunking import get_chunker
from ..embedding_model import get_embedding_model
from ..llm_client import get_llm_client
from ..retriever import get_retriever
from ..prompt import get_rag_prompt
from ..logging_config import get_logger, log_sequence
from operator import itemgetter

logger = get_logger("pipeline")

class RAGPipeline:
    """Caches state after first init. Logs every step of the data flow."""
    
    def __init__(self, config):
        logger.info(f"▶️ [PIPELINE] Initializing with doc_path={config.doc_path}")
        self.config = config
        self._retriever = None
        self._chain = None

    def _ensure_initialized(self):
        if self._retriever is not None and self._chain is not None:
            logger.debug("⏭️ [PIPELINE] Already initialized. Skipping re-init.")
            return
            
        # [1/6] Load Documents
        with log_sequence("Loading documents"):
            doc_path = self.config.doc_path
            if os.path.isfile(doc_path):
                docs = TextLoader(doc_path).load()
            elif os.path.isdir(doc_path):
                all_docs = []
                for f in Path(doc_path).rglob("*"):
                    if not f.is_file(): continue
                    loader = PyPDFLoader if f.suffix.lower() == ".pdf" else TextLoader
                    all_docs.extend(loader(str(f)).load())
                docs = all_docs
            else:
                raise FileNotFoundError(f"❌ [PIPELINE] Invalid doc_path: {doc_path}. Check .env.")
            
            if not docs:
                raise ValueError("⚠️ [PIPELINE] No documents found in path. Verify file extensions.")

        # [2/6] Chunk Text
        with log_sequence("Chunking text"):
            chunks = get_chunker(self.config).split_documents(docs)
            logger.info(f"◼️  [CHUNKING] Split into {len(chunks)} chunks (strategy={self.config.chunk_strategy})")

        # [3/6] Generate Embeddings
        with log_sequence("Generating embeddings"):
            embedder = get_embedding_model(self.config)
            logger.info(f"◼️  [EMBEDDER] Loaded {self.config.embedder_type} model: {self.config.embedding_model}")


        # [4/6] Build Vector Store
        with log_sequence("Building vector store"):
            vector_store = get_vectorstore(chunks, embedder, self.config)
            logger.info(f"◼️  [VECTORSTORE] Indexed {len(chunks)} docs into {self.config.vector_db_type}")

        # [5/6] Configure Retriever
        with log_sequence("Initializing retriever"):
            self._retriever = get_retriever(vector_store, chunks, self.config)
            logger.info(f"◼️  [RETRIEVER] Strategy={self.config.search_strategy} | top_k={self.config.k}")

        # [6/6] Construct Chain
        with log_sequence("Constructing LLM chain"):
            llm = get_llm_client(self.config)
            prompt_template = get_rag_prompt()
            
            self._chain = (
                {"context": itemgetter("question") | self._retriever, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            logger.info("✅ [PIPELINE] Fully initialized. Ready for queries.")

    def get_chain(self):
        """Lazy initialization gate. Returns cached chain if available."""
        self._ensure_initialized()
        return self._chain
