from langchain_core.prompts import ChatPromptTemplate
from .logging_config import get_logger

logger = get_logger("prompt")

SYSTEM_PROMPT = """You are a precise RAG assistant. 
Answer using ONLY the provided <Context>. Do not invent facts or reference document structure.
If information is missing, respond exactly: "I don't know".

<Context>
{context}
</Context>

<Question>
{question}
</Question>"""

def get_rag_prompt() -> ChatPromptTemplate:
    logger.info("▶️ [PROMPT] Loading system prompt template...")
    return ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT)])
