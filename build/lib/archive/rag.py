from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI          # ✅ changed from OpenAI
from langchain_core.messages import HumanMessage

# LLM via Ollama's OpenAI-compatible chat endpoint
llm = ChatOpenAI(
    api_key="ollama",
    model="gemma4:e4b",
    base_url="http://localhost:11434/v1"          # /v1 stays here ✅
)

# Build RAG system ONCE at startup
def setup_rag_system():
    loader = TextLoader("data/my_document.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url="http://localhost:11434"
    )

    vector_store = FAISS.from_documents(document_chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    return retriever



# Query function (sync)
async def  get_rag_response(query: str) -> str:
     
    retriever = setup_rag_system()
    retrieved_docs = retriever.invoke(query)                    # ✅ not get_relevant_documents()

    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}"

    response = llm.invoke([HumanMessage(content=prompt)])       # ✅ not llm.generate()
    return response.content


 