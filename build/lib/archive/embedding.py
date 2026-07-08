from langchain_ollama import OllamaEmbeddings

# Initialize the embedding model
qwen_embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434",  # default; omit if using default port
)

# Embed a single query
query_vector = qwen_embeddings.embed_query("What is machine learning?")
print(f"Vector length: {len(query_vector)}")
print(f"First 5 values: {query_vector[:5]}")

# Embed a list of documents
docs = [
    "LangChain is a framework for LLM apps.",
    "Ollama runs models locally on your machine.",
    "Embeddings convert text to dense numeric vectors.",
]
doc_vectors = qwen_embeddings.embed_documents(docs)
print(f"Embedded {len(doc_vectors)} documents")