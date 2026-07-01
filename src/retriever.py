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

#Databricks Vector Search
# from databricks.vector_search.client import VectorSearchClient
# vs = VectorSearchClient()
# index = vs.get_index(endpoint_name="my-endpoint", index_name="catalog.schema.my_index")
# def retrieve_databricks(query: str, top_k: int = 5, query_type: str = "ann") -> list[str]:
#     kwargs = {"query_text": query, "columns": ["doc_id"], "num_results": top_k}
#     if query_type == "hybrid":
#         kwargs["query_type"] = "hybrid"
#     res = index.similarity_search(**kwargs)
#     return [row[0] for row in res["result"]["data_array"]]

#Pinecone
# from pinecone import Pinecone
# pc = Pinecone(api_key="your-api-key")
# index = pc.Index("my-index")
# def retrieve_pinecone(query: str, top_k: int = 5, embedding_fn=None) -> list[str]:
#     query_vec = embedding_fn(query)  # your embedding model
#     res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
#     return [match["doc_id"] for match in res["matches"]]

# #LanceDB
# import lancedb
# db = lancedb.connect("./my_lancedb")
# table = db.open_table("my_docs")
# def retrieve_lancedb(query: str, top_k: int = 5) -> list[str]:
#     results = table.search(query).limit(top_k).to_pandas()
#     return results["doc_id"].tolist()
