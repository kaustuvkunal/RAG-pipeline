# # from langchain_openai import OpenAI  # Updated import
# # from langchain.chains import RetrievalQA
# # from src.retrival import retriever

# # llm = OpenAI(openai_api_key=openai_api_key)

# # qa_chain = RetrievalQA.from_chain_type(
# #     llm=llm,
# #     chain_type="stuff",
# #     retriever=retriever
# # )

# # query = "Are polar bears in danger?"
# # response = qa_chain.invoke({"query": query})
# # print(response)
# # src/querying.py
# from langchain_openai import OpenAI
# from langchain.chains import RetrievalQA
# from src.retriever import retriever

# llm = OpenAI(openai_api_key="your_openai_api_key")

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever
# )

# async def get_rag_response(query: str):
#     response = qa_chain.invoke({"query": query})
#     return response