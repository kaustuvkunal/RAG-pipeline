from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import documents from loader.py
from src.archive.loader import documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
document_chunks = splitter.split_documents(documents)