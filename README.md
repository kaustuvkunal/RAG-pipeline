# 🔍 Modular RAG Engine

A lightweight, production-ready Retrieval-Augmented Generation system built with FastAPI, LangChain v0.3+, and Pydantic Settings. Features explicit invocation tracking, secure config validation, and lazy pipeline initialization.

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install fastapi uvicorn langchain pydantic-settings langchain-openai langchain-community chromadb faiss-cpu ollama httpx

Fast API Execution : 


pip install pipreqs

pipreqs . --force

Prerequsites : 
 export KMP_DUPLICATE_LIB_OK=TRUE


To execute : 
  uvicorn src.main:app --reload



To execute 

check .env
update  config file 
update prompt.py with correct prompt template 

update  Embeeding model , vectorDB