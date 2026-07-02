# 🔍 Modular RAG Engine
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-blue)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-blue)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kaustuvkunal/...../blob/main/LICENSE)
[![Status](https://img.shields.io/badge/status-development-yellow)](https://github.com/kaustuvkunal/...../commits/main)

A lightweight, production-ready Retrieval-Augmented Generation system built with FastAPI, LangChain v0.3+, and Pydantic Settings. Features explicit invocation tracking, secure config validation, and lazy pipeline initialization.

## 🚀 Quick Start

To setup the environment properly, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/kaustuvkunal/.....git
   cd <repository-name>
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

4. **Install dependencies using pip**
   ```bash
   pip install .
   ```

   Or for development mode:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Copy and configure environment variables**
   ```bash
   cp .env.example .env
   ```

6. **Export Environment Variable**
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE
   ```

7. **Run the application using uvicorn**
   ```bash
   uvicorn src.main:app --reload
   ```

## 🛠️ Configuration

### Environment Variables
Create a `.env` file in the project root with the following variables:
```
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_DB_PATH=./vector_db
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

### Config File
Update `src/config.py` to match your specific requirements for:
- Embedding model settings
- Vector database configuration
- Retrieval parameters

## 📝 Prompt Template

Modify `src/prompt.py` to customize the prompt template according to your use case.

## 🧪 Testing

To run tests:
```bash
pytest
```

## 🏗️ Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── prompt.py
│   ├── embedding_model.py
│   ├── llm_client.py
│   ├── vectorstore.py
│   ├── chunking.py
│   ├── retriver.py
│   ├── api/
│   ├── pipeline/
│   ├── eval/
│   ├── ui/
│   └── archive/
├── test/
├── data/
├── .env.example
├── pyproject.toml
├── README.md
└── requirements.txt
```

## 📦 Dependencies

This project uses the following key dependencies:
- FastAPI: Web framework for building APIs
- LangChain: RAG framework
- Pydantic Settings: Configuration management
- Qdrant: Vector database
- OpenAI: LLM integration

## 🔧 Development

For development, install with dev dependencies:
```bash
pip install -e ".[dev]"
```

Run code formatting and linting:
```bash
ruff format
ruff check
mypy src/
```