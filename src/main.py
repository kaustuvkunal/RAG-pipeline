import os
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from .config import load_config
from .pipeline.rag_pipeline import RAGPipeline
from .api.endpoints import router as api_router
from .logging_config import setup_logging, get_logger

warnings.filterwarnings("ignore")  # Suppress LangChain deprecation noise

logger = get_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages app startup/shutdown lifecycle."""
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    logger.info("🚀 Starting Modular RAG Engine")
    
    try:
        app.state.config = load_config()
        app.state.pipeline = RAGPipeline(app.state.config)
        _ = app.state.pipeline.get_chain()  # Force init to catch config/pipeline errors early
        logger.info("✅ Pipeline ready. Server listening.")
    except Exception as e:
        logger.critical(f"❌ Startup failed: {e}")
        raise
    
    yield

app = FastAPI(title="Modular RAG Engine", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

app.include_router(api_router, prefix="/api")
app.state.templates = Jinja2Templates(directory="src/ui")

@app.get("/")
async def root(request: Request):
    return app.state.templates.TemplateResponse(request=request, name="index.html", context={})
