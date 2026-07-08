import os
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from src.config import load_config, RAGConfig
from src.pipeline.rag_pipeline import RAGPipeline
from src.api.endpoints import router as api_router  # Import after app creation
from src.logging_config import setup_logging, get_logger
import mlflow

warnings.filterwarnings("ignore")  # Suppress LangChain deprecation noise

# Setup global logging once at application startup
setup_logging(os.getenv("LOG_LEVEL", "INFO"), console_output=False, file_output=True)

logger = get_logger("main")
config: RAGConfig = load_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages app startup/shutdown lifecycle."""
    import mlflow
    #os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("http://localhost:5001/")
    mlflow.set_experiment("RAG-pipeline")

    with mlflow.start_run(run_name="pipeline_initialization", tags={"stage": "startup"}) as init_run:
        logger.info("🚀 Starting Modular RAG Engine")
        
        # Load config and create pipeline
        app.state.config = load_config()
        pipeline = RAGPipeline(app.state.config)
        _ = pipeline.get_chain()  # Force init
        
        # Set the pipeline in app state - THIS IS THE KEY FIX
        app.state.pipeline = pipeline
        
        # Log config artifact during startup
        artifact_path = pipeline.log_initialization_artifact()
        mlflow.log_artifact(artifact_path)
        logger.info(f"📋 Pipeline config logged to MLflow | run_id={init_run.info.run_id}")
        logger.info("✅ Pipeline ready. Server listening.")
        
    yield

# Create the FastAPI app instance AFTER defining lifespan
app = FastAPI(title="Modular RAG Engine", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Include router after app creation to avoid circular imports
app.include_router(api_router, prefix="/api")
app.state.templates = Jinja2Templates(directory="src/ui")

@app.get("/")
async def root(request: Request):
    return app.state.templates.TemplateResponse(request=request, 
                                                name="index.html", 
                                                context={})