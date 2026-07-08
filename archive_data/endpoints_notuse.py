# src/api/endpoints.py
import time
import json
import os
import mlflow
from fastapi import APIRouter, WebSocket, Request, Depends
from ..pipeline.rag_pipeline import RAGPipeline
from ..logging_config import get_logger
from fastapi.websockets import WebSocketDisconnect

logger = get_logger("endpoints")
router = APIRouter()

def get_app_state(request: Request) -> dict:
    return request.app.state

@router.get("/query/")
async def query_rag(question: str, state=Depends(get_app_state)):
    start_time = time.time()
    logger.info(f"▶️ [HTTP] Received query: '{question[:30]}...'")
    
    chain = state.pipeline.get_chain()
    
    # 🟢 MLflow run scoped to this single request
    with mlflow.start_run(run_name=f"http_{int(start_time)}", nested=True):
        mlflow.log_params({
            "doc_path": str(state.config.doc_path),
            "chunk_strategy": state.config.chunk_strategy,
            "llm_model": state.config.llm_model
        })
        
        response = await chain.ainvoke({"question": question})
        latency = time.time() - start_time
        
        mlflow.log_metric("query_latency_sec", latency)
        
        # Save query/response for lineage/debugging
        artifact_dir = "mlflow_artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        filepath = os.path.join(artifact_dir, f"resp_{int(start_time)}.json")
        with open(filepath, "w") as f:
            json.dump({"query": question, "response": response}, f)
        mlflow.log_artifact(filepath)

    logger.info(f"◼️  [HTTP] Response generated in {latency:.2f}s.")
    return {"query": question, "response": response}

@router.websocket("/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    
    # 🔑 FIX: Safely extract app state for WebSockets
    app = websocket.scope.get("app")
    if not app or not hasattr(app, "state"):
        await websocket.send_text("⚠️ Server not ready. Please try again.")
        return

    app_state = app.state
    if not hasattr(app_state, "pipeline"):
        await websocket.send_text("❌ RAG pipeline not initialized. Check server logs.")
        logger.error("🚫 [WS] Pipeline missing from app.state")
        return

    logger.info("🔌 [WS] Connection accepted.")
    
    try:
        while True:
            start_time = time.time()
            query = await websocket.receive_text()
            
            # 🟢 MLflow run scoped to this WS message
            with mlflow.start_run(run_name=f"ws_{int(time.time())}", nested=True):
                chain = app_state.pipeline.get_chain()
                logger.info(f"▶️ [WS] Received: '{query[:30]}...'")

                latency_start = time.time()
                logger.info("⏳ [RETRIEVAL] Searching...")
                
                response_chunks = []
                async for chunk in chain.astream({"question": query}):
                    if isinstance(chunk, str):
                        response_chunks.append(chunk)
                        await websocket.send_text(chunk)
                
                latency = time.time() - start_time
                mlflow.log_metric("ws_latency_sec", latency)

                # Save WS response artifact
                artifact_dir = "mlflow_artifacts"
                os.makedirs(artifact_dir, exist_ok=True)
                filepath = os.path.join(artifact_dir, f"ws_resp_{int(time.time())}.json")
                with open(filepath, "w") as f:
                    json.dump({"query": query, "response": "".join(response_chunks)}, f)
                mlflow.log_artifact(filepath)

            await websocket.send_text("\n\n")
            
    except WebSocketDisconnect:
        logger.info("⚠️ [WS] Client disconnected.")
    except Exception as e:
        logger.error(f"❌ [WS] Error: {e}")
        try: 
            await websocket.send_text(f"\n\n[Error: {str(e)}]")
        except: pass