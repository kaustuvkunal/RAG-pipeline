import time
import json
import os
import mlflow
from fastapi import APIRouter, WebSocket, Request
from ..logging_config import get_logger
from fastapi.websockets import WebSocketDisconnect

logger = get_logger("endpoints")
router = APIRouter()

@router.get("/query")
async def query_rag(question: str, request: Request):
    start_time = time.time()
    logger.info(f"▶️ [HTTP] Received query: '{question[:30]}...'")
    
    try:
        # Access pipeline from app state
        state = request.app.state
        if not hasattr(state, 'pipeline') or state.pipeline is None:
            logger.error("🚫 HTTP pipeline missing from app.state")
            raise Exception("Pipeline not available")
        
        chain = state.pipeline.get_chain()
        
        # 🟢 MLflow run scoped to this request
        with mlflow.start_run(run_name=f"http_{int(start_time)}", nested=True):
            mlflow.log_params({
                "doc_path": str(state.config.doc_path) if hasattr(state, 'config') else "unknown",
                "chunk_strategy": str(state.config.chunk_strategy) if hasattr(state, 'config') else "unknown"
            })
            
            response = await chain.ainvoke({"question": question})
            latency = time.time() - start_time
            
            mlflow.log_metric("http_latency", latency)
            
            logger.info(f"◼️ [HTTP] Response generated in {latency:.2f}s")
            return {"query": question, "response": response}
            
    except Exception as e:
        logger.error(f"❌ HTTP Error processing query: {e}")
        raise

@router.websocket("/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Get app from websocket scope
        app = websocket.scope.get("app")
        if not app:
            logger.error("🚫 [WS] No app found in websocket scope")
            await websocket.send_text("❌ Server error - no application context")
            return

        # Access the pipeline from app state
        state = app.state
        logger.info(f"📋 [WS] App state type: {type(state)}")
        
        if not hasattr(state, 'pipeline') or state.pipeline is None:
            logger.error("🚫 [WS] Pipeline missing from app.state")
            await websocket.send_text("❌ RAG pipeline not initialized. Check server logs.")
            return

        logger.info("🔌 [WS] Connection accepted and pipeline verified")
        
    except Exception as e:
        logger.error(f"🚫 [WS] Error accessing app state: {e}")
        await websocket.send_text("❌ Server error - please try again later.")
        return

    logger.info("🔌 [WS] Connection accepted.")
    
    try:
        while True:
            start_time = time.time()
            query = await websocket.receive_text()
            
            # 🟢 MLflow run scoped to this WS message
            with mlflow.start_run(run_name=f"ws_{int(time.time())}", nested=True):
                logger.info(f"▶️ [WS] Received: '{query[:50]}...'")
                
                try:
                    chain = state.pipeline.get_chain()
                except Exception as e:
                    logger.error(f"❌ Error getting chain from pipeline: {e}")
                    await websocket.send_text("❌ Error preparing response. Please try again.")
                    continue
                
                response_chunks = []
                async for chunk in chain.astream({"question": query}):
                    if isinstance(chunk, str):
                        response_chunks.append(chunk)
                        await websocket.send_text(chunk)
                    elif isinstance(chunk, dict) and 'answer' in chunk:
                        response_chunks.append(chunk['answer'])
                        await websocket.send_text(chunk['answer'])
                
                latency = time.time() - start_time
                mlflow.log_metric("ws_latency", latency)

                # Save WS response artifact
                artifact_dir = "mlflow_artifacts"
                os.makedirs(artifact_dir, exist_ok=True)
                filepath = os.path.join(artifact_dir, f"ws_resp_{int(time.time())}.json")
                with open(filepath, "w") as f:
                    json.dump({"query": query, "response": "".join(response_chunks)}, f)
                
                logger.info(f"📋 [WS] Response saved to {filepath}")
                await websocket.send_text("\n\n")
            
    except WebSocketDisconnect:
        logger.info("⚠️ [WS] Client disconnected.")
    except Exception as e:
        logger.error(f"❌ [WS] Error processing message: {e}")
        try: 
            await websocket.send_text(f"\n\n[Error: {str(e)}]")
        except: 
            pass