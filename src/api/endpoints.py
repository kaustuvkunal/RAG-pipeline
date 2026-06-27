from fastapi import APIRouter, WebSocket, Request, Depends
from ..pipeline.rag_pipeline import RAGPipeline
from ..logging_config import get_logger
import asyncio
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

logger = get_logger("endpoints")
router = APIRouter()

def get_app_state(request: Request) -> dict:
    return request.app.state

@router.get("/query/")
async def query_rag(question: str, state=Depends(get_app_state)):
    logger.info(f"▶️ [HTTP] Received query: '{question[:30]}...'")
    chain = state.pipeline.get_chain()
    response = await chain.ainvoke({"question": question})
    logger.info("◼️  [HTTP] Response generated successfully.")
    return {"query": question, "response": response}

@router.websocket("/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔌 [WS] Connection accepted.")
    
    try:
        while True:
            query = await websocket.receive_text()
            logger.info(f"▶️ [WS] Received Query: '{query[:30]}...'")

            chain = websocket.app.state.pipeline.get_chain()
            
            # 🚀 STEP 1: LOG RETRIEVAL START
            logger.info("⏳ [RETRIEVAL] Searching vector store...")
            
            # We use astream to get chunks. 
            # If you want to log retrieval completion, we can do that inside the loop 
            # or before it depending on where the retriever finishes.
            
            # 🚀 STEP 2: STREAM GENERATION
            logger.info("📡 [LLM] Generating response...")
            
            async for chunk in chain.astream({"question": query}):
                if isinstance(chunk, str):
                    await websocket.send_text(chunk)
            
            # Send a newline to format the output properly on frontend
            await websocket.send_text("\n\n")
            logger.info("✅ [WS] Response generation complete.")

    except WebSocketDisconnect:
        logger.info("⚠️ [WS] Client disconnected.")
    except Exception as e:
        logger.error(f"❌ [WS] Error: {e}")
        try:
            await websocket.send_text(f"\n\n[Error: {str(e)}]")
        except Exception:
            pass