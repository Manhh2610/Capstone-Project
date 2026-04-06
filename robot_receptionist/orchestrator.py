"""
Main Orchestrator tying all pipeline components together.
Runs a FastAPI server handling text and voice queries.
"""
import io
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from bim.graph import BIMGraph, NodeNotFoundError, NoPathError
from bim.resolver import RoomResolver
from nlp.intent import classify_intent, extract_room_query
from llm.connector import format_navigation, answer_faq
from stt.whisper_svc import transcribe_file
import tts.piper_svc as piper_svc

app = FastAPI(title="Smart Receptionist Backend")

# Initialize global components
ROOMS_PATH = "data/rooms.json"
try:
    with open(ROOMS_PATH, "r", encoding="utf-8") as f:
        rooms_json = json.load(f)
        
    bim = BIMGraph(ROOMS_PATH)
    resolver = RoomResolver(rooms_json.get("nodes", []))
except Exception as e:
    print(f"Failed to load BIM Graph or Resolver: {e}")
    bim = None
    resolver = None

# Global dictionary to hold active websocket tasks for cancellation
USER_TASKS = {}

class QueryRequest(BaseModel):
    text: str

def clarify_response() -> dict:
    return {
        "type": "clarify",
        "speak": "Xin lỗi, tôi không tìm thấy phòng đó. Bạn có thể nói rõ hơn không?"
    }

def greeting_response() -> dict:
    return {
        "type": "greeting",
        "speak": "Xin chào! Tôi có thể giúp gì cho bạn?"
    }

def faq_response(speak: str) -> dict:
    return {
        "type": "faq",
        "speak": speak,
        "display": speak
    }

def navigation_response(path_result, speak: str) -> dict:
    return {
        "type": "navigation",
        "speak": speak,
        "destination": {
            "id": path_result.to_id,
            "name": path_result.to_name,
            "floor": path_result.steps[-1].floor if path_result.steps else 0
        },
        "steps": [
            {
                "instruction": s.instruction,
                "distance": s.distance,
                "floor": s.floor
            } for s in path_result.steps
        ],
        "path": path_result.path,
        "floor_to_show": path_result.steps[-1].floor if path_result.steps else 0,
        "total_distance": path_result.total_distance,
        "estimated_time_seconds": path_result.estimated_time_seconds,
        "cached": False
    }

async def process_pipeline(text: str) -> dict:
    """Core pipeline execution."""
    if not text.strip():
        return clarify_response()
        
    intent = classify_intent(text)
    
    if intent == "navigation":
        room_query = extract_room_query(text)
        room_id = resolver.resolve(room_query)
        if not room_id:
            return clarify_response()
            
        try:
            path_result = bim.find_path("entrance", room_id)
            steps_text = [s.instruction for s in path_result.steps[:-1]] # skip the last 'arrived' step for LLM
            if not steps_text:
                steps_text = [path_result.steps[0].instruction]
            
            speak = await format_navigation(steps_text, path_result.to_name)
            
            # Optional: TTS rendering thread
            # asyncio.to_thread(piper_svc.speak, speak)
            
            return navigation_response(path_result, speak)
        except (NodeNotFoundError, NoPathError):
            return clarify_response()
            
    elif intent == "faq":
        speak = await answer_faq(text)
        return faq_response(speak)
        
    elif intent == "greeting":
        return greeting_response()
        
    else:
        return clarify_response()

@app.post("/query")
async def handle_query(req: QueryRequest):
    """HTTP endpoint for text queries."""
    response_data = await process_pipeline(req.text)
    return JSONResponse(content=response_data)

@app.post("/voice")
async def handle_voice(audio: UploadFile = File(...)):
    """HTTP endpoint for audio queries."""
    import tempfile
    import os
    
    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        # Transcribe
        text = await asyncio.to_thread(transcribe_file, tmp_path)
        if not text:
            return JSONResponse(content=clarify_response())
            
        # Process
        response_data = await process_pipeline(text)
        return JSONResponse(content=response_data)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint handling interruption."""
    await websocket.accept()
    client_id = id(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Cancel existing task if any
            if client_id in USER_TASKS and not USER_TASKS[client_id].done():
                USER_TASKS[client_id].cancel()
                
            async def run_and_send():
                try:
                    resp = await process_pipeline(data)
                    await websocket.send_json(resp)
                    # Trigger voice asynchronously without blocking the loop
                    # loop.run_in_executor(None, piper_svc.speak, resp.get("speak", ""))
                except asyncio.CancelledError:
                    pass
                    
            USER_TASKS[client_id] = asyncio.create_task(run_and_send())
            
    except WebSocketDisconnect:
        if client_id in USER_TASKS:
            USER_TASKS[client_id].cancel()
            del USER_TASKS[client_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
