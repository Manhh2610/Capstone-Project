"""
Receptionist Orchestrator Service
Port: 8000

Pipeline: STT → Intent/NER → BIM+RAG → LLM → [TTS + UI + Robot] (parallel)
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, List

from agents.receptionist_agent import ReceptionistAgent
from agents.tts_agent import TTSAgent
from agents.robot_signal_agent import RobotSignalAgent
from agents.stt_agent import STTAgent
from bim.models import WaypointPose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Agents ────────────────────────────────────────────────────────────────────

agent = ReceptionistAgent()
tts   = TTSAgent()
robot = RobotSignalAgent()
stt   = STTAgent()

# ── Startup lifecycle ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-render common TTS phrases at startup to reduce first-response latency."""
    logger.info("Orchestrator starting – pre-rendering TTS phrases...")
    await tts.prerender_common_phrases()
    logger.info("Orchestrator ready.")
    yield

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Receptionist Orchestrator",
    description=(
        "Central brain of the receptionist robot.\n\n"
        "Pipeline: STT → Intent/NER → BIM+RAG → LLM → TTS + UI + Robot (parallel)"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class UserInput(BaseModel):
    text: str
    speak: bool = True     # Phát âm qua TTS
    drive: bool = True     # Gửi tín hiệu đến robot

class DispatchResult(BaseModel):
    intent: str
    speak: str
    tts_played: bool   = False
    robot_sent: bool   = False
    data: Optional[Any] = None

# ── Parallel Dispatcher ───────────────────────────────────────────────────────

async def _dispatch(agent_response: dict, speak: bool, drive: bool) -> DispatchResult:
    """
    Dispatch agent response đồng thời đến 3 thành phần:
      1. TTS (Piper)    – phát âm câu speak
      2. UI (Kiosk)     – trả về qua HTTP response (không cần thêm agent)
      3. Robot (ROS2)   – gửi waypoints nếu có
    """
    speak_text = agent_response.get("speak", "")
    waypoints: List[WaypointPose] = []

    # Extract waypoints từ navigation data nếu có
    data = agent_response.get("data")
    if isinstance(data, dict) and "waypoints" in data:
        raw_wps = data.get("waypoints", [])
        waypoints = [WaypointPose(**wp) if isinstance(wp, dict) else wp for wp in raw_wps]

    # ── Tạo coroutines cho 2 side effects ────────────────────────────────────
    tasks = []

    # Task 1: TTS
    if speak and speak_text:
        tasks.append(tts.speak(speak_text))
    else:
        tasks.append(asyncio.sleep(0))   # no-op

    # Task 2: Robot signal
    if drive and waypoints:
        tasks.append(robot.send_waypoints(waypoints))
    else:
        tasks.append(asyncio.sleep(0))   # no-op

    # ── Chạy song song ────────────────────────────────────────────────────────
    results = await asyncio.gather(*tasks, return_exceptions=True)

    tts_ok   = not isinstance(results[0], Exception)
    robot_ok = isinstance(results[1], bool) and results[1]

    return DispatchResult(
        intent=agent_response.get("intent", "UNKNOWN"),
        speak=speak_text,
        tts_played=tts_ok,
        robot_sent=robot_ok,
        data=data,
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "orchestrator": "online",
        "tts_available": tts._piper_available,
        "robot_bridge": robot.bridge_url,
    }


@app.post("/process", response_model=DispatchResult)
async def process_user_input(input_data: UserInput):
    """
    Điểm vào chính của pipeline.

    1. Intent Detection (LLM)
    2. Action (BIM navigate / RAG chat)
    3. Parallel Dispatch → TTS + UI response + Robot signal
    """
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input is required")

    try:
        # Step 1+2: Agent processes input
        agent_response = await agent.process_user_input(input_data.text)

        # Step 3: Parallel dispatch
        result = await _dispatch(
            agent_response,
            speak=input_data.speak,
            drive=input_data.drive,
        )
        return result

    except Exception as e:
        logger.exception("Orchestrator error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speak")
async def speak_direct(text: str):
    """Direct TTS endpoint – phát âm văn bản bất kỳ (bypass agent)."""
    wav = await tts.speak(text)
    return {"status": "ok", "text": text, "wav": str(wav) if wav else None}


@app.post("/listen", response_model=DispatchResult)
async def listen_audio(
    audio: bytes,
    language: str = "vi",
    speak: bool = True,
    drive: bool = True,
):
    """
    Full pipeline từ audio thô (WAV bytes):
    Microphone bytes → STT (Whisper) → Intent → BIM+RAG → TTS+Robot (parallel)
    """
    text = await stt.transcribe_audio(audio, language=language)
    if not text:
        raise HTTPException(status_code=422, detail="STT could not transcribe audio.")

    agent_response = await agent.process_user_input(text)
    return await _dispatch(agent_response, speak=speak, drive=drive)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("receptionist_orchestrator:app", host="0.0.0.0", port=8000, reload=True)
