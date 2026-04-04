"""
Pipeline Integration Tests

Test toàn bộ luồng: Intent → BIM → Dispatch → TTS/Robot
Dùng mock để không phụ thuộc Ollama/Piper/ROS2 khi CI/CD.
"""
import asyncio
import math
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from receptionist_orchestrator import app as orch_app
from bim_service import app as bim_app
from agents.tts_agent import TTSAgent
from agents.robot_signal_agent import RobotSignalAgent
from agents.stt_agent import STTAgent
from bim.models import WaypointPose

orch_client = TestClient(orch_app)
bim_client  = TestClient(bim_app)

# ── Waypoint Tests ────────────────────────────────────────────────────────────

def test_waypoint_endpoint_valid():
    """Test GET /waypoint/{room_id} trả về đúng toạ độ."""
    resp = bim_client.get("/waypoint/entrance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "entrance"
    assert data["x"] == 0.0
    assert data["y"] == 0.0
    assert "theta" in data
    assert data["frame"] == "map"

def test_waypoint_endpoint_by_alias():
    """Test waypoint resolve từ alias tiếng Việt."""
    resp = bim_client.get("/waypoint/cửa chính")
    assert resp.status_code == 200
    assert resp.json()["node_id"] == "entrance"

def test_waypoint_endpoint_not_found():
    """Test waypoint không tồn tại → 404."""
    resp = bim_client.get("/waypoint/phong_ma")
    assert resp.status_code == 404

# ── Robot Signal Agent Tests ──────────────────────────────────────────────────

def test_robot_signal_quaternion():
    """Test quaternion conversion: theta=π/2 → z=0.707, w=0.707."""
    from agents.robot_signal_agent import RobotSignalAgent
    rob = RobotSignalAgent()
    theta = math.pi / 2
    wp = WaypointPose(node_id="test", name="Test", x=1.0, y=2.0, theta=theta, frame="map")
    payload = rob._to_ros2_payload([wp])
    quat = payload["poses"][0]["pose"]["orientation"]
    assert abs(quat["z"] - math.sin(theta / 2)) < 1e-4
    assert abs(quat["w"] - math.cos(theta / 2)) < 1e-4

@pytest.mark.asyncio
async def test_robot_signal_offline_graceful():
    """Test robot bridge offline → simulation mode, không raise exception."""
    rob = RobotSignalAgent(bridge_url="http://localhost:1")  # unreachable
    wps = [WaypointPose(node_id="room_101", name="Phòng 101", x=0.0, y=20.0, theta=-1.5708, frame="map")]
    result = await rob.send_waypoints(wps)
    assert result is False  # Graceful failure

# ── TTS Agent Tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tts_cache_deterministic():
    """Test that cache path is same for identical text."""
    tts = TTSAgent()
    p1 = tts._cache_path("Xin chào bạn")
    p2 = tts._cache_path("Xin chào bạn")
    assert p1 == p2

@pytest.mark.asyncio
async def test_tts_simulation_mode():
    """Test TTS in simulation (no Piper) does not crash."""
    tts = TTSAgent()
    tts._piper_available = False  # Force simulation
    result = await tts.synthesize("Thử nghiệm tổng hợp giọng nói.")
    assert result is None  # No WAV in simulation

# ── STT Agent Tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stt_offline_graceful():
    """Test STT agent returns empty string when both server and CLI unavailable."""
    stt = STTAgent(server_url="http://localhost:1", binary="nonexistent-binary")
    result = await stt.transcribe_audio(b"fake_audio_bytes")
    assert result == ""

# ── Full Orchestrator Pipeline Tests ─────────────────────────────────────────

@patch("agents.receptionist_agent.ReceptionistAgent.process_user_input")
def test_pipeline_navigate_dispatch(mock_process):
    """Test /process with navigation intent triggers correct response structure."""
    mock_process.return_value = {
        "intent": "NAVIGATE",
        "speak": "Đi đến phòng 101.",
        "data": {
            "path": ["entrance", "room_101"],
            "waypoints": [
                {"node_id": "entrance", "name": "Lối vào", "x": 0.0, "y": 0.0, "theta": 1.5708, "frame": "map"},
                {"node_id": "room_101", "name": "Phòng 101", "x": 0.0, "y": 20.0, "theta": -1.5708, "frame": "map"},
            ]
        }
    }
    resp = orch_client.post("/process", json={"text": "Dẫn tôi đến phòng 101", "speak": False, "drive": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "NAVIGATE"
    assert "đi đến" in data["speak"].lower()

@patch("agents.receptionist_agent.ReceptionistAgent.process_user_input")
def test_pipeline_chat_dispatch(mock_process):
    """Test /process with chat intent returns speak text."""
    mock_process.return_value = {
        "intent": "CHAT",
        "speak": "Xin chào! Tôi có thể giúp gì cho bạn?",
        "data": None
    }
    resp = orch_client.post("/process", json={"text": "Chào bạn", "speak": False, "drive": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "CHAT"
    assert len(data["speak"]) > 0

def test_pipeline_empty_input():
    """Test /process với input rỗng → 400 Bad Request."""
    resp = orch_client.post("/process", json={"text": "  "})
    assert resp.status_code == 400
