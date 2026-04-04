import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from receptionist_orchestrator import app

client = TestClient(app)

def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@patch("agents.receptionist_agent.ReceptionistAgent.process_user_input")
def test_process_navigation(mock_process):
    """Test navigation request routing."""
    mock_process.return_value = {
        "intent": "NAVIGATE",
        "speak": "Đi đến phòng sếp.",
        "data": {"path": ["entrance", "room_203"]}
    }
    
    response = client.post("/process", json={"text": "Dẫn tôi tới phòng sếp"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "NAVIGATE"
    assert "đi đến" in data["speak"].lower()

@patch("agents.receptionist_agent.ReceptionistAgent.process_user_input")
def test_process_chat(mock_process):
    """Test chat request routing."""
    mock_process.return_value = {
        "intent": "CHAT",
        "speak": "Chào bạn, tôi là robot lễ tân.",
        "data": None
    }
    
    response = client.post("/process", json={"text": "Chào robot"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "CHAT"
    assert "chào bạn" in data["speak"].lower()

def test_process_empty_input():
    """Test empty input error handling."""
    response = client.post("/process", json={"text": ""})
    assert response.status_code == 400
