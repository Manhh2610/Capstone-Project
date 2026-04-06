import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from orchestrator import app

client = TestClient(app)

# ── Pipeline Tests with Mocks ───────────────────────────────

@pytest.fixture(autouse=True)
def mock_llm_responses():
    """Mock the LLM to test the pipeline synchronously without waiting for Ollama."""
    with patch("llm.connector.format_navigation", new_callable=AsyncMock) as mock_nav:
        mock_nav.return_value = "Từ đây đi bộ đến phòng đó."
        
        with patch("llm.connector.answer_faq", new_callable=AsyncMock) as mock_faq:
            mock_faq.return_value = "Wifi là COMPANY_GUEST."
            
            with patch("tts.piper_svc.speak") as mock_speak:
                yield mock_nav, mock_faq, mock_speak


def test_navigation_query():
    """Test valid navigation query."""
    response = client.post("/query", json={"text": "đến phòng 101"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "navigation"
    assert "speak" in data
    assert data["destination"]["id"] == "room_101"
    assert "steps" in data

def test_faq_query():
    """Test standard FAQ intent."""
    response = client.post("/query", json={"text": "wifi là gì"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "faq"
    assert data["speak"] == "Wifi là COMPANY_GUEST."
    assert "display" in data

def test_greeting_query():
    """Test standard Greeting intent."""
    response = client.post("/query", json={"text": "xin chào"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "greeting"
    assert "giúp gì" in data["speak"]

def test_unknown_query():
    """Test unknown jargon that isn't matched by any intent regex."""
    response = client.post("/query", json={"text": "hôm nay trời thật đẹp"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "clarify"
    assert "không tìm thấy" in data["speak"] or "nói rõ hơn" in data["speak"]

def test_room_not_found():
    """Test navigation with an unresolvable room name."""
    response = client.post("/query", json={"text": "đến phòng ở vũ trụ"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "clarify"
    assert "Xin lỗi, tôi không tìm thấy phòng đó" in data["speak"]
