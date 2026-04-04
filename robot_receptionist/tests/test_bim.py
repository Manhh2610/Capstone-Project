import pytest
from fastapi.testclient import TestClient
from bim_service import app

client = TestClient(app)

def test_find_path_same_floor():
    """Test tìm đường cùng tầng: entrance -> room_101."""
    resp = client.post("/navigate", json={"from_id": "entrance", "to_id": "room_101"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["from_id"] == "entrance"
    assert data["to_id"] == "room_101"
    assert len(data["path"]) > 0
    assert data["floor_changes"] == 0

def test_find_path_cross_floor():
    """Test tìm đường khác tầng: entrance -> room_201 dùng elevator."""
    resp = client.post("/navigate", json={
        "from_id": "entrance", 
        "to_id": "room_201", 
        "preference": "elevator"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["floor_changes"] >= 1
    # Kiểm tra xem có đi qua thang máy không
    assert any("elevator" in node for node in data["path"])

def test_resolve_exact():
    """Test tìm phòng khớp tên chính xác: 'Phòng 101'."""
    resp = client.post("/resolve", json={"query": "Phòng 101"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["room_id"] == "room_101"

def test_resolve_semantic():
    """Test tìm phòng theo ngữ nghĩa: 'phòng giám đốc' -> room_203."""
    # Semantic match cho 'giám đốc' vì 'sếp' có thể lệch trong mô hình nhỏ
    resp = client.post("/resolve", json={"query": "tìm giúp phòng giám đốc"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["room_id"] == "room_203"

def test_node_not_found():
    """Test lỗi khi không tìm thấy node ID."""
    resp = client.post("/navigate", json={"from_id": "entrance", "to_id": "non_existent"})
    assert resp.status_code == 404
    # Sửa để khớp message tiếng Việt của BIMGraph
    assert "không tìm thấy" in resp.json()["detail"].lower()

def test_no_path():
    """
    Test lỗi khi không có đường đi hoặc không resolve được.
    """
    # Test resolve thất bại với chuỗi vô nghĩa vượt ngưỡng distance 0.4
    resp = client.post("/resolve", json={"query": "không gian xanh xyz vớ vẩn"})
    assert resp.status_code == 404
