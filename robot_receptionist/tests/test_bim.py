import pytest
import json
from bim.graph import BIMGraph, NodeNotFoundError, NoPathError
from bim.resolver import RoomResolver

# Shared fixtures
@pytest.fixture(scope="module")
def bim_instance():
    return BIMGraph("data/rooms.json")

@pytest.fixture(scope="module")
def resolver_instance(bim_instance):
    with open("data/rooms.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return RoomResolver(data.get("nodes", []))

# ── BIM Graph Tests ─────────────────────────────────────────

def test_find_path_same_floor(bim_instance):
    """Test finding a path on the same floor."""
    result = bim_instance.find_path("entrance", "room_101")
    assert result.from_id == "entrance"
    assert result.to_id == "room_101"
    assert len(result.steps) > 1
    assert result.floor_changes == 0

def test_find_path_cross_floor_elevator(bim_instance):
    """Test finding a cross-floor path using the elevator."""
    result = bim_instance.find_path("entrance", "room_201", preference="elevator")
    assert result.to_id == "room_201"
    # Ensure elevator is in the path
    types = [bim_instance._nodes[node]["type"] for node in result.path]
    assert "elevator" in types

def test_find_path_cross_floor_stairs(bim_instance):
    """Test finding a cross-floor path using stairs."""
    result = bim_instance.find_path("entrance", "room_201", preference="stairs")
    assert result.to_id == "room_201"
    types = [bim_instance._nodes[node]["type"] for node in result.path]
    assert "staircase" in types

def test_node_not_found(bim_instance):
    """Test asking for an invalid node."""
    with pytest.raises(NodeNotFoundError):
        bim_instance.find_path("entrance", "invalid_node_id")

def test_same_node(bim_instance):
    """Test routing from entrance to entrance."""
    result = bim_instance.find_path("entrance", "entrance")
    assert result.total_distance == 0.0
    assert len(result.steps) == 1
    assert "Bạn đang ở đây rồi!" in result.steps[0].instruction

# ── Resolver Tests ──────────────────────────────────────────

def test_resolve_exact(resolver_instance):
    """Test exact matching ignores case."""
    room_id = resolver_instance.resolve("phòng 101")
    assert room_id == "room_101"

def test_resolve_alias(resolver_instance):
    """Test exact matching via an alias."""
    room_id = resolver_instance.resolve("wc")
    assert room_id == "toilet_f0" or room_id == "toilet_f1" # Assuming one of them triggers exact match

def test_resolve_semantic(resolver_instance):
    """Test semantic search via ChromaDB."""
    # Depends on sentence-transformers and Chromadb
    room_id = resolver_instance.resolve("chỗ sếp làm việc")
    assert room_id == "room_203" # "Phòng Giám đốc" -> "room_203"

def test_resolve_not_found(resolver_instance):
    """Test fallback when no node matches semantically or exactly."""
    room_id = resolver_instance.resolve("xyz không tồn tại")
    assert room_id is None
