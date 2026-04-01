"""
Unit tests cho BIM Graph Engine + FastAPI Service
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

# ── Fixtures ─────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent / "data" / "rooms.json"


@pytest.fixture(scope="module")
def graph():
    from bim.graph import BIMGraph
    return BIMGraph(DATA_PATH)


@pytest.fixture(scope="module")
def client():
    from bim_service import app
    return TestClient(app)


# ── Graph construction ────────────────────────────────────────────────────────

class TestGraphLoading:
    def test_graph_loads(self, graph):
        """Graph load thành công từ rooms.json."""
        assert graph.G.number_of_nodes() > 0
        assert graph.G.number_of_edges() > 0

    def test_graph_has_expected_nodes(self, graph):
        assert "entrance" in graph.G.nodes
        assert "room_101" in graph.G.nodes
        assert "staircase" in graph.G.nodes
        assert "elevator" in graph.G.nodes

    def test_graph_info(self, graph):
        info = graph.graph_info()
        assert info.num_nodes >= 20
        assert info.num_edges >= 20
        assert info.floors == 2


# ── Alias resolution ──────────────────────────────────────────────────────────

class TestAliasResolution:
    def test_resolve_by_id(self, graph):
        assert graph.resolve_id("entrance") == "entrance"

    def test_resolve_by_alias(self, graph):
        assert graph.resolve_id("cửa chính") == "entrance"
        assert graph.resolve_id("thang máy") == "elevator"
        assert graph.resolve_id("phòng 101") == "room_101"

    def test_resolve_case_insensitive(self, graph):
        assert graph.resolve_id("ENTRANCE") == "entrance"

    def test_resolve_unknown_raises(self, graph):
        from bim.graph import NodeNotFoundError
        with pytest.raises(NodeNotFoundError):
            graph.resolve_id("phòng không tồn tại xyz")


# ── Pathfinding ───────────────────────────────────────────────────────────────

class TestPathfinding:
    def test_find_path_valid(self, graph):
        result = graph.find_path("entrance", "room_101")
        assert len(result.path) > 0
        assert result.path[0] == "entrance"
        assert result.path[-1] == "room_101"

    def test_find_path_same_node(self, graph):
        result = graph.find_path("entrance", "entrance")
        assert result.path == ["entrance"]
        assert result.total_distance == 0.0
        assert result.estimated_time_seconds == 0

    def test_find_path_cross_floor(self, graph):
        result = graph.find_path("entrance", "room_201")
        assert result.floor_changes >= 1
        # Path phải đi qua staircase hoặc elevator
        has_vertical = any(
            n in result.path for n in ["staircase", "elevator", "staircase_f1", "elevator_f1"]
        )
        assert has_vertical

    def test_find_path_returns_distance(self, graph):
        result = graph.find_path("entrance", "room_101")
        assert result.total_distance > 0

    def test_find_path_returns_steps(self, graph):
        result = graph.find_path("entrance", "room_101")
        assert len(result.steps) > 0
        for step in result.steps:
            assert step.instruction != ""
            assert step.distance >= 0

    def test_find_path_preference_elevator(self, graph):
        result = graph.find_path("entrance", "room_201", preference="elevator")
        assert "elevator" in result.path or "elevator_f1" in result.path

    def test_find_path_preference_stairs(self, graph):
        result = graph.find_path("entrance", "room_201", preference="stairs")
        assert "staircase" in result.path or "staircase_f1" in result.path

    def test_find_path_invalid_node(self, graph):
        from bim.graph import NodeNotFoundError
        with pytest.raises(NodeNotFoundError):
            graph.find_path("entrance", "phong_abc_xyz")

    def test_find_path_by_alias(self, graph):
        result = graph.find_path("lối vào", "phòng giám đốc")
        assert result.path[-1] == "room_203"


# ── Steps builder ─────────────────────────────────────────────────────────────

class TestSteps:
    def test_steps_not_empty(self, graph):
        result = graph.find_path("entrance", "room_104")
        assert len(result.steps) > 0

    def test_last_step_is_arrival(self, graph):
        result = graph.find_path("entrance", "room_101")
        last = result.steps[-1]
        assert "Đã đến" in last.instruction or "đến" in last.instruction.lower()

    def test_steps_distances_sum(self, graph):
        result = graph.find_path("entrance", "room_101")
        # Tổng distance trong steps ~ total_distance (bước cuối = 0)
        step_total = sum(s.distance for s in result.steps)
        assert abs(step_total - result.total_distance) < 0.5


# ── API Endpoints ─────────────────────────────────────────────────────────────

class TestAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_rooms(self, client):
        resp = client.get("/rooms")
        assert resp.status_code == 200
        rooms = resp.json()
        assert isinstance(rooms, list)
        assert len(rooms) > 0
        assert "id" in rooms[0]
        assert "name" in rooms[0]

    def test_graph_info(self, client):
        resp = client.get("/graph/info")
        assert resp.status_code == 200
        info = resp.json()
        assert info["num_nodes"] > 0

    def test_navigate_post(self, client):
        resp = client.post("/navigate", json={"from_id": "entrance", "to_id": "room_101"})
        assert resp.status_code == 200
        data = resp.json()
        assert "path" in data
        assert "steps" in data
        assert "total_distance" in data
        assert "estimated_time_seconds" in data

    def test_navigate_get(self, client):
        resp = client.get("/navigate?from_id=entrance&to_id=room_101")
        assert resp.status_code == 200

    def test_navigate_not_found(self, client):
        resp = client.post("/navigate", json={"from_id": "entrance", "to_id": "phong_xyz"})
        assert resp.status_code == 404

    def test_navigate_same_node(self, client):
        resp = client.post("/navigate", json={"from_id": "entrance", "to_id": "entrance"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_distance"] == 0.0

    def test_navigate_alias(self, client):
        resp = client.post("/navigate", json={"from_id": "cửa chính", "to_id": "phòng họp"})
        assert resp.status_code == 200
