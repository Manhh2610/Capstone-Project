"""
BIM Navigation Service — FastAPI
Port: 8001
"""

from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from bim.graph import BIMGraph, NodeNotFoundError, NoPathError
from bim.models import (
    GraphInfo, NavigateRequest, PathResult, PathResultWithSpeak,
    RoomInfo, ResolveRequest, ResolveResponse, WaypointPose
)
from bim.resolver import RoomResolver
from bim.llm_connector import format_navigation

# ── Khởi tạo ────────────────────────────────────────────────────────────────

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data" / "rooms.json"
WAYPOINTS_PATH = BASE_PATH / "data" / "waypoints.json"

app = FastAPI(
    title="BIM Navigation Service",
    description="Pathfinding trong tòa nhà dùng NetworkX + Dijkstra",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    rooms_data = json.load(f)

with open(WAYPOINTS_PATH, "r", encoding="utf-8") as f:
    waypoints_data = json.load(f).get("waypoints", {})

bim = BIMGraph(DATA_PATH)
resolver = RoomResolver(rooms_data)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_waypoints_for_path(path: list[str]) -> list[WaypointPose]:
    """Convert a list of node IDs to a list of ROS2 WaypointPose objects."""
    waypoints = []
    for node_id in path:
        wp = waypoints_data.get(node_id)
        if wp:
            waypoints.append(WaypointPose(
                node_id=node_id,
                name=wp.get("name", node_id),
                x=wp["x"],
                y=wp["y"],
                theta=wp["theta"],
                frame=wp.get("frame", "map")
            ))
    return waypoints

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    info = bim.graph_info()
    return {
        "status": "ok",
        "building": info.building_name,
        "nodes": info.num_nodes,
        "edges": info.num_edges,
        "waypoints_loaded": len(waypoints_data),
    }


@app.get("/rooms", response_model=list[RoomInfo])
def list_rooms():
    """Trả về danh sách tất cả phòng / điểm trong tòa nhà."""
    return bim.get_rooms()


@app.get("/graph/info", response_model=GraphInfo)
def graph_info():
    """Trả về thông tin đồ thị."""
    return bim.graph_info()


@app.get("/waypoint/{room_id}", response_model=WaypointPose)
def get_waypoint(room_id: str):
    """Trả về tọa độ ROS2 Nav2 (x, y, θ) của một node."""
    # Resolve alias first
    try:
        node_id = bim.resolve_id(room_id)
    except NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    wp = waypoints_data.get(node_id)
    if not wp:
        raise HTTPException(status_code=404, detail=f"Không có waypoint cho node '{node_id}'")

    return WaypointPose(
        node_id=node_id,
        name=wp.get("name", node_id),
        x=wp["x"], y=wp["y"], theta=wp["theta"],
        frame=wp.get("frame", "map")
    )


@app.post("/resolve", response_model=ResolveResponse)
def resolve_room(req: ResolveRequest):
    """Tìm room_id từ câu nói tự nhiên dùng semantic search."""
    room_id = resolver.resolve(req.query)
    if not room_id:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy phòng nào khớp với '{req.query}'")

    room_info = next((n for n in rooms_data["nodes"] if n["id"] == room_id), None)
    return ResolveResponse(room_id=room_id, name=room_info["name"] if room_info else room_id)


@app.post("/navigate", response_model=PathResult)
def navigate(req: NavigateRequest):
    """Tìm đường từ from_id → to_id."""
    try:
        return bim.find_path(req.from_id, req.to_id, req.preference)
    except NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NoPathError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/navigate/speak", response_model=PathResultWithSpeak)
async def navigate_speak(req: NavigateRequest):
    """
    Tìm đường + sinh câu hướng dẫn tiếng Việt + waypoints cho ROS2 Nav2.
    """
    try:
        result = bim.find_path(req.from_id, req.to_id, req.preference)
        steps_dicts = [step.dict() for step in result.steps]
        speak_text = await format_navigation(steps_dicts, result.to_name)
        waypoints = _get_waypoints_for_path(result.path)

        return PathResultWithSpeak(
            **result.dict(),
            speak=speak_text,
            waypoints=waypoints
        )
    except NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NoPathError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/navigate", response_model=PathResult)
def navigate_get(from_id: str, to_id: str, preference: str = "shortest"):
    """GET version của /navigate (tiện test nhanh trên browser)."""
    try:
        return bim.find_path(from_id, to_id, preference)
    except NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NoPathError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bim_service:app", host="0.0.0.0", port=8001, reload=True)
