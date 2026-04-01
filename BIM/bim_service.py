"""
BIM Navigation Service — FastAPI
Port: 8001
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from bim.graph import BIMGraph, NodeNotFoundError, NoPathError
from bim.models import GraphInfo, NavigateRequest, PathResult, RoomInfo

# ── Khởi tạo ────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "rooms.json"

app = FastAPI(
    title="BIM Navigation Service",
    description="Pathfinding trong tòa nhà dùng NetworkX + Dijkstra",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bim = BIMGraph(DATA_PATH)

# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    info = bim.graph_info()
    return {
        "status": "ok",
        "building": info.building_name,
        "nodes": info.num_nodes,
        "edges": info.num_edges,
    }


@app.get("/rooms", response_model=list[RoomInfo])
def list_rooms():
    """Trả về danh sách tất cả phòng / điểm trong tòa nhà."""
    return bim.get_rooms()


@app.get("/graph/info", response_model=GraphInfo)
def graph_info():
    """Trả về thông tin đồ thị."""
    return bim.graph_info()


@app.post("/navigate", response_model=PathResult)
def navigate(req: NavigateRequest):
    """
    Tìm đường từ from_id → to_id.

    - `from_id` / `to_id`: ID node (VD: "entrance", "room_101") hoặc alias (VD: "cửa chính")
    - `preference`: "shortest" (mặc định) | "elevator" | "stairs"
    """
    try:
        result = bim.find_path(req.from_id, req.to_id, req.preference)
        return result
    except NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NoPathError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/navigate", response_model=PathResult)
def navigate_get(from_id: str, to_id: str, preference: str = "shortest"):
    """GET version của /navigate (tiện test nhanh trên browser)."""
    try:
        result = bim.find_path(from_id, to_id, preference)
        return result
    except NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NoPathError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bim_service:app", host="0.0.0.0", port=8001, reload=True)
