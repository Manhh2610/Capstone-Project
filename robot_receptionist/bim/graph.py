"""
BIM Graph Engine
Xây dựng đồ thị NavigX từ rooms.json, tìm đường bằng Dijkstra,
trả về danh sách bước điều hướng tiếng Việt.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from bim.models import PathResult, RoomInfo, Step, GraphInfo


# ─── Hằng số ────────────────────────────────────────────────────────────────
WALK_SPEED_MS = 1.0          # tốc độ đi bộ (mét/giây)
FLOOR_CHANGE_PENALTY = 0     # không phạt đổi tầng (đã tính trong khoảng cách)

NODE_TYPE_LABELS = {
    "entrance":     "lối vào",
    "lobby":        "sảnh",
    "corridor":     "hành lang",
    "room":         "phòng",
    "meeting_room": "phòng họp",
    "toilet":       "nhà vệ sinh",
    "staircase":    "cầu thang bộ",
    "elevator":     "thang máy",
}


# ─── Helper geometry ────────────────────────────────────────────────────────

def _euclidean(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


# ─── Instruction builder ────────────────────────────────────────────────────

def _build_instruction(
    prev_node: Optional[dict],
    curr_node: dict,
    next_node: dict,
    distance: float,
) -> str:
    """
    Tạo câu hướng dẫn điều hướng tiếng Việt cho bước curr→next.
    """
    curr_type = curr_node.get("type", "")
    next_type = next_node.get("type", "")
    next_name = next_node["name"]
    curr_floor = curr_node.get("floor", 0)
    next_floor = next_node.get("floor", curr_floor)

    # Đổi tầng
    if next_type == "staircase" or curr_type == "staircase":
        if next_floor != curr_floor:
            direction = "lên" if next_floor > curr_floor else "xuống"
            return f"Đi theo cầu thang bộ {direction} tầng {next_floor + 1}"
        if curr_type == "staircase" and next_type not in ("staircase", "elevator"):
            return f"Ra khỏi cầu thang, đi đến {next_name} (~{distance:.0f}m)"

    if next_type == "elevator" or curr_type == "elevator":
        if next_floor != curr_floor:
            direction = "lên" if next_floor > curr_floor else "xuống"
            return f"Lên thang máy, {direction} tầng {next_floor + 1}"
        if curr_type == "elevator" and next_type not in ("staircase", "elevator"):
            return f"Ra khỏi thang máy, đi đến {next_name} (~{distance:.0f}m)"

    # Điểm đích cuối cùng
    if next_type in ("room", "meeting_room"):
        return f"Đến {next_name} (bên {'trái' if _relative_side(curr_node, next_node) < 0 else 'phải'}) (~{distance:.0f}m)"

    if next_type == "toilet":
        return f"Đến {next_name} (~{distance:.0f}m)"

    if next_type == "entrance":
        return f"Đi ra lối vào chính (~{distance:.0f}m)"

    if next_type == "lobby":
        return f"Đến {next_name} (~{distance:.0f}m)"

    # Hành lang / mặc định
    side = _relative_side(curr_node, next_node)
    if prev_node is not None:
        turn = _turn_description(prev_node, curr_node, next_node)
        return f"{turn} đến {next_name} (~{distance:.0f}m)"

    return f"Đi thẳng đến {next_name} (~{distance:.0f}m)"


def _relative_side(a: dict, b: dict) -> float:
    """Dương = phải, âm = trái (đơn giản theo trục X)."""
    return b["coordinates"]["x"] - a["coordinates"]["x"]


def _turn_description(prev: dict, curr: dict, nxt: dict) -> str:
    """Ước lượng hướng rẽ dựa trên vector."""
    v1x = curr["coordinates"]["x"] - prev["coordinates"]["x"]
    v1y = curr["coordinates"]["y"] - prev["coordinates"]["y"]
    v2x = nxt["coordinates"]["x"] - curr["coordinates"]["x"]
    v2y = nxt["coordinates"]["y"] - curr["coordinates"]["y"]
    cross = v1x * v2y - v1y * v2x
    dot = v1x * v2x + v1y * v2y

    if abs(cross) < 1e-6:
        return "Đi thẳng"
    if cross > 0:
        return "Rẽ trái"
    return "Rẽ phải"


# ─── BIMGraph ────────────────────────────────────────────────────────────────

class NodeNotFoundError(Exception):
    pass


class NoPathError(Exception):
    pass


class BIMGraph:
    """
    Đọc rooms.json → xây NetworkX graph → Dijkstra pathfinding → steps list.
    """

    def __init__(self, data_path: str | Path):
        self._data_path = Path(data_path)
        self.G: nx.Graph = nx.Graph()
        self._nodes: Dict[str, dict] = {}      # id → node dict
        self._alias_map: Dict[str, str] = {}   # alias (lower) → node id
        self._building_meta: dict = {}
        self._load()

    # ── Load ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        with open(self._data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._building_meta = data.get("building", {})

        for node in data["nodes"]:
            nid = node["id"]
            self._nodes[nid] = node
            self.G.add_node(nid, **node)

            # Build alias map (lowercase search)
            self._alias_map[nid.lower()] = nid
            self._alias_map[node["name"].lower()] = nid
            for alias in node.get("aliases", []):
                self._alias_map[alias.lower()] = nid

        for edge in data["edges"]:
            src, dst, dist = edge["from"], edge["to"], edge["distance"]
            self.G.add_edge(src, dst, weight=dist)
            if edge.get("bidirectional", True):
                self.G.add_edge(dst, src, weight=dist)

    # ── Public helpers ────────────────────────────────────────────────────

    def resolve_id(self, query: str) -> str:
        """
        Tìm node id từ id chính xác hoặc alias (không phân biệt hoa thường).
        Raise NodeNotFoundError nếu không tìm thấy.
        """
        resolved = self._alias_map.get(query.strip().lower())
        if resolved is None:
            raise NodeNotFoundError(
                f"Không tìm thấy node '{query}'. "
                f"Gợi ý: {', '.join(list(self._nodes.keys())[:5])}..."
            )
        return resolved

    def get_rooms(self) -> List[RoomInfo]:
        return [
            RoomInfo(**{k: v for k, v in node.items()})
            for node in self._nodes.values()
        ]

    def graph_info(self) -> GraphInfo:
        type_counts: Dict[str, int] = {}
        for node in self._nodes.values():
            t = node.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        floors = {n.get("floor", 0) for n in self._nodes.values()}

        return GraphInfo(
            building_name=self._building_meta.get("name", "BIM Building"),
            num_nodes=self.G.number_of_nodes(),
            num_edges=self.G.number_of_edges(),
            floors=len(floors),
            room_types=type_counts,
        )

    # ── Pathfinding ───────────────────────────────────────────────────────

    def find_path(
        self,
        from_id: str,
        to_id: str,
        preference: str = "shortest",
    ) -> PathResult:
        """
        Tìm đường từ from_id → to_id bằng Dijkstra.
        preference: 'shortest' | 'elevator' | 'stairs'
        """
        # Resolve aliases
        from_id = self.resolve_id(from_id)
        to_id = self.resolve_id(to_id)

        if from_id == to_id:
            node = self._nodes[from_id]
            return PathResult(
                from_id=from_id,
                to_id=to_id,
                from_name=node["name"],
                to_name=node["name"],
                path=[from_id],
                steps=[Step(
                    instruction="Bạn đang ở đây rồi!",
                    from_node=from_id,
                    to_node=to_id,
                    distance=0.0,
                    floor=node.get("floor", 0),
                )],
                total_distance=0.0,
                estimated_time_seconds=0,
                floor_changes=0,
            )

        # Có thể tuỳ chỉnh weight theo preference
        weight_fn = self._weight_fn(preference)

        try:
            path: List[str] = nx.dijkstra_path(
                self.G, from_id, to_id, weight=weight_fn
            )
            path_length: float = nx.dijkstra_path_length(
                self.G, from_id, to_id, weight=weight_fn
            )
        except nx.NetworkXNoPath:
            raise NoPathError(f"Không có đường từ '{from_id}' đến '{to_id}'.")
        except nx.NodeNotFound as e:
            raise NodeNotFoundError(str(e))

        steps = self._build_steps(path)
        floor_changes = self._count_floor_changes(path)

        return PathResult(
            from_id=from_id,
            to_id=to_id,
            from_name=self._nodes[from_id]["name"],
            to_name=self._nodes[to_id]["name"],
            path=path,
            steps=steps,
            total_distance=round(path_length, 1),
            estimated_time_seconds=int(path_length / WALK_SPEED_MS),
            floor_changes=floor_changes,
        )

    def _weight_fn(self, preference: str):
        """Trả về hàm weight cho Dijkstra theo preference."""
        avoid_type = None
        if preference == "stairs":
            avoid_type = "elevator"
        elif preference == "elevator":
            avoid_type = "staircase"

        if avoid_type is None:
            return "weight"  # dùng weight mặc định

        def _fn(u, v, data):
            penalty = 0.0
            for node_id in (u, v):
                if self._nodes.get(node_id, {}).get("type") == avoid_type:
                    penalty += 1000  # phạt nặng để tránh node đó
            return data.get("weight", 1.0) + penalty

        return _fn

    # ── Steps builder ─────────────────────────────────────────────────────

    def _build_steps(self, path: List[str]) -> List[Step]:
        if len(path) == 1:
            return []

        steps: List[Step] = []
        for i in range(len(path) - 1):
            prev = self._nodes[path[i - 1]] if i > 0 else None
            curr = self._nodes[path[i]]
            nxt  = self._nodes[path[i + 1]]

            edge_data = self.G.get_edge_data(curr["id"], nxt["id"]) or {}
            dist = edge_data.get("weight", 0.0)

            instruction = _build_instruction(prev, curr, nxt, dist)

            steps.append(Step(
                instruction=instruction,
                from_node=curr["id"],
                to_node=nxt["id"],
                distance=round(dist, 1),
                floor=curr.get("floor", 0),
            ))

        # Bước cuối: thông báo đến nơi
        last_node = self._nodes[path[-1]]
        steps.append(Step(
            instruction=f"✅ Đã đến {last_node['name']}",
            from_node=path[-2] if len(path) > 1 else path[-1],
            to_node=path[-1],
            distance=0.0,
            floor=last_node.get("floor", 0),
        ))

        return steps

    def _count_floor_changes(self, path: List[str]) -> int:
        changes = 0
        for i in range(1, len(path)):
            f_prev = self._nodes[path[i - 1]].get("floor", 0)
            f_curr = self._nodes[path[i]].get("floor", 0)
            if f_prev != f_curr:
                changes += 1
        return changes
