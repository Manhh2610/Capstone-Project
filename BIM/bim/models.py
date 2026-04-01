from pydantic import BaseModel, Field
from typing import List, Optional


class NavigateRequest(BaseModel):
    from_id: str = Field(..., description="ID node xuất phát")
    to_id: str = Field(..., description="ID node đích")
    preference: str = Field(default="shortest", description="Tiêu chí: 'shortest' | 'elevator' | 'stairs'")


class Step(BaseModel):
    instruction: str = Field(..., description="Hướng dẫn bằng tiếng Việt")
    from_node: str = Field(..., description="ID node bắt đầu của bước")
    to_node: str = Field(..., description="ID node kết thúc của bước")
    distance: float = Field(..., description="Khoảng cách bước này (mét)")
    floor: int = Field(..., description="Tầng hiện tại")


class PathResult(BaseModel):
    from_id: str
    to_id: str
    from_name: str
    to_name: str
    path: List[str] = Field(..., description="Danh sách node ID theo thứ tự")
    steps: List[Step] = Field(..., description="Danh sách bước điều hướng")
    total_distance: float = Field(..., description="Tổng khoảng cách (mét)")
    estimated_time_seconds: int = Field(..., description="Thời gian ước tính (giây), tốc độ đi bộ 1 m/s")
    floor_changes: int = Field(..., description="Số lần đổi tầng")


class RoomInfo(BaseModel):
    id: str
    name: str
    type: str
    floor: int
    coordinates: dict
    aliases: List[str]


class GraphInfo(BaseModel):
    building_name: str
    num_nodes: int
    num_edges: int
    floors: int
    room_types: dict
