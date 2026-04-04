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


class WaypointPose(BaseModel):
    """ROS2 Nav2 compatible pose for a single navigation waypoint."""
    node_id: str = Field(..., description="ID node trong BIM graph")
    name: str = Field(..., description="Tên điểm dừng")
    x: float = Field(..., description="Tọa độ X (mét)")
    y: float = Field(..., description="Tọa độ Y (mét)")
    theta: float = Field(..., description="Góc hướng (radian). 0=Đông, π/2=Bắc, -π/2=Nam, π=Tây")
    frame: str = Field(default="map", description="Reference frame cho ROS2")


class PathResultWithSpeak(PathResult):
    speak: str = Field(..., description="Câu nói tự nhiên tiếng Việt cho robot")
    waypoints: List[WaypointPose] = Field(default=[], description="Danh sách waypoint (x,y,θ) cho ROS2 Nav2")


class RoomInfo(BaseModel):
    id: str
    name: str
    type: str
    floor: int
    coordinates: dict
    aliases: List[str]


class ResolveRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi tự nhiên tìm phòng")


class ResolveResponse(BaseModel):
    room_id: str = Field(..., description="ID phòng tìm được")
    name: str = Field(..., description="Tên thực tế của phòng")


class GraphInfo(BaseModel):
    building_name: str
    num_nodes: int
    num_edges: int
    floors: int
    room_types: dict
