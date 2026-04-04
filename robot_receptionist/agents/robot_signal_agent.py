"""
Robot Signal Agent – Bridge sang ROS2 Nav2.

Gửi danh sách waypoint từ BIM đến robot thật qua:
  - HTTP (rosbridge / custom REST endpoint), hoặc
  - MQTT broker (nếu dùng micro-ROS over MQTT)

Mặc định: HTTP POST đến rosbridge-compatible endpoint.
"""

import asyncio
import httpx
import logging
from typing import List

from bim.models import WaypointPose

logger = logging.getLogger(__name__)

# ── Cấu hình ─────────────────────────────────────────────────────────────────

ROBOT_BRIDGE_URL = "http://localhost:11311/navigate"   # rosbridge REST endpoint
ROBOT_TIMEOUT = 5.0                                    # seconds

class RobotSignalAgent:
    """
    Gửi danh sách waypoint sang ROS2 Nav2 qua HTTP.
    Hoạt động ở chế độ simulation nếu robot không kết nối được.
    """

    def __init__(self, bridge_url: str = ROBOT_BRIDGE_URL):
        self.bridge_url = bridge_url

    def _to_ros2_payload(self, waypoints: List[WaypointPose]) -> dict:
        """
        Chuyển danh sách WaypointPose thành ROS2-compatible JSON payload.
        Format tương thích với action_msgs/NavigateToPose hoặc
        nav2_msgs/NavigateThrough Poses.
        """
        poses = []
        for wp in waypoints:
            poses.append({
                "header": {
                    "frame_id": wp.frame
                },
                "pose": {
                    "position": {"x": wp.x, "y": wp.y, "z": 0.0},
                    "orientation": {
                        # Convert theta (yaw) → quaternion (z, w only – flat floor)
                        "x": 0.0,
                        "y": 0.0,
                        "z": round(__import__("math").sin(wp.theta / 2), 6),
                        "w": round(__import__("math").cos(wp.theta / 2), 6),
                    }
                }
            })
        return {
            "type": "navigate_through_poses",
            "poses": poses,
            "behavior_tree": ""
        }

    async def send_waypoints(self, waypoints: List[WaypointPose]) -> bool:
        """
        Gửi danh sách waypoint đến ROS2 bridge.
        Returns True nếu thành công, False nếu offline (simulation mode).
        """
        if not waypoints:
            logger.warning("No waypoints to send.")
            return False

        payload = self._to_ros2_payload(waypoints)

        try:
            async with httpx.AsyncClient(timeout=ROBOT_TIMEOUT) as client:
                response = await client.post(self.bridge_url, json=payload)
                response.raise_for_status()
                logger.info(f"Robot signal sent: {len(waypoints)} waypoints → ROS2")
                return True
        except Exception as e:
            logger.warning(
                f"[ROBOT SIMULATE] Robot bridge offline ({e}). "
                f"Would navigate through: {[wp.node_id for wp in waypoints]}"
            )
            return False
