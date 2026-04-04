import httpx
import logging
from typing import Dict, Any

from agents.intent_detector import IntentDetector
from agents.knowledge_agent import KnowledgeAgent

# BIM Service defaults
BIM_BASE_URL = "http://localhost:8001"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

logger = logging.getLogger(__name__)

class ReceptionistAgent:
    """
    Main Orchestrator Agent for the Receptionist Robot.
    Updated with RAG (KnowledgeBase) support.
    """
    def __init__(self, bim_url: str = BIM_BASE_URL, ollama_url: str = OLLAMA_URL):
        self.bim_url = bim_url
        self.ollama_url = ollama_url
        self.detector = IntentDetector(ollama_url=ollama_url)
        self.knowledge = KnowledgeAgent()

    async def process_user_input(self, text: str) -> Dict[str, Any]:
        """
        Process the user input text and return a response object.
        """
        # 1. Detect Intent
        result = await self.detector.detect_intent(text)
        intent = result.get("intent", "CHAT")
        target = result.get("target")

        # 2. Execute Action
        if intent == "NAVIGATE":
            return await self._handle_navigation(target)
        elif intent == "RESOLVE":
            return await self._handle_resolution(target)
        else:
            # Default to CHAT with RAG context
            return await self._handle_chat(text)

    async def _handle_navigation(self, target: str) -> Dict[str, Any]:
        """Call BIM resolve and navigate/speak to guide the user."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res_resolve = await client.post(f"{self.bim_url}/resolve", json={"query": target or "entrance"})
                res_resolve.raise_for_status()
                resolve_data = res_resolve.json()
                room_id = resolve_data.get("room_id")
                room_name = resolve_data.get("name")

                res_navigate = await client.post(f"{self.bim_url}/navigate/speak", json={
                    "from_id": "entrance",
                    "to_id": room_id,
                    "preference": "shortest"
                })
                res_navigate.raise_for_status()
                nav_data = res_navigate.json()

                return {
                    "intent": "NAVIGATE",
                    "speak": nav_data.get("speak", f"Đi đến {room_name}."),
                    "data": nav_data
                }
        except Exception as e:
            logger.error(f"Error handling navigation for {target}: {e}")
            return {
                "intent": "ERROR",
                "speak": "Dạ, hiện tại tôi gặp chút khó khăn khi tìm đường. Bạn vui lòng thử lại sau.",
                "error": str(e)
            }

    async def _handle_resolution(self, target: str) -> Dict[str, Any]:
        """Resolve room and return basic information."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(f"{self.bim_url}/resolve", json={"query": target})
                res.raise_for_status()
                data = res.json()
                return {
                    "intent": "RESOLVE",
                    "speak": f"Vâng, {data.get('name')} ở vị trí tương ứng trong hệ thống. Tôi có thể dẫn bạn đến đó.",
                    "data": data
                }
        except Exception as e:
            return {"intent": "ERROR", "speak": "Tôi không tìm thấy phòng đó.", "error": str(e)}

    async def _handle_chat(self, text: str) -> Dict[str, Any]:
        """General chat using LLM powered by RAG context."""
        # --- RAG Part: Get context from knowledge agent ---
        context = self.knowledge.query_knowledge(text)
        
        system_prompt = (
            "Bạn là một robot lễ tân thân thiện. Hãy trả lời câu hỏi của khách hàng ngắn gọn, lịch sự bằng tiếng Việt."
        )
        
        if context:
            system_prompt += f"\nSử dụng thông tin sau đây để trả lời:\n{context}"
        
        payload = {
            "model": MODEL_NAME,
            "prompt": f"{system_prompt}\n\nKhách hàng: '{text}'\nRobot:",
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(self.ollama_url, json=payload)
                response.raise_for_status()
                result = response.json()
                return {
                    "intent": "CHAT",
                    "speak": result.get("response", "Dạ, tôi nghe ạ.").strip(),
                    "data": {"context_used": bool(context)}
                }
        except Exception as e:
            return {"intent": "ERROR", "speak": "Dạ, tôi đang bận một chút ạ.", "error": str(e)}
