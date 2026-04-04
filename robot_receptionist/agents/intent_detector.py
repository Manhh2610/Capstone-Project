import httpx
import json
import logging

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentDetector:
    """
    Agent that uses LLM to classify user intent for a receptionist robot.
    """
    def __init__(self, ollama_url: str = OLLAMA_URL, model_name: str = MODEL_NAME):
        self.ollama_url = ollama_url
        self.model_name = model_name

    async def detect_intent(self, text: str) -> dict:
        """
        Detect the intent from user text using LLM.
        Returns: {"intent": "NAVIGATE" | "CHAT" | "RESOLVE", "target": "..." | None}
        """
        system_prompt = (
            "Bạn là bộ não điều phối cho robot lễ tân. Hãy phân loại văn bản của khách hàng thành một trong các 'intent' sau:\n"
            "1. NAVIGATE: Khách muốn robot dẫn đi đâu đó.\n"
            "2. RESOLVE: Khách hỏi vị trí hoặc thông tin một phòng cụ thể.\n"
            "3. CHAT: Các câu chào hỏi hoặc trò chuyện xã giao.\n"
            "\n"
            "Kết quả trả về DUY NHẤT một JSON format như sau: {\"intent\": \"...\", \"target\": \"...\"}\n"
            "Nếu intent là NAVIGATE hoặc RESOLVE, hãy trích xuất tên phòng vào 'target' (ví dụ: 'phòng giám đốc').\n"
            "Nếu không chắc chắn, hãy dùng intent 'CHAT'."
        )

        payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\nClient text: \"{text}\"",
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1
            }
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(self.ollama_url, json=payload)
                response.raise_for_status()
                result = response.json()
                content = result.get("response", "{}")
                return json.loads(content)
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            # Fallback for common patterns if LLM is unavailable
            if "dẫn" in text.lower() or "đi tới" in text.lower():
                return {"intent": "NAVIGATE", "target": text.split("tới")[-1].strip()}
            return {"intent": "CHAT", "target": None}
