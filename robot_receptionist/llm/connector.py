"""
Connect to local Ollama API to format text and answer FAQs using Qwen2.5:3b.
"""
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"
TIMEOUT_SEC = 10.0

async def _call_ollama(prompt: str, system_prompt: str) -> str:
    """Helper to make async call to Ollama."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 80
        }
    }
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
        try:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Xin lỗi, hiện tại tôi không thể xử lý yêu cầu này."

async def format_navigation(steps: list[str], destination: str) -> str:
    """Format a list of step instructions into a natural Vietnamese sentence."""
    system_prompt = (
        "Bạn là trí tuệ nhân tạo chỉ làm nhiệm vụ nối các bước đi thành một câu tiếng Việt ngắn gọn, liền mạch. "
        "Tuyệt đối không thêm thông tin, không giải thích, không kể chuyện."
    )
    
    steps_text = " -> ".join(steps)
    prompt = f"Cách đi đến {destination}:\nCác bước: {steps_text}\nHãy chuyển thành 1 câu hướng dẫn ngắn gọn."
    
    return await _call_ollama(prompt, system_prompt)

async def answer_faq(question: str, context: str = "") -> str:
    """Answer a FAQ question briefly in Vietnamese."""
    system_prompt = (
        "Bạn là robot lễ tân thân thiện. Hãy trả lời câu hỏi trực tiếp, ngắn gọn bằng tiếng Việt. "
        "Không dùng ngôn ngữ dài dòng, chỉ trả lời trọng tâm."
    )
    
    prompt = f"Câu hỏi: {question}"
    if context:
        prompt += f"\nThông tin tham khảo: {context}"
        
    return await _call_ollama(prompt, system_prompt)
