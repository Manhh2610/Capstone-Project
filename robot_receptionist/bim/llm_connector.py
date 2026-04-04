import httpx
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

async def format_navigation(steps: list[dict], destination: str) -> str:
    """
    Format navigation steps into natural Vietnamese speech using Ollama.
    """
    if not steps:
        return f"Bạn đã đến {destination}."

    # Build the prompt
    system_prompt = (
        "Bạn là một robot lễ tân thân thiện. Hãy tóm tắt lộ trình di chuyển "
        "dưới đây thành một câu hướng dẫn tự nhiên bằng tiếng Việt cho khách hàng."
    )
    
    # Extract step instructions
    steps_list = [step.get("instruction", "") for step in steps if step.get("instruction")]
    steps_text = "\n".join(f"- {s}" for s in steps_list)
    
    prompt = (
        f"Lộ trình đi đến {destination}:\n"
        f"{steps_text}\n\n"
        "Hãy chuyển lộ trình này thành một câu nói duy nhất, ngắn gọn, lịch sự."
    )
    
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{system_prompt}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 80
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
    except Exception as e:
        # Fallback to simple join if LLM fails
        return f"Đi đến {destination}: " + " rồi ".join(steps_list)
