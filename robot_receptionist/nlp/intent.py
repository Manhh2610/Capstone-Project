"""
Simple Regex and Keyword-based intent classification and entity extraction.
"""
import re

NAV_KEYWORDS = ["đến", "tìm", "đường", "phòng", "tầng", "chỗ", "gặp", "đi", "muốn", "cần", "ở đâu", "hướng dẫn"]
FAQ_KEYWORDS = ["wifi", "giờ", "mấy giờ", "khi nào", "bao nhiêu", "làm việc", "nghỉ", "quy định", "mật khẩu"]
GREETING_KEYWORDS = ["xin chào", "hello", "chào", "hi", "hế lô"]

def classify_intent(text: str) -> str:
    """Classify user text into navigation, faq, greeting, or unknown based on keywords."""
    text_lower = text.lower()
    
    # Check greeting first for exact/start matches
    for kw in GREETING_KEYWORDS:
        if text_lower.startswith(kw) or text_lower == kw:
            return "greeting"
            
    # Check FAQ
    for kw in FAQ_KEYWORDS:
        if kw in text_lower:
            return "faq"
            
    # Check Navigation
    for kw in NAV_KEYWORDS:
        if kw in text_lower:
            return "navigation"
            
    return "unknown"

def extract_room_query(text: str) -> str:
    """Extract the room or destination from a navigation query."""
    text_lower = text.lower()
    
    # Patterns to strip out common prompt prefixes
    patterns = [
        r"tôi muốn đến\s+(.*)",
        r"cho tôi đến\s+(.*)",
        r"tìm đường đến\s+(.*)",
        r"chỉ đường đến\s+(.*)",
        r"muốn đi đến\s+(.*)",
        r"làm sao để đến\s+(.*)",
        r"hướng dẫn đi\s+(.*)",
        r"hướng dẫn đến\s+(.*)",
        r"phòng\s+(.*)\s+ở đâu",
        r"(.*)\s+ở đâu",
        r"đến\s+(.*)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).strip()
            
    return text.strip()
