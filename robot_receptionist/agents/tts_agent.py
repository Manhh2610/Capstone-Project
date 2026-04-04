"""
TTS Agent – Piper Vietnamese TTS bridge.

Yêu cầu: Piper phải được cài đặt và có sẵn model tiếng Việt.
  Cài đặt: https://github.com/rhasspy/piper
  Model: vi_VN-vivos-x_low.onnx (hoặc tương đương)

Chạy dưới dạng subprocess hoặc qua HTTP server của Piper.
"""

import asyncio
import hashlib
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Cấu hình ─────────────────────────────────────────────────────────────────

PIPER_BINARY = os.environ.get("PIPER_BINARY", "piper")
PIPER_MODEL = os.environ.get("PIPER_MODEL", "data/piper_models/vi_VN-vivos-x_low.onnx")
CACHE_DIR = Path("data/tts_cache")
OUTPUT_DIR = Path("/tmp/tts_output")

# Câu pre-render thường dùng – WAV sẽ được tạo lúc khởi động
PRERENDER_PHRASES = {
    "greeting": "Xin chào! Tôi là robot lễ tân. Tôi có thể giúp gì cho bạn?",
    "welcome": "Chào mừng bạn đến với tòa nhà. Tôi sẵn sàng hỗ trợ bạn.",
    "wait": "Vui lòng chờ một chút, tôi đang xử lý yêu cầu của bạn.",
    "understood": "Tôi đã hiểu yêu cầu của bạn.",
    "error": "Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại.",
    "goodbye": "Cảm ơn bạn đã sử dụng dịch vụ. Chúc bạn một ngày tốt lành!",
    "navigate_start": "Được rồi, tôi sẽ dẫn đường cho bạn ngay bây giờ.",
}

class TTSAgent:
    """
    Piper TTS bridge với:
    - Cache WAV theo SHA256 hash của text để tránh render lại.
    - Pre-render các câu thường dùng lúc khởi động.
    - Streaming sentence-level: phát từng câu khi LLM chưa xong.
    """

    def __init__(
        self,
        piper_binary: str = PIPER_BINARY,
        model_path: str = PIPER_MODEL,
        cache_dir: Path = CACHE_DIR,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.piper_binary = piper_binary
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.output_dir = output_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._piper_available = self._check_piper()

    def _check_piper(self) -> bool:
        """Check if piper binary exists and is executable."""
        try:
            result = subprocess.run(
                [self.piper_binary, "--version"],
                capture_output=True, timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning(
                f"Piper TTS not found at '{self.piper_binary}'. "
                "TTS will operate in simulation mode (no audio output)."
            )
            return False

    def _cache_path(self, text: str) -> Path:
        """Generate a deterministic WAV filename based on text content."""
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{h}.wav"

    async def synthesize(self, text: str) -> Path | None:
        """
        Synthesize text to WAV. Returns path to .wav file.
        Uses cache if available. Falls back to simulation if Piper unavailable.
        """
        wav_path = self._cache_path(text)

        # Return cached file immediately
        if wav_path.exists():
            logger.debug(f"TTS cache hit: {wav_path.name}")
            return wav_path

        if not self._piper_available:
            logger.info(f"[TTS SIMULATE] Would speak: \"{text[:60]}...\"")
            return None

        # Run Piper in subprocess
        try:
            proc = await asyncio.create_subprocess_exec(
                self.piper_binary,
                "--model", self.model_path,
                "--output_file", str(wav_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate(input=text.encode("utf-8"))

            if wav_path.exists():
                logger.info(f"TTS synthesized: {wav_path.name} ({len(text)} chars)")
                return wav_path
            else:
                logger.error("Piper ran but no WAV file was created.")
                return None

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None

    async def speak(self, text: str):
        """
        Synthesize and play audio via aplay (ALSA).
        Non-blocking: runs in background task.
        """
        wav_path = await self.synthesize(text)
        if wav_path and wav_path.exists():
            await asyncio.create_subprocess_exec("aplay", str(wav_path))

    async def prerender_common_phrases(self):
        """
        Pre-render all common phrases to WAV at startup.
        Reduces first-response latency significantly.
        """
        logger.info("Pre-rendering common TTS phrases...")
        tasks = [self.synthesize(text) for text in PRERENDER_PHRASES.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        rendered = sum(1 for r in results if r and not isinstance(r, Exception))
        logger.info(f"Pre-rendered {rendered}/{len(PRERENDER_PHRASES)} phrases.")

    def get_prerendered(self, key: str) -> Path | None:
        """Get a pre-rendered WAV by phrase key (e.g. 'greeting')."""
        text = PRERENDER_PHRASES.get(key)
        if not text:
            return None
        path = self._cache_path(text)
        return path if path.exists() else None
