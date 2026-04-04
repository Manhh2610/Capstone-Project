"""
STT Agent – Whisper.cpp bridge.

Giao tiếp với whisper.cpp HTTP server (llama-server compatible mode)
  Chạy: ./server -m ggml-medium.bin --host 0.0.0.0 --port 7000

Fallback: subprocess với whisper.cpp binary để xử lý file WAV.
"""

import asyncio
import httpx
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

WHISPER_SERVER_URL = os.environ.get("WHISPER_SERVER_URL", "http://localhost:7000")
WHISPER_BINARY     = os.environ.get("WHISPER_BINARY", "whisper-cli")
WHISPER_MODEL      = os.environ.get("WHISPER_MODEL", "data/whisper_models/ggml-medium.bin")


class STTAgent:
    """
    Whisper.cpp STT bridge với hai chế độ:
      1. HTTP Server mode  – gửi audio bytes → nhận JSON transcript (~200ms)
      2. Subprocess mode   – chạy whisper-cli với file WAV
    """

    def __init__(
        self,
        server_url: str = WHISPER_SERVER_URL,
        binary: str = WHISPER_BINARY,
        model: str = WHISPER_MODEL,
    ):
        self.server_url = server_url
        self.binary     = binary
        self.model      = model

    async def transcribe_audio(self, audio_bytes: bytes, language: str = "vi") -> str:
        """
        Transcribe audio bytes → Vietnamese text.
        Thử gọi Whisper HTTP server trước; nếu thất bại, dùng subprocess.
        """
        # Try HTTP server mode first (fast, ~200ms)
        text = await self._transcribe_via_server(audio_bytes, language)
        if text:
            return text

        # Fallback to subprocess
        return await self._transcribe_via_subprocess(audio_bytes, language)

    async def _transcribe_via_server(self, audio_bytes: bytes, language: str) -> str:
        """POST audio to Whisper HTTP server."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.server_url}/inference",
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                    data={"language": language, "response_format": "json"},
                )
                response.raise_for_status()
                result = response.json()
                text = result.get("text", "").strip()
                if text:
                    logger.info(f"STT (server): \"{text[:80]}\"")
                return text
        except Exception as e:
            logger.debug(f"STT server unavailable: {e}")
            return ""

    async def _transcribe_via_subprocess(self, audio_bytes: bytes, language: str) -> str:
        """Write WAV to temp file and run whisper-cli."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            proc = await asyncio.create_subprocess_exec(
                self.binary,
                "--model",    self.model,
                "--language", language,
                "--output-txt",
                "--no-timestamps",
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()

            # whisper-cli writes to <input>.txt
            txt_path = tmp_path + ".txt"
            if Path(txt_path).exists():
                with open(txt_path) as f:
                    text = f.read().strip()
                Path(txt_path).unlink(missing_ok=True)
                logger.info(f"STT (subprocess): \"{text[:80]}\"")
                return text

            # Try stdout fallback
            if stdout:
                return stdout.decode("utf-8").strip()

            return ""

        except FileNotFoundError:
            logger.warning(f"Whisper binary '{self.binary}' not found. STT unavailable.")
            return ""
        except Exception as e:
            logger.error(f"STT subprocess error: {e}")
            return ""
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def transcribe_file(self, wav_path: str, language: str = "vi") -> str:
        """Convenience: transcribe a WAV file directly."""
        with open(wav_path, "rb") as f:
            return await self.transcribe_audio(f.read(), language)
