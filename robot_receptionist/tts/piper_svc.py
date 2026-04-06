"""
Text-to-Speech module using piper-tts.
"""
import os
import tempfile
import subprocess
import hashlib

CACHE_DIR = "data/tts_cache"
VOICE_MODEL = "models/vi_VN-vivos-medium.onnx"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# Pre-rendered cache dict mapping text to file path
# We populate this during runtime as sentences are rendered
_PRE_RENDER_CACHE = {}

def _get_cache_path(text: str) -> str:
    """Get a consistent file path for a given text."""
    hs = hashlib.md5(text.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{hs}.wav")

def _check_model_exists():
    """Check if piper model exists, otherwise notify."""
    if not os.path.exists(VOICE_MODEL):
        print(f"WARNING: Piper voice model not found at {VOICE_MODEL}")
        print("Please download it. Fallback (gtts) might be needed if you integrate internet-based fallback.")
        return False
    return True

def save_audio(text: str, output_path: str) -> None:
    """Save text to a WAV file using Piper."""
    if not _check_model_exists():
        print("Cannot save audio: Model missing.")
        return

    # piper expects input on stdin and output path with --output_file
    command = [
        "piper",
        "--model", VOICE_MODEL,
        "--output_file", output_path
    ]
    
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        process.communicate(input=text.encode('utf-8'))
        
        # Add to cache
        _PRE_RENDER_CACHE[text] = output_path
    except Exception as e:
        print(f"Error running piper: {e}")

def speak(text: str) -> None:
    """Generate audio and play it through speaker."""
    if not text:
        return
        
    print(f"Robot says: {text}")
    
    # Check cache
    if text in _PRE_RENDER_CACHE and os.path.exists(_PRE_RENDER_CACHE[text]):
        wav_path = _PRE_RENDER_CACHE[text]
    else:
        wav_path = _get_cache_path(text)
        if not os.path.exists(wav_path):
            save_audio(text, wav_path)
    
    # Play audio using aplay or similar (assuming Linux)
    if os.path.exists(wav_path):
        try:
            subprocess.run(["aplay", "-q", wav_path], check=False)
        except Exception:
            try:
                # Install pygame or use sounddevice if aplay fails
                import sounddevice as sd
                import soundfile as sf
                data, fs = sf.read(wav_path)
                sd.play(data, fs)
                sd.wait()
            except Exception as e:
                print(f"Could not play audio: {e}")
