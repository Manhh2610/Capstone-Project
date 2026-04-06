"""
Speech-to-Text module using faster-whisper.
"""
import os
import tempfile
import numpy as np
from faster_whisper import WhisperModel
import scipy.io.wavfile as wav

# Initialize model globally to keep it loaded
MODEL_SIZE = "medium"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

print(f"Loading faster-whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

def transcribe_file(audio_path: str) -> str:
    """Transcribe an audio file to text."""
    try:
        segments, info = model.transcribe(audio_path, language="vi", beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def record_and_transcribe(duration: float = 5.0) -> str:
    """Record audio from microphone and transcribe it."""
    try:
        import sounddevice as sd
    except OSError:
        print("sounddevice (PortAudio) not available. Cannot record from mic.")
        return ""
        
    sample_rate = 16000
    try:
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        print("Recording finished.")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_filename = tmp_file.name
            wav.write(temp_filename, sample_rate, audio)
            
        # Transcribe
        text = transcribe_file(temp_filename)
        
        # Clean up
        os.remove(temp_filename)
        return text
    except Exception as e:
        print(f"Microphone or recording error: {e}")
        return ""
