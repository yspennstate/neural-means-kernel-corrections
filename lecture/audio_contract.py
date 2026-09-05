"""Bind actual narration audio to its text and synthesis settings."""
import hashlib
import json
from pathlib import Path
import wave

VOICE = "en-US-AndrewMultilingualNeural"
RATE = "+10%"


def normalized(text):
    return " ".join(text.split())


def identity(text, voice=VOICE, rate=RATE):
    return {"text_sha256": hashlib.sha256(normalized(text).encode("utf-8")).hexdigest(),
            "voice": voice, "rate": rate, "sample_rate": 24000, "channels": 1,
            "sample_width_bytes": 2, "engine": "edge-tts"}


def audio_info(path):
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (24000, 1, 2):
            raise ValueError("Narration must be 24 kHz mono signed 16-bit PCM")
        frames = wav.getnframes()
        if frames <= 0:
            raise ValueError("Empty narration")
    return {"wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frames": frames, "seconds": frames / 24000}


def validate(path, text, voice=VOICE, rate=RATE):
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if receipt["identity"] != identity(text, voice, rate):
        raise ValueError(f"Stale text, voice or settings: {path.name}")
    observed = audio_info(path)
    if receipt["audio"] != observed:
        raise ValueError(f"Audio changed after its receipt: {path.name}")
    return observed["seconds"]
