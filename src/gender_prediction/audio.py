import base64
from io import BytesIO

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16000


def decode_audio(b64_audio: str) -> np.ndarray:
    """Base64 audio → float32 mono waveform at 16kHz."""
    raw = base64.b64decode(b64_audio)
    data, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=True)
    # mix down to mono
    mono = data.mean(axis=1)
    if sr != TARGET_SR:
        mono = soxr.resample(mono, sr, TARGET_SR, quality="HQ")
    return mono.astype(np.float32)
