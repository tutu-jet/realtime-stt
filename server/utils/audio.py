import numpy as np
from scipy.signal import resample_poly
from math import gcd


def pcm_bytes_to_float32(data: bytes) -> np.ndarray:
    """Convert raw PCM int16 bytes to float32 numpy array in [-1.0, 1.0]."""
    audio = np.frombuffer(data, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def float32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array to PCM int16 bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio to target sample rate using polyphase filtering."""
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def float32_bytes_to_array(data: bytes) -> np.ndarray:
    """Convert raw float32 little-endian bytes to numpy array."""
    return np.frombuffer(data, dtype=np.float32).copy()


def is_adts_frame(data: bytes) -> bool:
    """Detect AAC-ADTS frame by sync word."""
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF0) == 0xF0


def decode_adts_to_float32(data: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode AAC-ADTS bytes to float32 PCM at target_sr using PyAV."""
    import av
    import io

    container = av.open(io.BytesIO(data), format="adts")
    samples = []
    orig_sr = target_sr

    for frame in container.decode(audio=0):
        orig_sr = frame.sample_rate
        arr = frame.to_ndarray()
        # Mono mix if stereo
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        samples.append(arr.astype(np.float32) / 32768.0)

    container.close()

    if not samples:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(samples)
    return resample_audio(audio, orig_sr, target_sr)
