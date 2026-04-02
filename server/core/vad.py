from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class VADEngine(Protocol):
    def process_frame(self, pcm_int16_frame: bytes) -> bool:
        """Return True if the frame contains speech."""
        ...

    def reset(self) -> None:
        """Reset internal state."""
        ...


class WebRTCVAD:
    """webrtcvad-based VAD engine operating on 10/20/30ms frames at 16kHz."""

    SAMPLE_RATE = 16000

    def __init__(self, aggressiveness: int = 2, frame_ms: int = 30):
        import webrtcvad
        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_ms = frame_ms
        self._frame_size = int(self.SAMPLE_RATE * frame_ms / 1000)  # samples per frame

    @property
    def frame_size(self) -> int:
        """Number of PCM int16 samples per frame."""
        return self._frame_size

    @property
    def frame_bytes(self) -> int:
        """Number of bytes per frame (int16 = 2 bytes/sample)."""
        return self._frame_size * 2

    def process_frame(self, pcm_int16_frame: bytes) -> bool:
        """Return True if the 30ms frame contains speech."""
        try:
            return self._vad.is_speech(pcm_int16_frame, self.SAMPLE_RATE)
        except Exception:
            return False

    def reset(self) -> None:
        pass  # webrtcvad is stateless per-frame


def create_vad(backend: str = "webrtcvad", aggressiveness: int = 2, frame_ms: int = 30) -> WebRTCVAD:
    """Factory function to create a VAD engine."""
    if backend == "webrtcvad":
        return WebRTCVAD(aggressiveness=aggressiveness, frame_ms=frame_ms)
    raise ValueError(f"Unknown VAD backend: {backend}")
