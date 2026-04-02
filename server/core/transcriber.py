import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_executor: Optional[ThreadPoolExecutor] = None


@dataclass
class SegmentResult:
    text: str
    start: float
    end: float
    language: str
    no_speech_prob: float


@dataclass
class TranscriptionResult:
    segments: List[SegmentResult]
    language: str
    duration: float


def load_model(model_size: str, device: str, compute_type: str, model_cache_dir: str, num_workers: int = 4) -> None:
    """Load faster-whisper model at application startup. Called once."""
    global _model, _executor
    from faster_whisper import WhisperModel

    logger.info(f"Loading faster-whisper model: {model_size} on {device} ({compute_type})")
    _model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=model_cache_dir,
    )
    _executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="whisper")
    logger.info("Model loaded successfully.")


def unload_model() -> None:
    global _model, _executor
    if _executor:
        _executor.shutdown(wait=False)
        _executor = None
    _model = None
    logger.info("Model unloaded.")


def _sync_transcribe(
    audio: np.ndarray,
    language: Optional[str],
    task: str,
    beam_size: int,
    best_of: int,
    temperature: float,
) -> TranscriptionResult:
    """Blocking transcription — runs in executor thread."""
    segments_gen, info = _model.transcribe(
        audio,
        language=language,
        task=task,
        beam_size=beam_size,
        best_of=best_of,
        temperature=temperature,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    segments = [
        SegmentResult(
            text=seg.text.strip(),
            start=seg.start,
            end=seg.end,
            language=info.language,
            no_speech_prob=seg.no_speech_prob,
        )
        for seg in segments_gen
        if seg.text.strip()
    ]
    return TranscriptionResult(
        segments=segments,
        language=info.language,
        duration=info.duration,
    )


async def transcribe_chunk(
    audio: np.ndarray,
    language: Optional[str] = None,
    task: str = "transcribe",
    beam_size: int = 5,
    best_of: int = 5,
    temperature: float = 0.0,
) -> TranscriptionResult:
    """Async wrapper — offloads blocking inference to thread pool."""
    if _model is None or _executor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        _sync_transcribe,
        audio,
        language,
        task,
        beam_size,
        best_of,
        temperature,
    )
    return result


def is_ready() -> bool:
    return _model is not None
