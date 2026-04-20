"""
Audio pipeline: per-session VAD state machine → speech buffering → transcription queue.

States:
  IDLE         – no speech detected
  COLLECTING   – accumulating speech frames
  COMMITTING   – silence threshold hit, submitting chunk for transcription
"""
import asyncio
import logging
import time
from enum import Enum, auto
from typing import AsyncIterator, List, Optional

import numpy as np

from core import transcriber as tr
from core.vad import WebRTCVAD, create_vad
from models.messages import SegmentMessage, TranscriptionMessage
from models.session_state import SessionState
from utils.audio import float32_bytes_to_array, float32_to_pcm_bytes, is_adts_frame, decode_adts_to_float32

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_SEGMENT_DURATION_S = 30.0
INTERIM_INTERVAL_S = 0.3


class PipelineState(Enum):
    IDLE = auto()
    COLLECTING = auto()


class AudioPipeline:
    """Per-session audio processing pipeline."""

    def __init__(self, session_state: SessionState, send_queue: asyncio.Queue, settings):
        self._state = session_state
        self._send_queue = send_queue
        self._settings = settings

        self._vad: WebRTCVAD = create_vad(
            backend=settings.vad_backend,
            aggressiveness=settings.vad_aggressiveness,
            frame_ms=settings.vad_frame_ms,
        )

        self._pipeline_state = PipelineState.IDLE
        self._speech_buffer: List[np.ndarray] = []  # float32 chunks
        self._silence_frames = 0
        self._speech_frames = 0
        self._byte_remainder = b""  # leftover bytes not yet forming a full frame

        # Silence threshold in VAD frames
        frames_per_second = 1000 / settings.vad_frame_ms
        self._silence_threshold_frames = int(settings.vad_silence_threshold_ms / settings.vad_frame_ms)
        self._min_speech_frames = int(settings.vad_min_speech_ms / settings.vad_frame_ms)

        self._chunk_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._interim_task: Optional[asyncio.Task] = None
        self._closed = False

        # Silence tracking for silence_timeout_sec
        self._silence_start_time: Optional[float] = None  # wall-clock time silence began

    def start(self) -> None:
        """Start background worker tasks."""
        self._silence_start_time = time.monotonic()  # start silence clock immediately
        self._worker_task = asyncio.create_task(self._transcription_worker(), name="transcription-worker")
        self._interim_task = asyncio.create_task(self._interim_sender(), name="interim-sender")

    @property
    def silence_duration_sec(self) -> float:
        """Seconds of continuous silence. 0.0 if speech is active or no audio yet."""
        if self._silence_start_time is None:
            return 0.0
        return max(0.0, time.monotonic() - self._silence_start_time)

    async def feed(self, raw_data: bytes) -> None:
        """Feed raw audio bytes (float32 PCM or AAC-ADTS) into the pipeline."""
        if self._closed:
            return

        # Decode audio to float32 numpy array
        if is_adts_frame(raw_data):
            audio_f32 = decode_adts_to_float32(raw_data, SAMPLE_RATE)
        else:
            audio_f32 = float32_bytes_to_array(raw_data)

        if len(audio_f32) == 0:
            return

        self._state.total_audio_duration += len(audio_f32) / SAMPLE_RATE

        # Convert to int16 bytes for VAD, combining with any leftover
        pcm_bytes = self._byte_remainder + float32_to_pcm_bytes(audio_f32)
        frame_bytes = self._vad.frame_bytes
        offset = 0

        while offset + frame_bytes <= len(pcm_bytes):
            frame = pcm_bytes[offset: offset + frame_bytes]
            offset += frame_bytes
            is_speech = self._vad.process_frame(frame)

            # Convert this frame back to float32 for buffering
            frame_f32 = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0

            if is_speech:
                self._silence_frames = 0
                self._speech_frames += 1
                self._speech_buffer.append(frame_f32)
                self._pipeline_state = PipelineState.COLLECTING
                self._silence_start_time = None  # speech resumed, reset silence timer
            else:
                if self._pipeline_state == PipelineState.COLLECTING:
                    self._silence_frames += 1
                    # Keep buffering during short silences (for smooth speech)
                    self._speech_buffer.append(frame_f32)

                    if self._silence_frames >= self._silence_threshold_frames:
                        await self._commit_chunk()
                        # Chunk committed: speech→silence transition, start silence timer
                        if self._silence_start_time is None:
                            self._silence_start_time = time.monotonic()

            # Force commit if segment is too long
            buffered_duration = sum(len(f) for f in self._speech_buffer) / SAMPLE_RATE
            if buffered_duration >= MAX_SEGMENT_DURATION_S:
                await self._commit_chunk()

        self._byte_remainder = pcm_bytes[offset:]

    async def finalize(self) -> None:
        """Flush any remaining buffered audio after END_OF_AUDIO."""
        if self._speech_buffer and self._speech_frames >= self._min_speech_frames:
            await self._commit_chunk()
        # Signal worker to drain
        await self._chunk_queue.put(None)
        if self._worker_task:
            await self._worker_task

    async def close(self) -> None:
        """Cancel background tasks and clean up."""
        self._closed = True
        if self._interim_task and not self._interim_task.done():
            self._interim_task.cancel()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()

    async def _commit_chunk(self) -> None:
        """Commit current buffer as a speech chunk for transcription."""
        if not self._speech_buffer or self._speech_frames < self._min_speech_frames:
            self._speech_buffer.clear()
            self._speech_frames = 0
            self._silence_frames = 0
            self._pipeline_state = PipelineState.IDLE
            return

        chunk = np.concatenate(self._speech_buffer)
        self._speech_buffer.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._pipeline_state = PipelineState.IDLE

        await self._chunk_queue.put(chunk)

    async def _transcription_worker(self) -> None:
        """Drain the chunk queue and run transcription."""
        while True:
            try:
                chunk = await self._chunk_queue.get()
                if chunk is None:
                    break  # finalize signal

                result = await tr.transcribe_chunk(
                    chunk,
                    language=self._state.language,
                    task=self._state.task,
                    beam_size=self._settings.beam_size,
                    best_of=self._settings.best_of,
                    temperature=self._settings.temperature,
                )

                # Update detected language from first segment
                if result.language and not self._state.language:
                    pass  # keep auto-detect per chunk

                self._state.segments.extend(result.segments)
                self._state.buffer_transcription = ""
                self._state.total_audio_duration += result.duration

                lines = [
                    SegmentMessage(
                        text=seg.text,
                        start=seg.start,
                        end=seg.end,
                        detected_language=seg.language,
                        no_speech_prob=seg.no_speech_prob,
                    )
                    for seg in result.segments
                ]

                if lines:
                    msg = TranscriptionMessage(
                        uid=self._state.uid,
                        status="active_transcription",
                        is_final=True,
                        language=result.language,
                        lines=lines,
                        buffer_transcription="",
                    )
                    await self._send_queue.put(msg.model_dump_json())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Transcription worker error: {e}")

    async def _interim_sender(self) -> None:
        """Periodically send interim transcription results while collecting."""
        try:
            while True:
                await asyncio.sleep(INTERIM_INTERVAL_S)
                if self._closed:
                    break

                if self._pipeline_state == PipelineState.COLLECTING and self._speech_buffer:
                    # Take a snapshot of current buffer
                    snapshot = np.concatenate(self._speech_buffer)
                    if len(snapshot) < SAMPLE_RATE * 0.2:  # skip very short < 0.2s
                        continue

                    try:
                        result = await tr.transcribe_chunk(
                            snapshot,
                            language=self._state.language,
                            task=self._state.task,
                            beam_size=1,   # fast beam for interim
                            best_of=1,
                            temperature=0.0,
                        )
                        interim_text = " ".join(seg.text for seg in result.segments)
                        if interim_text:
                            self._state.buffer_transcription = interim_text
                            msg = TranscriptionMessage(
                                uid=self._state.uid,
                                status="active_transcription",
                                is_final=False,
                                language=result.language,
                                lines=[
                                    SegmentMessage(
                                        text=s.text,
                                        start=s.start,
                                        end=s.end,
                                        detected_language=s.language,
                                        no_speech_prob=s.no_speech_prob,
                                    )
                                    for s in self._state.segments
                                ],
                                buffer_transcription=interim_text,
                            )
                            await self._send_queue.put(msg.model_dump_json())
                    except Exception as e:
                        logger.debug(f"Interim transcription error (non-fatal): {e}")

        except asyncio.CancelledError:
            pass
