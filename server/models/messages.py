from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClientConfigMessage(BaseModel):
    uid: str
    language: Optional[str] = None
    task: str = "transcribe"
    model: str = "medium"
    use_vad: bool = True
    max_clients: int = 10
    max_connection_time: int = 600


class ServerConfigMessage(BaseModel):
    type: str = "config"
    uid: str
    message: str = "WAIT"
    backend: str = "faster-whisper"
    model: str
    language: Optional[str]
    use_audio_worklet: bool = True


class SegmentMessage(BaseModel):
    text: str
    start: float
    end: float
    detected_language: str
    no_speech_prob: float


class TranscriptionMessage(BaseModel):
    uid: str
    status: str = "active_transcription"
    is_final: bool = False
    language: Optional[str] = None
    lines: List[SegmentMessage] = Field(default_factory=list)
    buffer_transcription: str = ""


class ReadyToStopMessage(BaseModel):
    uid: str
    type: str = "ready_to_stop"
    status: str = "completed"
    lines: List[SegmentMessage] = Field(default_factory=list)
    buffer_transcription: str = ""


class ErrorMessage(BaseModel):
    uid: str
    type: str = "error"
    code: str
    message: str
