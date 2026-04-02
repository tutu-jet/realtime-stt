from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from core.transcriber import SegmentResult


@dataclass
class SessionState:
    uid: str
    language: Optional[str]
    task: str
    connected_at: datetime = field(default_factory=datetime.utcnow)
    total_audio_duration: float = 0.0
    segments: List[SegmentResult] = field(default_factory=list)
    buffer_transcription: str = ""
    status: str = "active_transcription"
