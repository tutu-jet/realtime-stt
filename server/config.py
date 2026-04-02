from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model
    model_size: str = "medium"
    device: str = "auto"
    compute_type: str = "int8"
    model_cache_dir: str = "/models"

    # VAD
    vad_backend: str = "webrtcvad"
    vad_aggressiveness: int = 2
    vad_frame_ms: int = 30
    vad_silence_threshold_ms: int = 600
    vad_min_speech_ms: int = 250

    # Transcription
    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0

    # Server
    host: str = "0.0.0.0"
    port: int = 9090
    max_clients: int = 10
    max_connection_time: int = 600
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
