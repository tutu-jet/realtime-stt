from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model
    # 可选: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en,
    #       large-v1, large-v2, large-v3, large-v3-turbo, distil-large-v2, distil-large-v3
    model_size: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "float16"
    model_cache_dir: str = "/models"

    # VAD
    vad_backend: str = "webrtcvad"
    vad_aggressiveness: int = 2
    vad_frame_ms: int = 30
    vad_silence_threshold_ms: int = 300
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
    max_clients: int = 10              # 最大并发连接数 + 线程池大小
    session_timeout_sec: float = 60    # 单会话超时 (秒), 0=不限
    silence_timeout_sec: float = 5     # 持续静音断开 (秒), 0=不断开
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
