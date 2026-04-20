FROM python:3.11-slim

# System dependencies: ffmpeg for audio decode, libgomp for faster-whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model during build (baked into image at /models-baked)
ARG MODEL_SIZE=tiny
ARG HF_ENDPOINT=https://hf-mirror.com
ENV HF_ENDPOINT=${HF_ENDPOINT}
RUN mkdir -p /models-baked && \
    python -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='int8', download_root='/models-baked')"

# Runtime mount point — entrypoint will copy baked model if /models is empty
RUN mkdir -p /models

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:9090/v1/health || exit 1

CMD ["sh", "-c", "\
    if [ -z \"$(ls -A /models 2>/dev/null)\" ]; then \
        echo 'Copying baked model to /models...'; \
        cp -a /models-baked/. /models/; \
    fi && \
    exec python -m uvicorn main:app --host 0.0.0.0 --port 9090 --workers 1"]
