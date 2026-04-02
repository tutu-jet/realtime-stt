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

# Model cache mount point
RUN mkdir -p /models

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "9090", \
     "--workers", "1"]
