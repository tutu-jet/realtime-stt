import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import settings
from core import session as session_mgr
from core import transcriber as tr

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

DEMO_PATH = os.path.join(os.path.dirname(__file__), "..", "demo", "index.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    import multiprocessing
    num_workers = min(settings.max_clients, max(1, multiprocessing.cpu_count()))
    tr.load_model(
        model_size=settings.model_size,
        device=settings.device,
        compute_type=settings.compute_type,
        model_cache_dir=settings.model_cache_dir,
        num_workers=num_workers,
    )
    yield
    # Shutdown: unload model
    tr.unload_model()


app = FastAPI(title="Realtime STT", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_demo():
    demo_abs = os.path.abspath(DEMO_PATH)
    if os.path.exists(demo_abs):
        return FileResponse(demo_abs, media_type="text/html")
    return JSONResponse({"error": "Demo file not found"}, status_code=404)


@app.get("/health")
async def health():
    return {
        "status": "ok" if tr.is_ready() else "loading",
        "model": settings.model_size,
        "device": settings.device,
        "active_connections": session_mgr.get_active_connections(),
        "max_clients": settings.max_clients,
    }


@app.websocket("/asr")
async def websocket_asr(websocket: WebSocket):
    await session_mgr.handle_session(websocket, settings)
