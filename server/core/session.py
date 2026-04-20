"""
WebSocket session lifecycle management.
"""
import asyncio
import logging
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from core import transcriber as tr
from core.audio_pipeline import AudioPipeline
from models.messages import (
    ClientConfigMessage,
    ErrorMessage,
    ReadyToStopMessage,
    SegmentMessage,
    ServerConfigMessage,
)
from models.session_state import SessionState

logger = logging.getLogger(__name__)

_active_connections: int = 0


def get_active_connections() -> int:
    return _active_connections


async def handle_session(websocket: WebSocket, settings) -> None:
    """Handle a single WebSocket connection lifecycle."""
    global _active_connections

    await websocket.accept()

    # Enforce connection limit
    if _active_connections >= settings.max_clients:
        err = ErrorMessage(uid="", code="MAX_CLIENTS_REACHED", message="Server is at capacity.")
        await websocket.send_text(err.model_dump_json())
        await websocket.close()
        return

    if not tr.is_ready():
        err = ErrorMessage(uid="", code="MODEL_NOT_READY", message="Model is still loading.")
        await websocket.send_text(err.model_dump_json())
        await websocket.close()
        return

    _active_connections += 1
    uid = "unknown"
    send_queue: asyncio.Queue = asyncio.Queue()
    pipeline: AudioPipeline | None = None

    try:
        # Handshake: first message must be JSON config
        raw_config = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        config = ClientConfigMessage.model_validate_json(raw_config)
        uid = config.uid

        session_state = SessionState(
            uid=uid,
            language=config.language or settings.language or None,
            task=config.task,
            connected_at=datetime.utcnow(),
        )

        ack = ServerConfigMessage(
            uid=uid,
            model=settings.model_size,
            language=session_state.language,
        )
        await websocket.send_text(ack.model_dump_json())
        logger.info(f"[{uid}] Session started. language={session_state.language}, task={config.task}")

        pipeline = AudioPipeline(session_state, send_queue, settings)
        pipeline.start()

        # Determine effective session timeout
        session_timeout = settings.session_timeout_sec if settings.session_timeout_sec else None
        silence_timeout = settings.silence_timeout_sec if settings.silence_timeout_sec else None

        try:
            await asyncio.wait_for(
                _run_session(websocket, pipeline, send_queue, session_state, uid, silence_timeout),
                timeout=session_timeout,
            )
        except asyncio.TimeoutError:
            err = ErrorMessage(uid=uid, code="SESSION_TIMEOUT", message="Session time limit reached.")
            await websocket.send_text(err.model_dump_json())

    except WebSocketDisconnect:
        logger.info(f"[{uid}] Client disconnected.")
    except Exception as e:
        logger.exception(f"[{uid}] Session error: {e}")
    finally:
        if pipeline:
            await pipeline.close()
        _active_connections -= 1
        logger.info(f"[{uid}] Session ended. Active connections: {_active_connections}")


async def _run_session(
    websocket: WebSocket,
    pipeline: AudioPipeline,
    send_queue: asyncio.Queue,
    session_state: SessionState,
    uid: str,
    silence_timeout: float | None,
) -> None:
    """Run receive and send loops concurrently."""
    receive_task = asyncio.create_task(
        _receive_loop(websocket, pipeline, send_queue, session_state, uid, silence_timeout),
        name=f"recv-{uid}",
    )
    send_task = asyncio.create_task(
        _send_loop(websocket, send_queue, uid),
        name=f"send-{uid}",
    )

    done, pending = await asyncio.wait(
        [receive_task, send_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _receive_loop(
    websocket: WebSocket,
    pipeline: AudioPipeline,
    send_queue: asyncio.Queue,
    session_state: SessionState,
    uid: str,
    silence_timeout: float | None,
) -> None:
    """Receive audio frames and control messages from the client."""
    receive_task = asyncio.create_task(websocket.receive(), name=f"ws-recv-{uid}")
    try:
        while True:
            # Check silence timeout if enabled
            if silence_timeout is not None and pipeline.silence_duration_sec >= silence_timeout:
                receive_task.cancel()
                try:
                    await receive_task
                except (asyncio.CancelledError, Exception):
                    pass
                err = ErrorMessage(uid=uid, code="SILENCE_TIMEOUT", message=f"No speech detected for {silence_timeout}s.")
                await websocket.send_text(err.model_dump_json())
                await websocket.close(1008)
                return

            wait_timeout = 1.0 if silence_timeout is not None else None
            try:
                done, _ = await asyncio.wait({receive_task}, timeout=wait_timeout)
            except Exception:
                receive_task.cancel()
                try:
                    await receive_task
                except (asyncio.CancelledError, Exception):
                    pass
                raise

            if not done:
                # Poll interval elapsed — re-check silence timeout
                continue

            msg = receive_task.result()
            receive_task = asyncio.create_task(websocket.receive(), name=f"ws-recv-{uid}")

            if msg["type"] == "websocket.disconnect":
                break

            if msg["type"] == "websocket.receive":
                if msg.get("bytes"):
                    await pipeline.feed(msg["bytes"])
                elif msg.get("text"):
                    text = msg["text"].strip()
                    if text == "END_OF_AUDIO":
                        logger.info(f"[{uid}] END_OF_AUDIO received, finalizing…")
                        await pipeline.finalize()

                        lines = [
                            SegmentMessage(
                                text=seg.text,
                                start=seg.start,
                                end=seg.end,
                                detected_language=seg.language,
                                no_speech_prob=seg.no_speech_prob,
                            )
                            for seg in session_state.segments
                        ]
                        done_msg = ReadyToStopMessage(
                            uid=uid,
                            lines=lines,
                            buffer_transcription="",
                        )
                        await send_queue.put(done_msg.model_dump_json())
                        await send_queue.put(None)  # signal send loop to stop
                        break
    finally:
        if not receive_task.done():
            receive_task.cancel()
            try:
                await receive_task
            except (asyncio.CancelledError, Exception):
                pass


async def _send_loop(websocket: WebSocket, send_queue: asyncio.Queue, uid: str) -> None:
    """Forward queued JSON messages to the WebSocket client."""
    while True:
        msg = await send_queue.get()
        if msg is None:
            break
        try:
            await websocket.send_text(msg)
        except Exception as e:
            logger.warning(f"[{uid}] Send error: {e}")
            break
