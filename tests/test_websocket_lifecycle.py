"""
Feature: WebSocket 握手 & 完整会话生命周期
"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import core.session as session_module
import core.transcriber as tr_module
from core.session import handle_session


def _make_settings(**overrides):
    defaults = dict(
        max_connections=0,
        max_clients=10,
        session_timeout_sec=0,
        max_connection_time=600,
        silence_timeout_sec=0,
        model_size="medium",
        language=None,
        vad_backend="webrtcvad",
        vad_aggressiveness=2,
        vad_frame_ms=30,
        vad_silence_threshold_ms=600,
        vad_min_speech_ms=250,
        beam_size=5,
        best_of=5,
        temperature=0.0,
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_websocket(recv_side_effect=None):
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    ws.receive_text = AsyncMock(
        side_effect=recv_side_effect if recv_side_effect is not None else [None]
    )
    ws.receive = AsyncMock()
    return ws


def _make_pipeline():
    p = MagicMock()
    p.start = MagicMock()
    p.close = AsyncMock()
    p.finalize = AsyncMock()
    p.silence_duration_sec = 0.0
    return p


class TestWebSocketLifecycle(unittest.TestCase):

    def setUp(self):
        session_module._active_connections = 0

    def test_config_ack_sent_and_counter_restored(self):
        """After successful handshake: first message is type='config' ack; counter returns to 0."""
        config_json = json.dumps({"uid": "test-uid", "language": "en", "task": "transcribe"})
        ws = _make_websocket(recv_side_effect=[config_json])
        ws.receive = AsyncMock(side_effect=[
            {"type": "websocket.receive", "text": "END_OF_AUDIO", "bytes": None},
            {"type": "websocket.disconnect"},
        ])

        async def run():
            with patch.object(tr_module, "is_ready", return_value=True), \
                 patch("core.session.AudioPipeline", return_value=_make_pipeline()):
                await handle_session(ws, _make_settings())

        asyncio.run(run())
        ack = json.loads(ws.send_text.call_args_list[0][0][0])
        self.assertEqual(ack["type"], "config")
        self.assertEqual(session_module._active_connections, 0)

    def test_disconnect_cleans_up_pipeline_and_counter(self):
        """Client disconnect must call pipeline.close() and restore the counter."""
        from fastapi import WebSocketDisconnect

        config_json = json.dumps({"uid": "disc-uid", "language": None, "task": "transcribe"})
        ws = _make_websocket(recv_side_effect=[config_json])
        ws.receive = AsyncMock(side_effect=WebSocketDisconnect())

        pipeline = _make_pipeline()

        async def run():
            with patch.object(tr_module, "is_ready", return_value=True), \
                 patch("core.session.AudioPipeline", return_value=pipeline):
                await handle_session(ws, _make_settings())

        asyncio.run(run())
        self.assertEqual(session_module._active_connections, 0)
        pipeline.close.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
