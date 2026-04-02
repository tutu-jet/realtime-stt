"""
Feature: 连接数限制 — max_clients / max_connections 拒绝逻辑 & 配置字段默认值
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


def _make_websocket():
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    ws.receive_text = AsyncMock(side_effect=[None])
    ws.receive = AsyncMock()
    return ws


class TestConnectionLimitRejection(unittest.TestCase):

    def setUp(self):
        session_module._active_connections = 0

    def test_rejects_when_at_max_clients_capacity(self):
        session_module._active_connections = 10
        ws = _make_websocket()

        async def run():
            with patch.object(tr_module, "is_ready", return_value=True):
                await handle_session(ws, _make_settings(max_connections=0, max_clients=10))

        asyncio.run(run())
        sent = json.loads(ws.send_text.call_args[0][0])
        self.assertEqual(sent["code"], "MAX_CLIENTS_REACHED")
        ws.close.assert_called_once()

    def test_rejects_when_max_connections_exceeded(self):
        session_module._active_connections = 3
        ws = _make_websocket()

        async def run():
            with patch.object(tr_module, "is_ready", return_value=True):
                await handle_session(ws, _make_settings(max_connections=3, max_clients=100))

        asyncio.run(run())
        sent = json.loads(ws.send_text.call_args[0][0])
        self.assertEqual(sent["code"], "MAX_CLIENTS_REACHED")

    def test_counter_unchanged_after_rejection(self):
        session_module._active_connections = 10
        ws = _make_websocket()

        async def run():
            with patch.object(tr_module, "is_ready", return_value=True):
                await handle_session(ws, _make_settings(max_connections=0, max_clients=10))

        asyncio.run(run())
        self.assertEqual(session_module._active_connections, 10)


class TestConfigFields(unittest.TestCase):

    def test_new_fields_have_correct_defaults(self):
        from config import Settings
        s = Settings()
        self.assertEqual(s.max_connections, 0)
        self.assertEqual(s.session_timeout_sec, 0)
        self.assertEqual(s.silence_timeout_sec, 0)

    def test_fields_can_be_overridden(self):
        from config import Settings
        s = Settings(max_connections=5, session_timeout_sec=60, silence_timeout_sec=10)
        self.assertEqual(s.max_connections, 5)
        self.assertEqual(s.session_timeout_sec, 60)
        self.assertEqual(s.silence_timeout_sec, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
