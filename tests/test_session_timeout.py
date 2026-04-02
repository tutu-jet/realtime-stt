"""
Feature: 会话超时 & 静音超时
"""
import asyncio
import json
import time
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


# ---------------------------------------------------------------------------
# silence_duration_sec — mirrors AudioPipeline property logic without scipy
# ---------------------------------------------------------------------------

def _silence_duration_sec(silence_start_time):
    if silence_start_time is None:
        return 0.0
    return max(0.0, time.monotonic() - silence_start_time)


class TestSilenceDurationSec(unittest.TestCase):

    def test_zero_before_silence(self):
        """Returns 0.0 when silence tracking hasn't started."""
        self.assertEqual(_silence_duration_sec(None), 0.0)

    def test_grows_during_silence(self):
        """Returns elapsed seconds once silence begins."""
        duration = _silence_duration_sec(time.monotonic() - 2.5)
        self.assertGreaterEqual(duration, 2.4)
        self.assertLess(duration, 3.5)

    def test_resets_to_zero_on_speech(self):
        """Setting silence_start_time to None models speech resumption."""
        self.assertGreater(_silence_duration_sec(time.monotonic() - 3.0), 0)
        self.assertEqual(_silence_duration_sec(None), 0.0)

    def test_non_negative(self):
        """Must never return negative (clock-skew edge case: future timestamp)."""
        self.assertEqual(_silence_duration_sec(time.monotonic() + 10.0), 0.0)


# ---------------------------------------------------------------------------
# Session-level guards
# ---------------------------------------------------------------------------

class TestSessionGuards(unittest.TestCase):

    def setUp(self):
        session_module._active_connections = 0

    def test_rejects_when_model_not_ready(self):
        ws = _make_websocket()

        async def run():
            with patch.object(tr_module, "is_ready", return_value=False):
                await handle_session(ws, _make_settings())

        asyncio.run(run())
        sent = json.loads(ws.send_text.call_args[0][0])
        self.assertEqual(sent["code"], "MODEL_NOT_READY")

    def test_cleans_up_on_handshake_timeout(self):
        ws = _make_websocket(recv_side_effect=asyncio.TimeoutError())

        async def run():
            with patch.object(tr_module, "is_ready", return_value=True):
                await handle_session(ws, _make_settings())

        asyncio.run(run())
        self.assertEqual(session_module._active_connections, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
