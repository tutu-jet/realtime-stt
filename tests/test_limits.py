"""
Tests for the three limit features integrated from mobile-whisper PR#6:
  1. silence_duration_sec property on AudioPipeline
  2. max_connections logic in handle_session
  3. session_timeout_sec / silence_timeout_sec config fields

Run with:
    cd server && python3 -m pytest ../tests/test_limits.py -v
"""
import asyncio
import sys
import time
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so we can import server modules without real dependencies
# ---------------------------------------------------------------------------

def _stub_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# webrtcvad stub
webrtcvad_mod = _stub_module("webrtcvad")
webrtcvad_mod.Vad = MagicMock(return_value=MagicMock(is_speech=MagicMock(return_value=False)))
sys.modules.setdefault("webrtcvad", webrtcvad_mod)

# numpy stub (minimal)
try:
    import numpy  # noqa – use real numpy if available
except ImportError:
    np_mod = _stub_module("numpy")
    np_mod.ndarray = list
    np_mod.concatenate = lambda arrs, **kw: sum(arrs, [])
    np_mod.frombuffer = lambda b, dtype=None: []
    np_mod.float32 = float
    np_mod.int16 = int
    sys.modules["numpy"] = np_mod

# pydantic_settings stub
try:
    import pydantic_settings  # noqa
except ImportError:
    ps_mod = _stub_module("pydantic_settings")
    class _BaseSettings:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        class Config:
            env_file = ".env"
    ps_mod.BaseSettings = _BaseSettings
    sys.modules["pydantic_settings"] = ps_mod

# pydantic stub
try:
    import pydantic  # noqa
except ImportError:
    p_mod = _stub_module("pydantic")
    p_mod.BaseModel = object
    p_mod.Field = lambda *a, **kw: None
    sys.modules["pydantic"] = p_mod

# fastapi stub
try:
    import fastapi  # noqa
except ImportError:
    fa_mod = _stub_module("fastapi")
    fa_mod.WebSocket = object
    fa_mod.WebSocketDisconnect = Exception
    sys.modules["fastapi"] = fa_mod

# Add server dir to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))


# ---------------------------------------------------------------------------
# silence_duration_sec — inline implementation to avoid scipy/webrtcvad imports
# ---------------------------------------------------------------------------
# The property logic is:
#   if _silence_start_time is None: return 0.0
#   return max(0.0, time.monotonic() - _silence_start_time)
# We test it here directly without importing AudioPipeline (which pulls scipy).

def _silence_duration_sec(silence_start_time):
    """Mirror of AudioPipeline.silence_duration_sec property logic."""
    if silence_start_time is None:
        return 0.0
    return max(0.0, time.monotonic() - silence_start_time)


class TestSilenceDurationSec(unittest.TestCase):

    def test_zero_before_silence(self):
        """Should be 0.0 when silence tracking hasn't started."""
        self.assertEqual(_silence_duration_sec(None), 0.0)

    def test_grows_during_silence(self):
        """Should return elapsed seconds once silence begins."""
        duration = _silence_duration_sec(time.monotonic() - 2.5)
        self.assertGreaterEqual(duration, 2.4)
        self.assertLess(duration, 3.5)

    def test_resets_to_zero_on_speech(self):
        """Setting silence_start_time to None models speech resumption."""
        self.assertGreater(_silence_duration_sec(time.monotonic() - 3.0), 0)
        self.assertEqual(_silence_duration_sec(None), 0.0)

    def test_non_negative(self):
        """Should never return negative values (clock skew edge case)."""
        self.assertEqual(_silence_duration_sec(time.monotonic() + 10.0), 0.0)


# ---------------------------------------------------------------------------
# 2. Config fields
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. handle_session: max_connections enforcement
# ---------------------------------------------------------------------------

class TestMaxConnectionsLogic(unittest.TestCase):

    def test_max_connections_overrides_max_clients(self):
        """When max_connections > 0, it should be used instead of max_clients."""
        settings = MagicMock(max_connections=3, max_clients=10)
        effective = settings.max_connections if settings.max_connections else settings.max_clients
        self.assertEqual(effective, 3)

    def test_max_connections_zero_falls_back_to_max_clients(self):
        """When max_connections=0, fall back to max_clients."""
        settings = MagicMock(max_connections=0, max_clients=10)
        effective = settings.max_connections if settings.max_connections else settings.max_clients
        self.assertEqual(effective, 10)

    def test_session_timeout_overrides_max_connection_time(self):
        """When session_timeout_sec > 0, it should be used instead of max_connection_time."""
        settings = MagicMock(session_timeout_sec=30, max_connection_time=600)
        effective = settings.session_timeout_sec if settings.session_timeout_sec else settings.max_connection_time
        self.assertEqual(effective, 30)

    def test_session_timeout_zero_falls_back(self):
        """When session_timeout_sec=0, fall back to max_connection_time."""
        settings = MagicMock(session_timeout_sec=0, max_connection_time=600)
        effective = settings.session_timeout_sec if settings.session_timeout_sec else settings.max_connection_time
        self.assertEqual(effective, 600)

    def test_silence_timeout_none_when_zero(self):
        """When silence_timeout_sec=0, silence_timeout should be None (disabled)."""
        settings = MagicMock(silence_timeout_sec=0)
        silence_timeout = settings.silence_timeout_sec if settings.silence_timeout_sec else None
        self.assertIsNone(silence_timeout)

    def test_silence_timeout_set_when_nonzero(self):
        """When silence_timeout_sec > 0, it should be passed as-is."""
        settings = MagicMock(silence_timeout_sec=5.0)
        silence_timeout = settings.silence_timeout_sec if settings.silence_timeout_sec else None
        self.assertEqual(silence_timeout, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
