"""
Shared test fixtures: stub heavy dependencies and add server/ to sys.path.
Imported automatically by pytest before any test file.
"""
import os
import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# ---------------------------------------------------------------------------
# webrtcvad stub
# ---------------------------------------------------------------------------
_vad_instance = MagicMock()
_vad_instance.is_speech = MagicMock(return_value=False)
webrtcvad_mod = _stub_module("webrtcvad")
webrtcvad_mod.Vad = MagicMock(return_value=_vad_instance)
sys.modules.setdefault("webrtcvad", webrtcvad_mod)

# expose for tests that need to manipulate Vad mock
sys.modules["webrtcvad"]  # ensure it's registered


# ---------------------------------------------------------------------------
# numpy — use real if installed, else minimal stub
# ---------------------------------------------------------------------------
try:
    import numpy  # noqa
except ImportError:
    np_mod = _stub_module("numpy")
    np_mod.ndarray = list
    np_mod.concatenate = lambda arrs, **kw: sum(arrs, [])
    np_mod.frombuffer = lambda b, dtype=None: []
    np_mod.float32 = float
    np_mod.int16 = int
    np_mod.clip = lambda a, lo, hi: a
    sys.modules["numpy"] = np_mod


# ---------------------------------------------------------------------------
# scipy — use real if installed, else minimal stub
# ---------------------------------------------------------------------------
try:
    import scipy  # noqa
except ImportError:
    sp_mod = _stub_module("scipy")
    sp_signal = _stub_module("scipy.signal")
    sp_signal.resample_poly = lambda audio, up, down: audio
    sp_mod.signal = sp_signal
    sys.modules.setdefault("scipy", sp_mod)
    sys.modules.setdefault("scipy.signal", sp_signal)


# ---------------------------------------------------------------------------
# pydantic / pydantic_settings — use real if installed, else minimal stub
# ---------------------------------------------------------------------------
try:
    import pydantic  # noqa
except ImportError:
    p_mod = _stub_module("pydantic")
    p_mod.BaseModel = object
    p_mod.Field = lambda *a, **kw: None
    sys.modules["pydantic"] = p_mod

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


# ---------------------------------------------------------------------------
# fastapi — use real if installed, else minimal stub
# ---------------------------------------------------------------------------
try:
    import fastapi  # noqa
except ImportError:
    fa_mod = _stub_module("fastapi")
    fa_mod.WebSocket = object
    fa_mod.WebSocketDisconnect = Exception
    fa_mod.FastAPI = MagicMock
    sys.modules["fastapi"] = fa_mod


# ---------------------------------------------------------------------------
# Add server/ to sys.path so tests can import server modules directly
# ---------------------------------------------------------------------------
_server_dir = os.path.join(os.path.dirname(__file__), "..", "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)
