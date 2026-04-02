"""
Feature: 健康检查接口 /health
"""
import importlib
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

import core.session as session_module
import core.transcriber as tr_module


def _get_test_client():
    with patch("core.transcriber.load_model"), patch("core.transcriber.unload_model"):
        import main as main_mod
        importlib.reload(main_mod)
        return TestClient(main_mod.app, raise_server_exceptions=False)


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed")
class TestHealthEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = _get_test_client()

    def test_ok_when_model_ready(self):
        with patch.object(tr_module, "is_ready", return_value=True), \
             patch.object(session_module, "get_active_connections", return_value=2):
            resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model", data)
        self.assertIn("active_connections", data)

    def test_loading_when_model_not_ready(self):
        with patch.object(tr_module, "is_ready", return_value=False):
            resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "loading")


if __name__ == "__main__":
    unittest.main(verbosity=2)
