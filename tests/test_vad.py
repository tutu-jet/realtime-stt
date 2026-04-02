"""
Feature: VAD 语音活动检测（精简版）
"""
import sys
import unittest

from core.vad import WebRTCVAD, create_vad

_webrtcvad_mod = sys.modules["webrtcvad"]


class TestWebRTCVAD(unittest.TestCase):

    def setUp(self):
        self.vad = WebRTCVAD(aggressiveness=2, frame_ms=30)

    def test_frame_size(self):
        # 16000 Hz × 30ms / 1000 = 480 samples — wrong constant → silent VAD failure
        self.assertEqual(self.vad.frame_size, 480)

    def test_frame_bytes(self):
        # 480 samples × 2 bytes/sample (int16) = 960 bytes
        self.assertEqual(self.vad.frame_bytes, 960)

    def test_process_frame_exception_returns_false(self):
        """VAD exceptions must be swallowed — this is explicit design, not an oversight."""
        _webrtcvad_mod.Vad.return_value.is_speech.side_effect = RuntimeError("oops")
        vad = WebRTCVAD()
        result = vad.process_frame(b"\x00" * vad.frame_bytes)
        self.assertFalse(result)
        _webrtcvad_mod.Vad.return_value.is_speech.side_effect = None  # restore


class TestCreateVad(unittest.TestCase):

    def test_unknown_backend_raises(self):
        """Adding a new backend without a matching branch must raise, not silently return None."""
        with self.assertRaises(ValueError):
            create_vad(backend="unknown_backend")


if __name__ == "__main__":
    unittest.main(verbosity=2)
