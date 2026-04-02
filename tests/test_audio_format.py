"""
Feature: 音频格式支持 — PCM/AAC 输入、int16 边界、重采样
"""
import struct
import unittest

import numpy as np

from utils.audio import (
    float32_bytes_to_array,
    float32_to_pcm_bytes,
    is_adts_frame,
    pcm_bytes_to_float32,
    resample_audio,
)


class TestPcmBytesToFloat32(unittest.TestCase):

    def test_max_positive(self):
        """int16 max (32767) maps to ~1.0 — divisor 32768 is easy to get wrong."""
        arr = pcm_bytes_to_float32(struct.pack("<h", 32767))
        self.assertAlmostEqual(arr[0], 32767 / 32768.0, places=4)

    def test_min_negative(self):
        """int16 min (-32768) maps to -1.0."""
        arr = pcm_bytes_to_float32(struct.pack("<h", -32768))
        self.assertAlmostEqual(arr[0], -32768 / 32768.0, places=4)


class TestFloat32ToPcmBytes(unittest.TestCase):

    def test_clipping_positive(self):
        result = float32_to_pcm_bytes(np.array([2.0], dtype=np.float32))
        self.assertEqual(struct.unpack("<h", result)[0], 32767)

    def test_clipping_negative(self):
        result = float32_to_pcm_bytes(np.array([-2.0], dtype=np.float32))
        self.assertEqual(struct.unpack("<h", result)[0], -32767)

    def test_roundtrip(self):
        audio = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        back = pcm_bytes_to_float32(float32_to_pcm_bytes(audio))
        for a, b in zip(audio, back):
            self.assertAlmostEqual(a, b, places=3)


class TestFloat32BytesToArray(unittest.TestCase):

    def test_is_writable_copy(self):
        """Result must be a writable copy — silently using a read-only view causes hard-to-find bugs."""
        result = float32_bytes_to_array(np.array([1.0], dtype=np.float32).tobytes())
        result[0] = 99.0  # must not raise


class TestIsAdtsFrame(unittest.TestCase):

    def test_valid_adts_header(self):
        self.assertTrue(is_adts_frame(b"\xFF\xF1" + b"\x00" * 5))

    def test_invalid_first_byte(self):
        self.assertFalse(is_adts_frame(b"\xFE\xF1" + b"\x00" * 5))

    def test_invalid_second_byte(self):
        self.assertFalse(is_adts_frame(b"\xFF\x00" + b"\x00" * 5))

    def test_too_short(self):
        self.assertFalse(is_adts_frame(b"\xFF"))

    def test_empty(self):
        self.assertFalse(is_adts_frame(b""))


class TestResampleAudio(unittest.TestCase):

    def test_no_op_same_rate(self):
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(audio, resample_audio(audio, orig_sr=16000, target_sr=16000))


if __name__ == "__main__":
    unittest.main(verbosity=2)
