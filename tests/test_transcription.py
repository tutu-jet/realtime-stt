"""
Feature: 转写 — guard、初始状态、executor offload
"""
import asyncio
import unittest
from unittest.mock import MagicMock

import numpy as np

import core.transcriber as tr_module


class TestTranscriberState(unittest.TestCase):

    def setUp(self):
        tr_module._model = None
        tr_module._executor = None

    def test_is_ready_false_initially(self):
        self.assertFalse(tr_module.is_ready())

    def test_unload_model_clears_state(self):
        """unload_model must shut down the executor and clear both module-level vars."""
        executor_mock = MagicMock()
        tr_module._model = MagicMock()
        tr_module._executor = executor_mock
        tr_module.unload_model()
        self.assertIsNone(tr_module._model)
        self.assertIsNone(tr_module._executor)
        executor_mock.shutdown.assert_called_once_with(wait=False)

    def test_transcribe_chunk_raises_when_not_loaded(self):
        async def run():
            with self.assertRaises(RuntimeError):
                await tr_module.transcribe_chunk(np.zeros(1000, dtype=np.float32))
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
