from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))

import web_server  # noqa: E402


class WebServerTests(unittest.TestCase):
    def test_static_assets_exist(self):
        for name in ("index.html", "styles.css", "app.js"):
            path = web_server.STATIC_DIR / name
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)

    def test_gpu_guard_blocks_selected_gpu_above_half_memory(self):
        supervisor = web_server.TrainingSupervisor()
        config = mock.Mock(device="cuda", gpu_ids=(1,))
        fake = [{"index": 1, "memory_used_mb": 13000, "memory_total_mb": 24000, "processes": []}]
        with mock.patch.object(web_server, "gpu_status", return_value=fake):
            with self.assertRaisesRegex(web_server.ConsoleError, "超过 50%"):
                supervisor._assert_gpu_available(config)

    def test_gpu_guard_allows_processes_at_half_memory_or_less(self):
        supervisor = web_server.TrainingSupervisor()
        config = mock.Mock(device="cuda", gpu_ids=(1,))
        fake = [
            {
                "index": 1,
                "memory_used_mb": 12000,
                "memory_total_mb": 24000,
                "processes": [{"pid": 123, "memory_mb": 800}],
            }
        ]
        with mock.patch.object(web_server, "gpu_status", return_value=fake):
            supervisor._assert_gpu_available(config)

    def test_gpu_guard_ignores_busy_unselected_gpu(self):
        supervisor = web_server.TrainingSupervisor()
        config = mock.Mock(device="cuda", gpu_ids=(0,))
        fake = [
            {"index": 0, "memory_used_mb": 0, "memory_total_mb": 24000, "processes": []},
            {"index": 1, "memory_used_mb": 20000, "memory_total_mb": 24000, "processes": []},
        ]
        with mock.patch.object(web_server, "gpu_status", return_value=fake):
            supervisor._assert_gpu_available(config)


if __name__ == "__main__":
    unittest.main()
