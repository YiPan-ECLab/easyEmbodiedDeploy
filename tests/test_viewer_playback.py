"""Playback reads stay byte-equivalent, bounded and read-only under concurrency."""

import hashlib
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/PiperViewer"))
from piper_viewer import server


class PlaybackCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "episode.hdf5"
        self.frames = np.random.default_rng(0).integers(0, 256, (24, 16, 16, 3), dtype=np.uint8)
        with h5py.File(self.path, "w") as handle:
            handle.attrs.update(format="pyagxarm_direct_v1", normalized=False)
            demo = handle.create_group("data/demo_0")
            for name in ("states", "actions"):
                demo[name] = np.zeros((24, 14))
            for name in ("timestamps", "camera_timestamps", "action_source_timestamps",
                         "trajectory_phase", "rewards", "dones"):
                demo[name] = np.arange(24)
            obs = demo.create_group("obs")
            for name in ("agentview", "left_eye_in_hand", "right_eye_in_hand"):
                obs.create_dataset(name, data=self.frames, compression="lzf", chunks=(8, 8, 8, 1))
        self.cache = server.FrameBlockCache()
        for name, value in (("FRAME_BLOCK_CACHE", self.cache), ("IMAGE_CACHE", server.ImageCache())):
            mock = patch.object(server, name, value)
            mock.start()
            self.addCleanup(mock.stop)

    def expected(self, index):
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(self.frames[index], cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.assertTrue(ok)
        return jpeg.tobytes()

    def test_sequential_and_concurrent_reads_reuse_blocks_without_changing_images(self):
        digest = hashlib.sha256(self.path.read_bytes()).digest()
        with patch.object(server.h5py, "File", wraps=h5py.File) as opened:
            with ThreadPoolExecutor(max_workers=8) as pool:
                actual = list(pool.map(lambda i: server.read_frame(self.path, None, "agentview", i), range(8)))
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(actual, [self.expected(i) for i in range(8)])
            for index in (8, 15, 16, 23, 0):
                self.assertEqual(server.read_frame(self.path, None, "agentview", index), self.expected(index))
            self.assertEqual(opened.call_count, 3)
            self.assertTrue(all(call.args[1] == "r" for call in opened.call_args_list))
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).digest(), digest)

    def test_memory_bound_eviction_and_random_seek(self):
        frame_bytes = self.frames[0].nbytes
        self.cache.limit = frame_bytes * 5
        self.cache.block_limit = frame_bytes * 3
        for index in (23, 0, 10, 7, 22, 4):
            for camera in ("agentview", "left_eye_in_hand", "right_eye_in_hand"):
                self.assertEqual(server.read_frame(self.path, None, camera, index), self.expected(index))
                self.assertLessEqual(self.cache.size, self.cache.limit)
                self.assertEqual(self.cache.size, sum(block.nbytes for block in self.cache.values.values()))

    def test_source_update_invalidates_both_caches_and_releases_file(self):
        before = server.read_frame(self.path, None, "agentview", 0)
        stat = self.path.stat()
        self.frames[0] = 0
        with h5py.File(self.path, "r+") as handle:
            handle["data/demo_0/obs/agentview"][0] = self.frames[0]
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        after = server.read_frame(self.path, None, "agentview", 0)
        self.assertNotEqual(before, after)
        self.assertEqual(after, self.expected(0))


if __name__ == "__main__":
    unittest.main()
