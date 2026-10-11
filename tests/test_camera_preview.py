"""Camera preview never owns devices or consumes recording/operator events."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np

from easy_embodied_deploy.camera_preview import CameraPreview
from easy_embodied_deploy.recorder_console import RecorderSession, Handler
from http.server import ThreadingHTTPServer


class CameraPreviewTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((32, 32, 3), [230, 30, 10], dtype=np.uint8)
        self.camera = Mock()
        self.camera.read.return_value = SimpleNamespace(rgb=self.rgb, timestamp=1.0)
        self.preview = CameraPreview({name: self.camera for name in
                                      ('agentview', 'left_eye_in_hand', 'right_eye_in_hand')})
        self.addCleanup(self.preview.close)

    def test_jpeg_has_correct_colors_and_leaves_recording_frames_unchanged(self):
        original = self.rgb.copy()
        first = self.preview.frame('agentview')
        self.assertEqual(first, self.preview.frame('agentview'))
        bgr = cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_allclose(bgr[0, 0], [10, 30, 230], atol=3)
        np.testing.assert_array_equal(self.rgb, original)
        self.camera.start.assert_not_called()
        self.camera.close.assert_not_called()

    def test_camera_failure_does_not_serve_stale_cached_image_and_can_recover(self):
        self.preview.frame('agentview')
        self.camera.read.side_effect = RuntimeError('camera stopped')
        with self.assertRaises(HTTPError) as exc:
            urlopen(f'http://127.0.0.1:{self.preview.port}/agentview')
        self.assertEqual(exc.exception.code, 503)
        self.camera.read.side_effect = None
        with urlopen(f'http://127.0.0.1:{self.preview.port}/agentview') as response:
            self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
            self.assertTrue(response.read().startswith(b'\xff\xd8'))

    def session(self):
        session = RecorderSession.__new__(RecorderSession)
        session.lock = threading.RLock()
        session.generation = 2
        session.state = {'preview_port': self.preview.port, 'token': 8, 'accepts': ['finish']}
        session.process = Mock()
        session.process.poll.return_value = None
        return session

    def test_proxy_rejects_old_sessions_and_stopped_process(self):
        session = self.session()
        before = session.state.copy()
        for name in self.preview.cameras:
            self.assertTrue(session.camera_frame(name, '2').startswith(b'\xff\xd8'))
        self.assertEqual(before, session.state)
        session.process.stdin.write.assert_not_called()
        with self.assertRaises(RuntimeError):
            session.camera_frame('agentview', '1')
        with self.assertRaises(ValueError):
            session.camera_frame('../api/input', '2')
        session.process.poll.return_value = 0
        with self.assertRaises(RuntimeError):
            session.camera_frame('agentview', '2')

    def test_console_http_camera_route(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        server.session = self.session()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/api/camera-frame?camera=agentview&session_id=2'
            with urlopen(url) as response:
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
            self.camera.read.side_effect = RuntimeError('stale frame')
            with self.assertRaises(HTTPError) as exc:
                urlopen(url)
            self.assertEqual(exc.exception.code, 503)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
