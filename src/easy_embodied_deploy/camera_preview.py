"""Read-only, on-demand preview of cameras already owned by the recorder."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from .http import LocalHandlerMixin


class CameraPreview:
    def __init__(self, cameras):
        self.cameras = cameras
        self.locks = {name: threading.Lock() for name in cameras}
        self.cache = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), PreviewHandler)
        self.server.daemon_threads = True
        self.server.preview = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True,
                                       name="camera-preview")
        self.thread.start()

    @property
    def port(self):
        return self.server.server_port

    def frame(self, name):
        if name not in self.cameras:
            raise KeyError(name)
        with self.locks[name]:
            # read() only snapshots the latest immutable RGB frame; it neither
            # consumes an SDK frame nor holds the capture lock while encoding.
            frame = self.cameras[name].read()
            now = time.monotonic()
            previous = self.cache.get(name)
            if previous and (previous[0] == frame.timestamp or now - previous[1] < 0.1):
                return previous[2]
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError("图像编码失败")
            value = encoded.tobytes()
            self.cache[name] = (frame.timestamp, now, value)
            return value

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class PreviewHandler(LocalHandlerMixin, BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        try:
            body = self.server.preview.frame(self.path.removeprefix("/"))
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except KeyError:
            self.send_error(404, "Unknown camera")
        except (RuntimeError, cv2.error):
            self.send_error(503, "Camera frame unavailable")
