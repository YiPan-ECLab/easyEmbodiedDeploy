"""HTTP shell for the local resource and calibration consoles."""

import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .http import LocalHandlerMixin, json_safe

ROOT = Path(__file__).resolve().parents[2]


class ModuleHandler(LocalHandlerMixin):
    def log_message(self, *_args):
        pass

    def send(self, value, status=200, content_type="application/json; charset=utf-8"):
        data = (
            value
            if isinstance(value, bytes)
            else json.dumps(
                json_safe(value), ensure_ascii=False, allow_nan=False
            ).encode()
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def static(self, directory):
        route = unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
        if route == "shared.css":
            path = ROOT / "web/shared/module.css"
        else:
            base = (ROOT / "web" / directory).resolve()
            path = (base / route).resolve()
            if not path.is_relative_to(base):
                raise ValueError("无效资源路径")
        if not path.is_file():
            self.send({"error": "资源不存在"}, 404)
            return
        self.send(
            path.read_bytes(),
            content_type=mimetypes.guess_type(path.name)[0]
            or "application/octet-stream",
        )

    def fail(self, error):
        self.send(
            {"error": str(error)},
            400 if isinstance(error, (ValueError, KeyError)) else 409,
        )
