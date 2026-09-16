"""Small, dependency-free HTTP helpers shared by isolated module services."""

import json
import math
from urllib.parse import urlsplit


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(handler, error=ValueError):
    """Reject unbounded/ambiguous bodies before attempting a blocking read."""
    try:
        if handler.headers.get("Transfer-Encoding"):
            raise ValueError("不支持 Transfer-Encoding")
        lengths = handler.headers.get_all("Content-Length", [])
        if len(lengths) > 1:
            raise ValueError("Content-Length 重复")
        size = int(handler.headers.get("Content-Length", "0"))
        if not 0 <= size <= 1_000_000:
            raise ValueError("请求体长度必须在 0 到 1 MB 之间")
        origin = handler.headers.get("Origin")
        if origin and urlsplit(origin).netloc != handler.headers.get("Host"):
            raise ValueError("拒绝来自其他网站的控制请求")
        value = json.loads(handler.rfile.read(size) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("请求体必须为 JSON 对象")
        return value
    except (ValueError, UnicodeError) as exc:
        raise error(str(exc)) from exc


class LocalHandlerMixin:
    """Bound idle reads, handle browser disconnects and disallow cross-site POST."""

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    def parse_request(self):
        if not super().parse_request():
            return False
        if self.command == "POST":
            origin = self.headers.get("Origin")
            if (
                origin and urlsplit(origin).netloc != self.headers.get("Host")
            ) or self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.send_error(403, "Cross-site control requests are not allowed")
                return False
        return True

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        super().end_headers()
