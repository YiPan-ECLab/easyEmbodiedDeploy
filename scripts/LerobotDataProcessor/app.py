"""Small local web UI for the HDF5 -> LeRobot processing pipeline."""
from __future__ import annotations

import json
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from processor import PROJECT_ROOT, ProcessingCancelled, Settings, run, scan
from convert_piper_hdf5 import (DEFAULT_ACTION_MODE, DEFAULT_FPS, DEFAULT_INPUT_DIR,
                                DEFAULT_LIMIT, DEFAULT_OUTPUT_ROOT, DEFAULT_REPO_ID,
                                DEFAULT_SOURCE_FPS, TASK_DESCRIPTION)

STATIC = Path(__file__).parent / "static"
LOCK = threading.Lock()
STATE = {"status": "idle", "events": [], "result": None, "error": None}
CANCEL_EVENT = threading.Event()


def settings_from(payload: dict) -> Settings:
    return Settings(
        input_dir=str(payload.get("input_dir") or DEFAULT_INPUT_DIR),
        output_root=payload.get("output_root") or DEFAULT_OUTPUT_ROOT,
        task=str(payload.get("task") or TASK_DESCRIPTION),
        target_fps=int(payload.get("target_fps", DEFAULT_FPS)),
        source_fps=(int(payload["source_fps"]) if payload.get("source_fps") else DEFAULT_SOURCE_FPS),
        action_mode=str(payload.get("action_mode") or DEFAULT_ACTION_MODE),
        repo_id=payload.get("repo_id") or DEFAULT_REPO_ID,
        limit=int(payload.get("limit", DEFAULT_LIMIT) or 0),
    )


def start_job(settings: Settings) -> None:
    global STATE
    with LOCK:
        if STATE["status"] == "running":
            raise RuntimeError("已有转换任务正在运行")
        STATE = {"status": "running", "events": [], "result": None, "error": None}
        CANCEL_EVENT.clear()
    def progress(event: dict) -> None:
        with LOCK:
            STATE["events"].append(event)
            STATE["events"] = STATE["events"][-200:]
    def worker() -> None:
        global STATE
        try:
            result = run(settings, progress, CANCEL_EVENT)
            with LOCK:
                STATE.update(status="complete", result=result)
        except ProcessingCancelled as exc:
            with LOCK:
                STATE.update(status="cancelled", error=str(exc))
        except Exception as exc:
            with LOCK:
                STATE.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    threading.Thread(target=worker, name="data-processor", daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "DataProcessor/1.0"

    def _send(self, payload, status=HTTPStatus.OK, content_type="application/json; charset=utf-8"):
        data = payload if isinstance(payload, bytes) else (json.dumps(payload, ensure_ascii=False).encode() if not isinstance(payload, str) else payload.encode())
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/api/defaults":
            return self._send({"project_root": str(PROJECT_ROOT), "input_dir": DEFAULT_INPUT_DIR,
                               "output_root": DEFAULT_OUTPUT_ROOT, "repo_id": DEFAULT_REPO_ID,
                               "task": TASK_DESCRIPTION, "source_fps": DEFAULT_SOURCE_FPS,
                               "target_fps": DEFAULT_FPS, "action_mode": DEFAULT_ACTION_MODE,
                               "limit": DEFAULT_LIMIT})
        if url.path == "/api/status":
            with LOCK:
                return self._send(dict(STATE))
        if url.path == "/api/inspect":
            try:
                query = parse_qs(url.query)
                payload = {"input_dir": query.get("input_dir", [DEFAULT_INPUT_DIR])[0],
                           "target_fps": int(query.get("target_fps", [DEFAULT_FPS])[0]),
                           "action_mode": query.get("action_mode", [DEFAULT_ACTION_MODE])[0],
                           "source_fps": (int(query["source_fps"][0]) if query.get("source_fps", [""])[0] else None),
                           "task": TASK_DESCRIPTION}
                return self._send(scan(settings_from(payload)))
            except Exception as exc:
                return self._send({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)
        if url.path == "/" or url.path == "/index.html":
            return self._send((STATIC / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
        if url.path in ("/app.js", "/styles.css"):
            path = STATIC / url.path[1:]
            if path.is_file():
                return self._send(path.read_bytes(), content_type="text/javascript; charset=utf-8" if path.suffix == ".js" else "text/css; charset=utf-8")
        return self._send({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/cancel":
            with LOCK:
                if STATE["status"] != "running":
                    return self._send({"error": "当前没有正在运行的任务"}, HTTPStatus.CONFLICT)
                CANCEL_EVENT.set()
            return self._send({"status": "cancelling"}, HTTPStatus.ACCEPTED)
        if route != "/api/start":
            return self._send({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            start_job(settings_from(payload))
            return self._send({"status": "started"}, HTTPStatus.ACCEPTED)
        except Exception as exc:
            return self._send({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format, *args):
        return


def run_server(host="127.0.0.1", port=8766):
    print(f"DataProcessor: http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
