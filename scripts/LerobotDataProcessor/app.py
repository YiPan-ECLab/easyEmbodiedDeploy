"""Local web service for the validated HDF5 to LeRobot pipeline."""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from convert_piper_hdf5 import (  # noqa: E402
    DEFAULT_ACTION_MODE,
    DEFAULT_FPS,
    DEFAULT_GAP_POLICY,
    DEFAULT_INPUT_DIR,
    DEFAULT_LIMIT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPO_ID,
    DEFAULT_SOURCE_FPS,
    TASK_DESCRIPTION,
)
from processor import (  # noqa: E402
    ACTION_MODES,
    GAP_POLICIES,
    PROJECT_ROOT,
    ProcessingCancelled,
    Settings,
    run,
    scan,
    validate_lerobot_bundle,
)

STATIC_ROOT = Path(__file__).parent / "static"
STATE_LOCK = threading.Lock()
CANCEL_EVENT = threading.Event()
STATE = {
    "job_id": 0,
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "settings": None,
    "plan": None,
    "events": [],
    "event_seq": 0,
    "result": None,
    "error": None,
    "traceback": None,
}


def settings_from(payload: dict) -> Settings:
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    source_fps = payload.get("source_fps")
    return Settings(
        input_dir=str(payload.get("input_dir") or DEFAULT_INPUT_DIR),
        output_root=payload.get("output_root") or DEFAULT_OUTPUT_ROOT,
        task=str(payload.get("task") or TASK_DESCRIPTION),
        target_fps=int(payload.get("target_fps") or DEFAULT_FPS),
        source_fps=int(source_fps) if source_fps else DEFAULT_SOURCE_FPS,
        action_mode=str(payload.get("action_mode") or DEFAULT_ACTION_MODE),
        repo_id=payload.get("repo_id") or DEFAULT_REPO_ID,
        limit=int(payload.get("limit") or DEFAULT_LIMIT),
        gap_policy=str(payload.get("gap_policy") or DEFAULT_GAP_POLICY),
    )


def state_snapshot() -> dict:
    with STATE_LOCK:
        return copy.deepcopy(STATE)


def start_job(settings: Settings, plan: dict) -> int:
    with STATE_LOCK:
        if STATE["status"] == "running":
            raise RuntimeError("已有转换任务正在运行")
        job_id = int(STATE["job_id"]) + 1
        STATE.update(
            job_id=job_id,
            status="running",
            started_at=time.time(),
            finished_at=None,
            settings=settings.__dict__.copy(),
            plan=plan,
            events=[],
            event_seq=0,
            result=None,
            error=None,
            traceback=None,
        )
        CANCEL_EVENT.clear()

    def progress(event: dict) -> None:
        entry = {key: value for key, value in event.items() if key != "result"}
        with STATE_LOCK:
            if STATE["job_id"] != job_id:
                return
            STATE["event_seq"] += 1
            STATE["events"].append({**entry, "seq": STATE["event_seq"]})
            STATE["events"] = STATE["events"][-400:]

    def worker() -> None:
        try:
            result = run(settings, progress, CANCEL_EVENT, prepared_plan=plan)
            with STATE_LOCK:
                if STATE["job_id"] == job_id:
                    STATE.update(status="complete", result=result, finished_at=time.time())
        except ProcessingCancelled as exc:
            with STATE_LOCK:
                if STATE["job_id"] == job_id:
                    STATE.update(status="cancelled", error=str(exc), finished_at=time.time())
        except Exception as exc:  # noqa: BLE001 - surface worker failures in the UI
            with STATE_LOCK:
                if STATE["job_id"] == job_id:
                    STATE.update(
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                        finished_at=time.time(),
                    )

    threading.Thread(target=worker, name=f"data-processor-{job_id}", daemon=True).start()
    return job_id


class Handler(BaseHTTPRequestHandler):
    server_version = "DataProcessor/2.0"

    def send_payload(
        self,
        payload,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str = "application/json; charset=utf-8",
        cache: str = "no-store",
    ) -> None:
        if isinstance(payload, bytes):
            body = payload
        elif isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("请求体为空或超过 1 MB")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return value

    def api_error(self, exc: Exception, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self.send_payload({"error": f"{type(exc).__name__}: {exc}"}, status)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = urlparse(self.path).path
        if route == "/api/health":
            self.send_payload({"status": "ok", "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == ""})
            return
        if route == "/api/defaults":
            self.send_payload(
                {
                    "project_root": str(PROJECT_ROOT),
                    "input_dir": DEFAULT_INPUT_DIR,
                    "output_root": DEFAULT_OUTPUT_ROOT,
                    "repo_id": DEFAULT_REPO_ID,
                    "task": TASK_DESCRIPTION,
                    "source_fps": DEFAULT_SOURCE_FPS,
                    "target_fps": DEFAULT_FPS,
                    "action_mode": DEFAULT_ACTION_MODE,
                    "gap_policy": DEFAULT_GAP_POLICY,
                    "limit": DEFAULT_LIMIT,
                    "action_modes": list(ACTION_MODES),
                    "gap_policies": list(GAP_POLICIES),
                }
            )
            return
        if route == "/api/status":
            self.send_payload(state_snapshot())
            return
        if route in ("/", "/index.html"):
            self.send_payload(
                (STATIC_ROOT / "index.html").read_bytes(),
                content_type="text/html; charset=utf-8",
                cache="no-cache",
            )
            return
        static_files = {
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        if route in static_files:
            name, content_type = static_files[route]
            self.send_payload(
                (STATIC_ROOT / name).read_bytes(), content_type=content_type, cache="no-cache"
            )
            return
        self.send_payload({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = urlparse(self.path).path
        try:
            if route == "/api/cancel":
                with STATE_LOCK:
                    if STATE["status"] != "running":
                        raise RuntimeError("当前没有正在运行的转换")
                    CANCEL_EVENT.set()
                self.send_payload({"status": "cancelling"}, HTTPStatus.ACCEPTED)
                return
            payload = self.read_json()
            if route == "/api/inspect":
                self.send_payload(scan(settings_from(payload)))
                return
            if route == "/api/start":
                settings = settings_from(payload)
                plan = scan(settings)
                job_id = start_job(settings, plan)
                self.send_payload(
                    {"status": "started", "job_id": job_id, "plan": plan},
                    HTTPStatus.ACCEPTED,
                )
                return
            if route == "/api/validate":
                path = payload.get("path")
                if not path:
                    raise ValueError("缺少待验收数据集路径")
                self.send_payload(validate_lerobot_bundle(path))
                return
            self.send_payload({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except RuntimeError as exc:
            self.api_error(exc, HTTPStatus.CONFLICT)
        except Exception as exc:  # noqa: BLE001 - return validation details
            self.api_error(exc)

    def log_message(self, _format: str, *_args) -> None:
        return


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_server(host: str = "127.0.0.1", port: int = 8766) -> None:
    server = Server((host, port), Handler)
    actual_port = server.server_address[1]
    print(f"DataProcessor: http://{host}:{actual_port}")
    print("CPU-only conversion service; press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("DataProcessor stopped")
