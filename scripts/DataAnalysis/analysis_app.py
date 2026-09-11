"""Read-only web service for Piper HDF5 and LeRobot quality analysis."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
STATIC_ROOT = ROOT / "static"
DEFAULT_HDF5 = "datasets/hdf5/撕下快递单并贴到桌子上的快递上"
DEFAULT_LEROBOT_ROOT = PROJECT_ROOT / "datasets/lerobot"
LOCK = threading.Lock()
CANCEL = threading.Event()
PROCESS: subprocess.Popen | None = None
STATE = {
    "job_id": 0,
    "status": "idle",
    "kind": None,
    "path": None,
    "started_at": None,
    "finished_at": None,
    "elapsed": None,
    "events": [],
    "report": None,
    "error": None,
}


def default_lerobot_path() -> str:
    if DEFAULT_LEROBOT_ROOT.is_dir():
        candidates = sorted(
            DEFAULT_LEROBOT_ROOT.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True
        )
        for candidate in candidates:
            if (
                (candidate / "lerobot/meta/info.json").is_file()
                or (candidate / "meta/info.json").is_file()
            ):
                return str(candidate.relative_to(PROJECT_ROOT))
    return str(DEFAULT_LEROBOT_ROOT.relative_to(PROJECT_ROOT))


def snapshot() -> dict:
    with LOCK:
        value = copy.deepcopy(STATE)
    if value["status"] == "running" and value["started_at"]:
        value["elapsed"] = round(time.time() - value["started_at"], 1)
    return value


def add_event(job_id: int, message: str) -> None:
    with LOCK:
        if STATE["job_id"] == job_id:
            STATE["events"].append({"time": time.time(), "message": message})
            STATE["events"] = STATE["events"][-100:]


def start_analysis(payload: dict) -> int:
    kind = str(payload.get("kind") or "")
    target = str(payload.get("path") or "").strip()
    if kind not in ("hdf5", "lerobot"):
        raise ValueError("kind 必须为 hdf5 或 lerobot")
    if not target:
        raise ValueError("请填写待分析路径")
    target_path = Path(target).expanduser()
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path
    if not target_path.exists():
        raise FileNotFoundError(target_path.resolve())
    with LOCK:
        if STATE["status"] == "running":
            raise RuntimeError("已有分析任务正在运行")
        job_id = int(STATE["job_id"]) + 1
        STATE.update(
            job_id=job_id,
            status="running",
            kind=kind,
            path=str(target_path.resolve()),
            started_at=time.time(),
            finished_at=None,
            elapsed=None,
            events=[],
            report=None,
            error=None,
        )
        CANCEL.clear()

    def worker() -> None:
        global PROCESS
        report_file = tempfile.NamedTemporaryFile(prefix="data-analysis-", suffix=".json", delete=False)
        log_file = tempfile.NamedTemporaryFile(prefix="data-analysis-", suffix=".log", delete=False)
        report_path, log_path = Path(report_file.name), Path(log_file.name)
        report_file.close()
        log_file.close()
        report_path.unlink(missing_ok=True)
        try:
            script = ROOT / ("analyze_hdf5.py" if kind == "hdf5" else "analyze_lerobot.py")
            command = [sys.executable, str(script), str(target_path), "--json", str(report_path)]
            if kind == "hdf5":
                command.extend([
                    "--expected-fps", str(float(payload.get("expected_fps") or 30)),
                    "--gap-factor", str(float(payload.get("gap_factor") or 1.5)),
                ])
            elif payload.get("source_dir"):
                command.extend(["--source-dir", str(payload["source_dir"])])
            add_event(job_id, f"启动 {kind.upper()} 全量质量分析")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ""
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            with log_path.open("w", encoding="utf-8") as output:
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdout=output, stderr=subprocess.STDOUT,
                    text=True, env=environment,
                )
                with LOCK:
                    if STATE["job_id"] == job_id:
                        PROCESS = process
                while process.poll() is None:
                    if CANCEL.wait(0.2):
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise InterruptedError("用户取消了分析")
            if not report_path.is_file():
                details = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"分析器未生成报告，退出码 {process.returncode}\n{details}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            add_event(job_id, f"分析完成：{report.get('status', 'UNKNOWN')}")
            with LOCK:
                if STATE["job_id"] == job_id:
                    now = time.time()
                    STATE.update(
                        status="complete", report=report, finished_at=now,
                        elapsed=round(now - STATE["started_at"], 1),
                    )
        except InterruptedError as exc:
            with LOCK:
                if STATE["job_id"] == job_id:
                    now = time.time()
                    STATE.update(status="cancelled", error=str(exc), finished_at=now,
                                 elapsed=round(now - STATE["started_at"], 1))
        except Exception as exc:  # noqa: BLE001 - surface analyzer failures
            with LOCK:
                if STATE["job_id"] == job_id:
                    now = time.time()
                    STATE.update(
                        status="failed", error=f"{type(exc).__name__}: {exc}",
                        finished_at=now, elapsed=round(now - STATE["started_at"], 1),
                        traceback=traceback.format_exc(),
                    )
        finally:
            with LOCK:
                if STATE["job_id"] == job_id:
                    PROCESS = None
            report_path.unlink(missing_ok=True)
            log_path.unlink(missing_ok=True)

    threading.Thread(target=worker, name=f"data-analysis-{job_id}", daemon=True).start()
    return job_id


class Handler(BaseHTTPRequestHandler):
    server_version = "DataAnalysis/1.0"

    def send(self, payload, status=HTTPStatus.OK, content_type="application/json; charset=utf-8") -> None:
        if isinstance(payload, bytes):
            body = payload
        else:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("请求体为空或过大")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("请求体必须为 JSON 对象")
        return value

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/health":
            self.send({"status": "ok", "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == ""})
        elif route == "/api/defaults":
            self.send({
                "project_root": str(PROJECT_ROOT), "hdf5_path": DEFAULT_HDF5,
                "lerobot_path": default_lerobot_path(), "expected_fps": 30, "gap_factor": 1.5,
            })
        elif route == "/api/status":
            self.send(snapshot())
        elif route in ("/", "/index.html"):
            self.send((STATIC_ROOT / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
        elif route == "/app.js":
            self.send((STATIC_ROOT / "app.js").read_bytes(), content_type="text/javascript; charset=utf-8")
        elif route == "/styles.css":
            self.send((STATIC_ROOT / "styles.css").read_bytes(), content_type="text/css; charset=utf-8")
        else:
            self.send({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route == "/api/start":
                job_id = start_analysis(self.json_body())
                self.send({"status": "started", "job_id": job_id}, HTTPStatus.ACCEPTED)
            elif route == "/api/cancel":
                with LOCK:
                    if STATE["status"] != "running":
                        raise RuntimeError("当前没有正在运行的分析")
                    CANCEL.set()
                self.send({"status": "cancelling"}, HTTPStatus.ACCEPTED)
            else:
                self.send({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except RuntimeError as exc:
            self.send({"error": f"RuntimeError: {exc}"}, HTTPStatus.CONFLICT)
        except Exception as exc:  # noqa: BLE001
            self.send({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)

    def log_message(self, _format: str, *_args) -> None:
        return


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_server(host: str, port: int) -> None:
    server = Server((host, port), Handler)
    print(f"DataAnalysis: http://{host}:{server.server_address[1]}")
    print("Read-only CPU analysis; press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with LOCK:
            process = PROCESS
        if process and process.poll() is None:
            process.terminate()
        server.server_close()
        print("DataAnalysis stopped")
