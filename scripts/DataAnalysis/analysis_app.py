"""Web service for Piper HDF5 and LeRobot quality analysis."""

from __future__ import annotations

import copy
import contextlib
import json
import math
import os
import shutil
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from easy_embodied_deploy.http import LocalHandlerMixin, read_json

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
STATIC_ROOT = ROOT / "static"
DEFAULT_HDF5 = "datasets/hdf5/撕下快递单并贴到桌子上的快递上"
DEFAULT_LEROBOT_ROOT = PROJECT_ROOT / "datasets/lerobot"
LOCK = threading.Lock()
CANCEL = threading.Event()
PROCESS: subprocess.Popen | None = None
WORKER: threading.Thread | None = None
DELETE_LOCK = threading.Lock()
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
    global WORKER
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
    if kind == "hdf5":
        for key, default, minimum in (("expected_fps", 30, 0), ("gap_factor", 1.5, 1)):
            value = float(payload.get(key, default))
            if not math.isfinite(value) or value <= minimum:
                raise ValueError(f"{key} 必须是大于 {minimum} 的有限数值")
    from easy_embodied_deploy.pose_quality import PoseThresholds
    pose_thresholds = PoseThresholds(**{key: float(payload.get("pose_thresholds", {}).get(key, value))
                                       for key, value in vars(PoseThresholds()).items()})
    with LOCK:
        if STATE["status"] in ("running", "deleting"):
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
            for key, value in vars(pose_thresholds).items():
                command.extend(["--pose-" + key.replace("_", "-"), str(value)])
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

    WORKER = threading.Thread(target=worker, name=f"data-analysis-{job_id}", daemon=True)
    WORKER.start()
    return job_id


def _resolved_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _dataset_root(raw: str | Path) -> tuple[Path, Path]:
    """Return (bundle_root, LeRobot root) for either supported layout."""
    bundle = _resolved_path(raw)
    root = bundle / "lerobot" if (bundle / "lerobot/meta/info.json").is_file() else bundle
    if not (root / "meta/info.json").is_file():
        raise FileNotFoundError(f"找不到 LeRobot meta/info.json：{bundle}")
    return (root.parent if root.name == "lerobot" else root), root


def _physical_gap_episodes(report: dict) -> set[int]:
    """Select only episodes rejected for a source physical time gap."""
    selected = set()
    for item in report.get("episodes", []):
        errors = item.get("errors", [])
        if any("相邻源采样间隔" in str(error) for error in errors):
            try:
                selected.add(int(item["episode"]))
            except (KeyError, TypeError, ValueError):
                continue
    return selected


def _update_processing_manifest(bundle: Path, deleted: set[int]) -> None:
    """Keep the converter manifest aligned with the reindexed dataset."""
    manifest_path = bundle / "processing.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = manifest.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("files"), list):
        return
    files = plan["files"]
    kept = [item for index, item in enumerate(files) if index not in deleted]
    plan["files"] = kept
    plan["episodes"] = len(kept)
    plan["input_frames"] = int(sum(int(item.get("frames", 0)) for item in kept))
    plan["output_frames"] = int(sum(int(item.get("output_frames", 0)) for item in kept))
    plan["gaps"] = int(sum(int(item.get("gaps", 0)) for item in kept))
    plan["warnings"] = [
        f"{item.get('name', '?')}: {warning}"
        for item in kept for warning in item.get("warnings", [])
    ]
    manifest["deleted_episodes"] = sorted(deleted)
    manifest["episode_index_mapping"] = {
        str(old): new for new, old in enumerate(index for index in range(len(files)) if index not in deleted)
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


def delete_error_episodes(payload: dict) -> dict:
    """Delete reported physical-gap episodes with LeRobot's official editor."""
    target = str(payload.get("path") or "").strip()
    if not target:
        raise ValueError("请先完成 LeRobot 分析")
    bundle, root = _dataset_root(target)
    requested_raw = payload.get("episodes")
    if not isinstance(requested_raw, list) or not requested_raw:
        raise ValueError("episodes 必须是非空数组")
    requested = set()
    for value in requested_raw:
        if isinstance(value, bool):
            raise ValueError("episode 索引必须是整数")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("episode 索引必须是整数") from exc
        if isinstance(value, float) and value != integer:
            raise ValueError("episode 索引必须是整数")
        requested.add(integer)
    if any(value < 0 for value in requested):
        raise ValueError("episode 索引不能为负数")

    with LOCK:
        if STATE["status"] in ("running", "deleting"):
            raise RuntimeError("分析任务正在运行，请完成后再删除")
        report = copy.deepcopy(STATE.get("report"))
        report_job_id = STATE["job_id"]
        report_path = _resolved_path(report.get("bundle", "")) if report else None
    if not report or report_path != bundle:
        raise RuntimeError("请先对当前 LeRobot 路径完成一次分析")
    allowed = _physical_gap_episodes(report)
    invalid = requested - allowed
    if invalid:
        raise ValueError(f"只能删除物理时间间隔异常 episode：{sorted(invalid)}")

    with DELETE_LOCK:
        with LOCK:
            if STATE["status"] in ("running", "deleting") or STATE["job_id"] != report_job_id:
                raise RuntimeError("分析任务正在运行，请完成后再删除")
            STATE.update(status="deleting", error=None, events=[{
                "time": time.time(), "message": f"准备删除 {len(requested)} 个物理间隔异常 episode"
            }])

        # LeRobotDatasetMetadata.create requires a non-existent destination.
        temp_dir = None
        backup = root.with_name(f".{root.name}.before-delete-{os.getpid()}-{time.time_ns()}")
        manifest_path = bundle / "processing.json"
        original_manifest = None
        try:
            original_manifest = manifest_path.read_bytes() if manifest_path.is_file() else None
            temp_dir = Path(tempfile.mkdtemp(prefix=f".{root.name}.delete-", dir=root.parent))
            temp_dir.rmdir()
            from lerobot.datasets import LeRobotDataset, delete_episodes

            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            if len(requested) >= int(info["total_episodes"]):
                raise ValueError("不能删除全部 episode；请保留至少一条有效数据")
            repo_id = str(info.get("repo_id") or f"local/{root.name}")
            dataset = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)
            delete_episodes(dataset, sorted(requested), output_dir=temp_dir, repo_id=repo_id)
            del dataset
            from easy_embodied_deploy.pose_dataset import copy_retained_episodes
            copy_retained_episodes(root, temp_dir, requested)

            # Validate the generated root without consulting the old manifest.
            processor_dir = PROJECT_ROOT / "scripts" / "LerobotDataProcessor"
            sys.path.insert(0, str(processor_dir))
            try:
                from processor import validate_lerobot_bundle
                validation = validate_lerobot_bundle(temp_dir, expected={})
            finally:
                with contextlib.suppress(ValueError):
                    sys.path.remove(str(processor_dir))
            if not validation["valid"]:
                raise ValueError("删除后数据集验收失败：" + "；".join(validation["errors"]))

            root.rename(backup)
            if original_manifest is not None and bundle != root:
                (backup / "processing.json").write_bytes(original_manifest)
            temp_dir.rename(root)
            if original_manifest is not None:
                manifest_path.write_bytes(original_manifest)
            _update_processing_manifest(bundle, requested)
            # Retain the complete original. Deleting it here makes rollback
            # destructive if rmtree only partially succeeds (disk/IO failure).
            result = {"deleted": sorted(requested), "remaining": int(info["total_episodes"]) - len(requested),
                      "path": str(bundle), "backup": str(backup)}
            with LOCK:
                STATE.update(status="idle", report=None, error=None, finished_at=time.time(),
                             events=[{"time": time.time(), "message": f"删除完成，原始备份保留在 {backup}；请重新分析确认"}])
            return result
        except Exception:
            if temp_dir is not None and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            if backup.exists():
                if root.exists():
                    shutil.rmtree(root, ignore_errors=True)
                backup.rename(root)
            if original_manifest is not None:
                manifest_path.write_bytes(original_manifest)
            with LOCK:
                STATE.update(status="complete", error="删除失败", finished_at=time.time())
            raise


class Handler(LocalHandlerMixin, BaseHTTPRequestHandler):
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
        return read_json(self)

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
            elif route == "/api/delete-error-episodes":
                result = delete_error_episodes(self.json_body())
                self.send(result)
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
    print("CPU analysis with physical-gap episode deletion; press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        CANCEL.set()
        with LOCK:
            process = PROCESS
        if process and process.poll() is None:
            process.terminate()
        if WORKER:
            WORKER.join(timeout=10)
        server.server_close()
        print("DataAnalysis stopped")
