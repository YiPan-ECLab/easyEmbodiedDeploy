#!/usr/bin/env python3
"""Dependency-free Web console and process supervisor for LerobotTrainer."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from trainer import (
    DEFAULT_DATASET,
    DEFAULT_PRETRAINED,
    PROJECT_ROOT,
    TrainerConfigError,
    TrainingConfig,
    build_command,
    build_environment,
    config_for_json,
    config_from_mapping,
    validate_dataset,
    validation_contract,
    wrapper_command,
)


MODULE_DIR = Path(__file__).resolve().parent
STATIC_DIR = MODULE_DIR / "web"
DEFAULT_PORT = 8891
ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
PROGRESS_RE = re.compile(
    r"Training:.*?(\d[\d,]*)/(\d[\d,]*).*?\[[^<]*<([^,]*),\s*([\d.]+)s/step"
)
METRIC_RE = re.compile(r"\b(loss|grdn|lr|step_s|mem_gb|epch):([0-9.eE+-]+)")
STEP_RE = re.compile(r"\bstep:(\d[\d,]*)")


def _default_output() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return str(PROJECT_ROOT / f"models/finetuned_models/pi05_trainer_{stamp}")


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_root": str(DEFAULT_DATASET),
    "repo_id": "",
    "pretrained": str(DEFAULT_PRETRAINED),
    "output_dir": _default_output(),
    "job_name": "pi05_piper_lora",
    "action_mode": "delta_joint",
    "normalization_mode": "quantiles",
    "finetune_mode": "lora",
    "training_scope": "expert_only",
    "rtc_mode": "normal",
    "rtc_max_delay": 5,
    "lora_rank": 16,
    "lora_alpha": 16,
    "batch_size": 8,
    "steps": 30_000,
    "save_freq": 5_000,
    "log_freq": 200,
    "num_workers": 4,
    "seed": 1000,
    "n_action_steps": 10,
    "dtype": "bfloat16",
    "gradient_checkpointing": True,
    "save_checkpoint": True,
    "device": "cuda",
    "gpu_ids": "0",
    "wandb_enabled": True,
    "wandb_project": "piper-pi05",
    "wandb_entity": "",
    "wandb_mode": "online",
    "resume_config": "",
    "dry_run": False,
}


class ConsoleError(RuntimeError):
    """Error safe to return through the Web API."""


def normalize_web_config(raw: dict[str, Any] | None, *, check_output: bool) -> TrainingConfig:
    values = DEFAULT_CONFIG.copy()
    if raw:
        values.update({key: value for key, value in raw.items() if key in values})
    try:
        return config_from_mapping(values, check_output=check_output)
    except TrainerConfigError as exc:
        raise ConsoleError(str(exc)) from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def discover_datasets() -> list[dict[str, Any]]:
    base = PROJECT_ROOT / "datasets/lerobot"
    result: list[dict[str, Any]] = []
    if not base.is_dir():
        return result
    for info_path in sorted(base.glob("*/lerobot/meta/info.json")):
        root = info_path.parents[1]
        info = _read_json(info_path)
        action_mode = None
        processing = _read_json(root.parent / "processing.json")
        raw_mode = processing.get("settings", {}).get("action_mode")
        if raw_mode:
            action_mode = "absolute_joint" if raw_mode == "absolute" else raw_mode
        result.append(
            {
                "path": str(root.parent),
                "resolved_root": str(root),
                "label": root.parent.name,
                "fps": info.get("fps"),
                "frames": info.get("total_frames"),
                "episodes": info.get("total_episodes"),
                "action_mode": action_mode,
            }
        )
    return result


def discover_models() -> list[dict[str, Any]]:
    candidates = [
        *(PROJECT_ROOT / "models/base_models").glob("*/config.json"),
        *(PROJECT_ROOT / "models/finetuned_models").glob(
            "**/pretrained_model/config.json"
        ),
    ]
    result: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for config_path in sorted(candidates):
        directory = config_path.parent.resolve()
        if directory in seen:
            continue
        config = _read_json(config_path)
        if config.get("type") != "pi05":
            continue
        seen.add(directory)
        result.append(
            {
                "path": str(directory),
                "label": str(directory.relative_to(PROJECT_ROOT / "models")),
                "kind": "LoRA" if (directory / "adapter_config.json").is_file() else "Full",
                "chunk_size": config.get("chunk_size", 50),
                "rtc_training_max_delay": config.get("rtc_training_max_delay", 0),
            }
        )
    return result


def discover_resume_configs() -> list[dict[str, Any]]:
    base = PROJECT_ROOT / "models/finetuned_models"
    result: list[dict[str, Any]] = []
    if not base.is_dir():
        return result
    for config_path in sorted(base.glob("**/checkpoints/*/pretrained_model/train_config.json")):
        config = _read_json(config_path)
        state_path = config_path.parents[1] / "training_state/training_step.json"
        state = _read_json(state_path)
        result.append(
            {
                "path": str(config_path),
                "label": str(config_path.relative_to(base)),
                "step": state.get("step"),
                "output_dir": config.get("output_dir"),
                "job_name": config.get("job_name"),
            }
        )
    return result


def gpu_status() -> list[dict[str, Any]]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(query, check=True, capture_output=True, text=True, timeout=3)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    gpus: list[dict[str, Any]] = []
    by_uuid: dict[str, dict[str, Any]] = {}
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 7:
            continue
        item = {
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "memory_used_mb": int(row[3]),
            "memory_total_mb": int(row[4]),
            "utilization": int(row[5]),
            "temperature": int(row[6]),
            "processes": [],
        }
        gpus.append(item)
        by_uuid[item["uuid"]] = item
    try:
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        for row in csv.reader(processes.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 4 or row[0] not in by_uuid:
                continue
            by_uuid[row[0]]["processes"].append(
                {"pid": int(row[1]), "name": Path(row[2]).name, "memory_mb": int(row[3])}
            )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return gpus


def external_training_processes(excluded_pgid: int | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            argv = [value.decode(errors="replace") for value in args if value]
            if excluded_pgid is not None and os.getpgid(pid) == excluded_pgid:
                continue
        except (OSError, ProcessLookupError):
            continue
        is_train = any(Path(arg).name == "lerobot-train" for arg in argv) or any(
            "lerobot.scripts.lerobot_train" in arg for arg in argv
        )
        if is_train:
            result.append({"pid": pid, "command": shlex.join(argv)})
    return sorted(result, key=lambda item: item["pid"])


@dataclass
class ManagedTraining:
    process: subprocess.Popen[str]
    config: TrainingConfig
    command: list[str]
    started_at: float
    state: str = "validating"
    exit_code: int | None = None
    step: int = 0
    total_steps: int = 0
    eta: str = "—"
    seconds_per_step: float | None = None
    last_metrics: dict[str, float] = field(default_factory=dict)
    metrics: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=1200))
    dry_run: bool = False


class TrainingSupervisor:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._training: ManagedTraining | None = None
        self._starting = False
        self._logs: deque[dict[str, Any]] = deque(maxlen=5000)
        self._next_log_id = 1
        self._dataset_report: dict[str, Any] | None = None

    def log(self, source: str, message: str, level: str = "info") -> None:
        with self._lock:
            self._logs.append(
                {
                    "id": self._next_log_id,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "source": source,
                    "level": level,
                    "message": message,
                }
            )
            self._next_log_id += 1

    def logs_after(self, after: int) -> dict[str, Any]:
        with self._lock:
            items = [item for item in self._logs if item["id"] > after]
            return {"items": items, "last_id": self._next_log_id - 1}

    def clear_logs(self) -> None:
        with self._lock:
            self._logs.clear()

    def _is_running(self) -> bool:
        return self._training is not None and self._training.process.poll() is None

    def _assert_gpu_available(self, config: TrainingConfig) -> None:
        if config.device != "cuda":
            return
        by_index = {gpu["index"]: gpu for gpu in gpu_status()}
        conflicts: list[str] = []
        for index in config.gpu_ids:
            processes = by_index.get(index, {}).get("processes", [])
            if processes:
                details = ", ".join(f"PID {item['pid']} ({item['memory_mb']} MiB)" for item in processes)
                conflicts.append(f"GPU {index}: {details}")
        if conflicts:
            raise ConsoleError("所选 GPU 已有计算进程，已阻止启动：" + "; ".join(conflicts))

    def start(self, raw: dict[str, Any]) -> None:
        with self._lock:
            if self._is_running() or self._starting:
                raise ConsoleError("本控制台已有训练在运行")
            self._starting = True
        try:
            config = normalize_web_config(raw, check_output=True)
            dataset_root, normalization_mode, action_mode = validation_contract(config)
            report = validate_dataset(
                dataset_root,
                normalization_mode=normalization_mode,
                action_mode=action_mode,
                decode_sample=False,
            )
            self._assert_gpu_available(config)
            dry_run = bool(raw.get("dry_run", False))
            command = wrapper_command(config, dry_run=dry_run)
            environment = build_environment(config)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            total_steps = config.steps
            if config.resume_config is not None:
                total_steps = int(_read_json(config.resume_config).get("steps", total_steps))
            managed = ManagedTraining(
                process=process,
                config=config,
                command=command,
                started_at=time.time(),
                total_steps=total_steps,
                dry_run=dry_run,
            )
            with self._lock:
                self._training = managed
                self._dataset_report = report
                self._logs.clear()
        finally:
            with self._lock:
                self._starting = False
        self.log("system", f"启动 PID {process.pid}；物理 GPU {list(config.gpu_ids)}")
        threading.Thread(target=self._read_process, args=(managed,), daemon=True).start()

    def _parse_training_line(self, managed: ManagedTraining, line: str) -> None:
        if line.startswith("LEROBOT_TRAINER_PLAN="):
            managed.state = "launching"
        elif "Creating dataset" in line or "Creating policy" in line or "Loading" in line:
            managed.state = "loading"
        elif "Checkpoint policy after step" in line:
            managed.state = "saving"
        elif "Training:" in line or "Start offline training" in line:
            managed.state = "running"

        progress = PROGRESS_RE.search(line)
        if progress:
            managed.step = int(progress.group(1).replace(",", ""))
            managed.total_steps = int(progress.group(2).replace(",", ""))
            managed.eta = progress.group(3).strip() or "—"
            managed.seconds_per_step = float(progress.group(4))

        step_match = STEP_RE.search(line)
        metrics = {key: float(value) for key, value in METRIC_RE.findall(line)}
        if step_match:
            managed.step = int(step_match.group(1).replace(",", ""))
        if metrics:
            managed.last_metrics.update(metrics)
            managed.metrics.append({"step": managed.step, "time": time.time(), **metrics})

    def _read_process(self, managed: ManagedTraining) -> None:
        assert managed.process.stdout is not None
        for raw_line in managed.process.stdout:
            line = ANSI_ESCAPE.sub("", raw_line).replace("\r", "").rstrip("\n")
            if not line:
                continue
            with self._lock:
                self._parse_training_line(managed, line)
            level = "error" if any(word in line.lower() for word in ("error", "traceback", "out of memory")) else "info"
            self.log("train", line, level)
        exit_code = managed.process.wait()
        with self._lock:
            managed.exit_code = exit_code
            if managed.state == "stopping":
                managed.state = "stopped"
            elif exit_code == 0:
                managed.state = "completed"
                if managed.step == 0 and managed.config.steps and not managed.dry_run:
                    managed.step = managed.config.steps
            else:
                managed.state = "failed"
        self.log("system", f"训练进程退出，code={exit_code}", "error" if exit_code else "info")

    def stop(self) -> None:
        with self._lock:
            managed = self._training
            if managed is None or managed.process.poll() is not None:
                return
            managed.state = "stopping"
        self.log("system", "发送 SIGINT，等待训练进程退出")

        def terminate() -> None:
            try:
                os.killpg(managed.process.pid, signal.SIGINT)
            except ProcessLookupError:
                return
            try:
                managed.process.wait(timeout=20)
                return
            except subprocess.TimeoutExpired:
                self.log("system", "SIGINT 超时，发送 SIGTERM", "error")
            try:
                os.killpg(managed.process.pid, signal.SIGTERM)
                managed.process.wait(timeout=8)
                return
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            try:
                os.killpg(managed.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        threading.Thread(target=terminate, daemon=True).start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            managed = self._training
            dataset_report = self._dataset_report
            if managed is None:
                training = {
                    "state": "idle",
                    "pid": None,
                    "step": 0,
                    "total_steps": 0,
                    "exit_code": None,
                    "eta": "—",
                    "elapsed_s": 0,
                    "seconds_per_step": None,
                    "metrics": {},
                    "history": [],
                    "gpu_ids": [],
                    "command": None,
                    "dry_run": False,
                }
                pgid = None
            else:
                pgid = managed.process.pid if managed.process.poll() is None else None
                training = {
                    "state": managed.state,
                    "pid": managed.process.pid,
                    "step": managed.step,
                    "total_steps": managed.total_steps,
                    "exit_code": managed.exit_code,
                    "eta": managed.eta,
                    "elapsed_s": max(0, time.time() - managed.started_at),
                    "seconds_per_step": managed.seconds_per_step,
                    "metrics": dict(managed.last_metrics),
                    "history": list(managed.metrics),
                    "gpu_ids": list(managed.config.gpu_ids),
                    "command": shlex.join(managed.command),
                    "dry_run": managed.dry_run,
                }
        return {
            "training": training,
            "dataset": dataset_report,
            "gpu": gpu_status(),
            "external_training": external_training_processes(pgid),
        }

    def shutdown(self) -> None:
        if self._is_running():
            self.stop()
            deadline = time.time() + 30
            while self._is_running() and time.time() < deadline:
                time.sleep(0.1)


class ConsoleHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], supervisor: TrainingSupervisor):
        self.supervisor = supervisor
        super().__init__(address, ConsoleHandler)


class ConsoleHandler(BaseHTTPRequestHandler):
    server: ConsoleHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            raise ConsoleError("请求 JSON 无效") from exc
        if not isinstance(payload, dict):
            raise ConsoleError("请求体必须是 JSON object")
        return payload

    def _static(self, path: str) -> None:
        relative = "index.html" if path == "/" else path.lstrip("/")
        candidate = (STATIC_DIR / relative).resolve()
        if STATIC_DIR not in candidate.parents or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'",
        )
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/bootstrap":
                self._json(
                    {
                        "defaults": DEFAULT_CONFIG,
                        "datasets": discover_datasets(),
                        "models": discover_models(),
                        "resume_configs": discover_resume_configs(),
                        "status": self.server.supervisor.status(),
                    }
                )
            elif parsed.path == "/api/status":
                self._json(self.server.supervisor.status())
            elif parsed.path == "/api/logs":
                after = int(parse_qs(parsed.query).get("after", ["0"])[0])
                self._json(self.server.supervisor.logs_after(after))
            elif parsed.path.startswith("/api/"):
                self._json({"error": "API 不存在"}, HTTPStatus.NOT_FOUND)
            else:
                self._static(parsed.path)
        except (ConsoleError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json({"error": f"服务器错误：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._body()
            if self.path == "/api/preview":
                config = normalize_web_config(body.get("config"), check_output=False)
                dataset_root, normalization_mode, action_mode = validation_contract(config)
                report = validate_dataset(
                    dataset_root,
                    normalization_mode=normalization_mode,
                    action_mode=action_mode,
                    decode_sample=False,
                )
                dry_run = bool((body.get("config") or {}).get("dry_run", False))
                self._json(
                    {
                        "ok": True,
                        "config": config_for_json(config),
                        "dataset": report,
                        "wrapper": shlex.join(wrapper_command(config, dry_run=dry_run)),
                        "lerobot": shlex.join(build_command(config)),
                        "cuda_visible_devices": build_environment(config)["CUDA_VISIBLE_DEVICES"],
                    }
                )
            elif self.path == "/api/validate":
                config = normalize_web_config(body.get("config"), check_output=False)
                command = wrapper_command(config, validate_only=True)
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=build_environment(config),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    raise ConsoleError((result.stderr or result.stdout).strip()[-3000:])
                marker = next(
                    (line for line in result.stdout.splitlines() if line.startswith("LEROBOT_TRAINER_VALIDATION=")),
                    None,
                )
                if marker is None:
                    raise ConsoleError("验证进程没有返回结构化结果")
                self._json({"ok": True, "dataset": json.loads(marker.split("=", 1)[1])})
            elif self.path == "/api/start":
                config = body.get("config")
                if not isinstance(config, dict):
                    raise ConsoleError("缺少训练配置")
                self.server.supervisor.start(config)
                self._json({"ok": True}, HTTPStatus.ACCEPTED)
            elif self.path == "/api/stop":
                self.server.supervisor.stop()
                self._json({"ok": True}, HTTPStatus.ACCEPTED)
            elif self.path == "/api/logs/clear":
                self.server.supervisor.clear_logs()
                self._json({"ok": True})
            else:
                self._json({"error": "API 不存在"}, HTTPStatus.NOT_FOUND)
        except ConsoleError as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except subprocess.TimeoutExpired:
            self._json({"error": "数据集首帧验证超时"}, HTTPStatus.REQUEST_TIMEOUT)
        except Exception as exc:
            self._json({"error": f"服务器错误：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    supervisor = TrainingSupervisor()
    server = ConsoleHTTPServer((args.host, args.port), supervisor)
    print(f"LeRobot Trainer: http://{args.host}:{args.port}", flush=True)
    print("Ctrl-C 会停止本页面启动的训练；不会触碰旧 Trainer 或外部训练。", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n正在关闭 LeRobot Trainer…", flush=True)
    finally:
        supervisor.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
