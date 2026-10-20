#!/usr/bin/env python3
"""Dependency-free local web console for OpenpiDeployer."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from easy_embodied_deploy.http import LocalHandlerMixin, json_safe, read_json
from easy_embodied_deploy.action_contract import ACTION_MODES, POSE_MODES
from easy_embodied_deploy.contracts import boolean


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
STATIC_DIR = SCRIPT_DIR / "web"
PIPER_PYTHON = ROOT / "thirdParty/.venvs/piper-home/bin/python"
OPENPI_PYTHON = ROOT / "thirdParty/.venvs/openpi-jax/bin/python"
BRIDGE_SCRIPT = ROOT / "scripts/LerobotDeployer/hardware_bridge.py"
DEPLOYER_SCRIPT = SCRIPT_DIR / "OpenpiDeployer.py"
DEFAULT_ENDPOINT = "tcp://127.0.0.1:6007"

DEFAULT_CONFIG: dict[str, Any] = {
    "checkpoint": str(
        ROOT
        / "models/finetuned_models/openpi_pi05_lora20hz_absolute"
    ),
    "action_mode": "absolute_joint",
    "calibration_mount_id": "",
    "pose_start_margin_deg": 2.0,
    "inference_mode": "rtc",
    "rtc_mode": "trained",
    "rtc_horizon": 20,
    "rtc_queue_threshold": 40,
    "rtc_schedule": "exp",
    "rtc_max_guidance_weight": 5.0,
    "rtc_mask_grippers": True,
    "sample_steps": 10,
    "n_action_steps": 0,
    "warmup": True,
    "bridge_endpoint": DEFAULT_ENDPOINT,
    "task": "Remove the shipping label and place it on the package on the table",
    "device": "cuda:1",
    "policy_hz": 30.0,
    "control_hz": 150.0,
    "smoothing_time_constant": 0.1,
    # The web console always starts in continuous mode. Set a positive value
    # in the max_steps field when a bounded run is needed.
    "max_steps": 0,
    "seed": 20260909,
    "rpc_timeout": 5.0,
    "watchdog_timeout": 30.0,
    # Empty means allocate a persistent, unique log for this web deployment.
    "diagnostics_jsonl": "",
    "dry_run": False,
    "offline_dataset_root": "",
}

ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
ACTION_PROGRESS = re.compile(r"action\s+(\d+)(?:/(\d+))?")


class ConsoleError(RuntimeError):
    """A user-facing web console error."""


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> int | float:
    if isinstance(value, bool):
        raise ConsoleError(f"{name} 必须是数字")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConsoleError(f"{name} 必须是数字") from exc
    if not (float("-inf") < numeric < float("inf")):
        raise ConsoleError(f"{name} 必须是有限数字")
    if integer and not numeric.is_integer():
        raise ConsoleError(f"{name} 必须是整数")
    parsed: int | float = int(numeric) if integer else numeric
    if minimum is not None and parsed < minimum:
        raise ConsoleError(f"{name} 不能小于 {minimum:g}")
    if maximum is not None and parsed > maximum:
        raise ConsoleError(f"{name} 不能大于 {maximum:g}")
    return parsed


def _resolve_user_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConsoleError(f"{name} 不能为空")
    path = Path(value.strip()).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _endpoint_port(endpoint: str) -> tuple[str, int]:
    match = re.fullmatch(r"tcp://([^:/]+):(\d{1,5})", endpoint)
    if match is None:
        raise ConsoleError("桥接地址必须采用 tcp://host:port 格式")
    port = int(match.group(2))
    if not 1 <= port <= 65535:
        raise ConsoleError("桥接端口必须位于 1 到 65535")
    host = match.group(1)
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "localhost"} else host
    return connect_host, port


def port_is_open(endpoint: str) -> bool:
    try:
        host, port = _endpoint_port(endpoint)
        with socket.create_connection((host, port), timeout=0.08):
            return True
    except (ConsoleError, OSError):
        return False


def normalize_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, dict):
        raise ConsoleError("部署配置必须为对象")
    config = DEFAULT_CONFIG.copy()
    if raw:
        config.update({key: value for key, value in raw.items() if key in config})
    from runtime import OpenpiDeploymentError, load_training_manifest, resolve_checkpoint

    try:
        checkpoint = resolve_checkpoint(_resolve_user_path(config["checkpoint"], "checkpoint"))
        _manifest_path, manifest = load_training_manifest(checkpoint)
        checkpoint_config = manifest["config"]
    except OpenpiDeploymentError as exc:
        raise ConsoleError(str(exc)) from exc

    action_mode = str(config["action_mode"])
    inference_mode = str(config["inference_mode"])
    rtc_mode = str(config["rtc_mode"])
    device = str(config["device"])
    if action_mode not in ACTION_MODES:
        raise ConsoleError("action mode 无效")
    if inference_mode not in {"rtc", "sync"}:
        raise ConsoleError("inference mode 无效")
    if rtc_mode not in {"guided", "trained"}:
        raise ConsoleError("RTC mode 无效")
    rtc_schedule = str(config["rtc_schedule"])
    if rtc_schedule not in {"exp", "linear", "ones", "zeros"}:
        raise ConsoleError("RTC schedule 无效")
    if device != "cpu" and re.fullmatch(r"cuda(?::\d+)?", device) is None:
        raise ConsoleError("device 必须是 cpu、cuda 或 cuda:N（例如 cuda:0）")
    if inference_mode == "sync" and rtc_mode == "trained":
        raise ConsoleError("sync 模式不能使用 trained RTC")

    endpoint = str(config["bridge_endpoint"]).strip()
    _endpoint_port(endpoint)
    task = str(config["task"]).strip()
    if not task:
        raise ConsoleError("任务指令不能为空")

    normalized = {
        **config,
        "checkpoint": str(checkpoint),
        "pose_start_margin_deg": _number(config.get("pose_start_margin_deg", 2), "位姿初始关节偏移", minimum=0, maximum=5),
        "action_mode": action_mode,
        "inference_mode": inference_mode,
        "rtc_mode": rtc_mode,
        "rtc_schedule": rtc_schedule,
        "device": device,
        "bridge_endpoint": endpoint,
        "task": task,
        "rtc_horizon": _number(
            config["rtc_horizon"], "RTC horizon", minimum=1, maximum=50, integer=True
        ),
        "rtc_queue_threshold": _number(
            config["rtc_queue_threshold"],
            "RTC queue threshold",
            minimum=0,
            maximum=49,
            integer=True,
        ),
        "rtc_max_guidance_weight": _number(
            config["rtc_max_guidance_weight"], "RTC 最大引导权重", minimum=0
        ),
        "sample_steps": _number(config["sample_steps"], "采样步数", minimum=1, integer=True),
        "n_action_steps": _number(
            config["n_action_steps"], "同步复用步数", minimum=0, maximum=50, integer=True
        ),
        "policy_hz": _number(
            config["policy_hz"], "Policy Hz", minimum=0.001, maximum=200
        ),
        "control_hz": _number(
            config["control_hz"], "Control Hz", minimum=1, maximum=200
        ),
        "smoothing_time_constant": _number(
            config["smoothing_time_constant"],
            "滤波时间常数",
            minimum=0,
            maximum=2,
        ),
        "max_steps": _number(
            config["max_steps"], "最大步数", minimum=0, integer=True
        ),
        "seed": _number(config["seed"], "随机种子", minimum=0, integer=True),
        "rpc_timeout": _number(config["rpc_timeout"], "RPC 超时", minimum=0.1),
        "watchdog_timeout": _number(
            config["watchdog_timeout"], "看门狗超时", minimum=5
        ),
        "dry_run": boolean(config.get("dry_run", False), "dry_run", ConsoleError),
        "rtc_mask_grippers": boolean(config.get("rtc_mask_grippers", True), "rtc_mask_grippers", ConsoleError),
        "warmup": boolean(config.get("warmup", True), "warmup", ConsoleError),
    }

    diagnostics = str(config.get("diagnostics_jsonl", "")).strip()
    normalized["diagnostics_jsonl"] = (
        str(_resolve_user_path(diagnostics, "诊断文件")) if diagnostics else ""
    )

    if checkpoint_config.get("action_mode") != action_mode:
        raise ConsoleError(
            f"checkpoint 动作模式为 {checkpoint_config.get('action_mode')}，页面选择的是 {action_mode}"
        )
    chunk_size = int(checkpoint_config.get("action_horizon", 50) or 50)
    threshold = int(normalized["rtc_queue_threshold"])
    if inference_mode == "rtc" and chunk_size and threshold >= chunk_size:
        raise ConsoleError(
            f"RTC queue threshold 必须小于 checkpoint chunk_size={chunk_size}"
        )
    if rtc_mode == "trained" and int(
        checkpoint_config.get("rtc_max_delay", 0) or 0
    ) <= 0:
        raise ConsoleError("该 checkpoint 没有 trained RTC 能力，请使用 guided")
    trained_delay = int(checkpoint_config.get("rtc_max_delay", 0) or 0)
    if rtc_mode == "trained" and (
        int(normalized["rtc_horizon"]) < trained_delay
        or threshold < trained_delay
        or int(normalized["rtc_horizon"]) > chunk_size - trained_delay
    ):
        raise ConsoleError(
            f"trained RTC 要求 horizon/threshold >= {trained_delay}，且 horizon <= {chunk_size - trained_delay}"
        )
    offline = str(config.get("offline_dataset_root", "")).strip()
    if offline:
        dataset = _resolve_user_path(offline, "离线数据集")
        if not (dataset / "meta/info.json").is_file() and (dataset / "lerobot").is_dir():
            dataset = dataset / "lerobot"
        if not (dataset / "meta/info.json").is_file():
            raise ConsoleError(f"离线数据集缺少 meta/info.json：{dataset}")
        normalized["offline_dataset_root"] = str(dataset)
    try:
        if action_mode in POSE_MODES:
            from easy_embodied_deploy.target_runtime import checkpoint_pose_contract
            checkpoint_pose_contract(checkpoint, action_mode,
                mount_id=str(config.get("calibration_mount_id", "")).strip(), policy_hz=normalized["policy_hz"])
    except (ValueError, OSError, KeyError) as exc:
        raise ConsoleError(str(exc)) from exc
    return normalized


def bridge_command(endpoint: str) -> tuple[list[str], dict[str, str]]:
    _endpoint_port(endpoint)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "scripts/PiperRecorder")
    env["PYTHONUNBUFFERED"] = "1"
    return (
        [str(PIPER_PYTHON), str(BRIDGE_SCRIPT), "--endpoint", endpoint],
        env,
    )


def deployment_command(config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    command = [
        str(OPENPI_PYTHON),
        str(DEPLOYER_SCRIPT),
        "--checkpoint",
        config["checkpoint"],
        "--action-mode",
        config["action_mode"],
        "--inference-mode",
        config["inference_mode"],
        "--rtc-mode",
        config["rtc_mode"],
        "--rtc-horizon",
        str(config["rtc_horizon"]),
        "--rtc-queue-threshold",
        str(config["rtc_queue_threshold"]),
        "--rtc-schedule",
        config["rtc_schedule"],
        "--rtc-max-guidance-weight",
        str(config["rtc_max_guidance_weight"]),
        "--sample-steps",
        str(config["sample_steps"]),
        "--n-action-steps",
        str(config["n_action_steps"]),
        "--bridge-endpoint",
        config["bridge_endpoint"],
        "--task",
        config["task"],
        "--device",
        config["device"],
        "--policy-hz",
        str(config["policy_hz"]),
        "--control-hz",
        str(config["control_hz"]),
        "--smoothing-time-constant",
        str(config["smoothing_time_constant"]),
        "--max-steps",
        str(config["max_steps"]),
        "--seed",
        str(config["seed"]),
        "--rpc-timeout",
        str(config["rpc_timeout"]),
        "--watchdog-timeout",
        str(config["watchdog_timeout"]),
    ]
    command.extend(["--pose-start-margin-deg", str(config.get("pose_start_margin_deg", 2))])
    command.extend(["--calibration-mount-id", str(config.get("calibration_mount_id", "")).strip()])
    if config["diagnostics_jsonl"]:
        command.extend(["--diagnostics-jsonl", config["diagnostics_jsonl"]])
    if config.get("offline_dataset_root"):
        command.extend(["--offline-dataset-root", config["offline_dataset_root"]])
    if config["dry_run"]:
        command.append("--dry-run")
    command.append("--rtc-mask-grippers" if config["rtc_mask_grippers"] else "--no-rtc-mask-grippers")
    command.append("--warmup" if config["warmup"] else "--no-warmup")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPT_DIR)
    env["PYTHONUNBUFFERED"] = "1"
    return command, env


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[str]
    command: list[str]
    started_at: float
    state: str = "starting"
    exit_code: int | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    progress_current: int = 0
    progress_total: int = 0

    @property
    def running(self) -> bool:
        return self.process.poll() is None


class ProcessSupervisor:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._processes: dict[str, ManagedProcess] = {}
        self._logs: deque[dict[str, Any]] = deque(maxlen=5000)
        self._next_log_id = 1
        self._operation = "idle"
        self._operation_error: str | None = None
        self._orchestrator: threading.Thread | None = None
        self._diagnostics_path: Path | None = None
        self._runtime_summary: dict[str, Any] | None = None
        self._gpu_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])
        self._gpu_error: str | None = None
        self._last_endpoint = DEFAULT_ENDPOINT
        self._generation = 0
        self._closed = False
        self._stop_thread = None

    def log(self, source: str, message: str, level: str = "info") -> None:
        clean = ANSI_ESCAPE.sub("", message).replace("\r", "").rstrip("\n")
        if not clean:
            return
        with self._lock:
            self._logs.append(
                {
                    "id": self._next_log_id,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "source": source,
                    "level": level,
                    "message": clean,
                }
            )
            self._next_log_id += 1

    def _spawn(
        self, name: str, command: list[str], env: dict[str, str]
    ) -> ManagedProcess:
        with self._lock:
            if self._closed or self._operation == "stopping":
                raise ConsoleError("控制台正在停止，不能启动新进程")
            existing = self._processes.get(name)
            if existing is not None and existing.running:
                raise ConsoleError(f"{name} 已经在运行")
            self.log("system", f"启动 {name}: {shlex.join(command)}")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            managed = ManagedProcess(name, process, command, time.time())
            self._processes[name] = managed
            threading.Thread(
                target=self._read_process,
                args=(managed,),
                daemon=True,
                name=f"{name}-log-reader",
            ).start()
            return managed

    def _read_process(self, managed: ManagedProcess) -> None:
        assert managed.process.stdout is not None
        for raw_line in managed.process.stdout:
            line = ANSI_ESCAPE.sub("", raw_line).replace("\r", "").rstrip("\n")
            if not line:
                continue
            self.log(managed.name, line)
            with self._lock:
                if managed.state == "stopping" and "Returning followers and grippers to ZERO" not in line:
                    continue
                if managed.name == "bridge" and "Returning followers and grippers to ZERO" in line:
                    managed.state = "homing"
                elif managed.name == "bridge" and "hardware bridge ready:" in line:
                    managed.state = "ready"
                    managed.ready.set()
                elif managed.name == "deployer":
                    if "OPENPI_DEPLOYER_PLAN=" in line or "policy loaded" in line:
                        managed.state = "loading"
                    elif "Compiling OpenPI sampling paths" in line:
                        managed.state = "prefilling"
                    elif line.startswith("action "):
                        managed.state = "running"
                        match = ACTION_PROGRESS.search(line)
                        if match:
                            managed.progress_current = int(match.group(1))
                            managed.progress_total = int(match.group(2) or 0)
                    elif line.startswith("OPENPI_DEPLOYER_SUMMARY="):
                        try:
                            self._runtime_summary = json.loads(line.split("=", 1)[1])
                        except ValueError:
                            pass
        managed.process.stdout.close()
        exit_code = managed.process.wait()
        with self._lock:
            managed.exit_code = exit_code
            if managed.state == "stopping":
                managed.state = "stopped"
            elif exit_code == 0:
                managed.state = "completed"
            else:
                managed.state = "failed"
            managed.ready.set()
        level = "success" if exit_code == 0 else "error"
        self.log(managed.name, f"进程退出，状态码 {exit_code}", level)
        # A load failure happens before the model creates its RPC client.
        # The console still owns the bridge and must return it to ZERO.
        if managed.name == "deployer":
            with self._lock:
                cleanup = self._operation != "stopping" and self._managed_running("bridge")
            if cleanup:
                self.stop_all()

    def _managed_running(self, name: str) -> bool:
        process = self._processes.get(name)
        return process is not None and process.running

    def _external_processes(self) -> list[dict[str, Any]]:
        with self._lock:
            owned = {
                item.process.pid
                for item in self._processes.values()
                if item.running
            }
        matches: list[dict[str, Any]] = []
        for process_dir in Path("/proc").iterdir():
            try:
                pid = int(process_dir.name)
            except ValueError:
                continue
            if pid in owned or pid == os.getpid():
                continue
            try:
                args = [
                    item.decode("utf-8", errors="replace")
                    for item in (process_dir / "cmdline").read_bytes().split(b"\0")
                    if item
                ]
            except (OSError, PermissionError):
                continue
            script_names = {
                Path(argument).name
                for argument in args
                if not any(character.isspace() for character in argument)
            }
            matched = script_names.intersection(
                {"LerobotDeployer.py", "OpenpiDeployer.py", "hardware_bridge.py"}
            )
            if matched:
                matches.append(
                    {
                        "pid": pid,
                        "kind": (
                            "deployer"
                            if matched.intersection({"LerobotDeployer.py", "OpenpiDeployer.py"})
                            else "bridge"
                        ),
                        "command": shlex.join(args),
                    }
                )
        return matches

    def start_bridge(self, endpoint: str) -> ManagedProcess:
        with self._lock:
            generation = self._generation
            if self._managed_running("bridge"):
                raise ConsoleError("硬件桥已经由此控制台启动")
        if port_is_open(endpoint):
            raise ConsoleError(f"{endpoint} 已被外部桥接进程占用")
        with self._lock:
            self._last_endpoint = endpoint
        command, env = bridge_command(endpoint)
        with self._lock:
            if generation != self._generation:
                raise ConsoleError("启动已取消")
            return self._spawn("bridge", command, env)

    def start_deployment(self, raw: dict[str, Any] | None) -> ManagedProcess:
        with self._lock:
            generation = self._generation
        config = normalize_config(raw)
        with self._lock:
            self._last_endpoint = config["bridge_endpoint"]
        with self._lock:
            if self._managed_running("deployer"):
                raise ConsoleError("模型部署已经在运行")
        external = [
            item
            for item in self._external_processes()
            if item["kind"] == "deployer"
        ]
        if external:
            raise ConsoleError(f"检测到外部 Deployer 进程 PID {external[0]['pid']}")
        if not config.get("offline_dataset_root"):
            with self._lock:
                bridge = self._processes.get("bridge")
                if bridge is None or not bridge.running or bridge.state != "ready":
                    raise ConsoleError("请先由此控制台启动硬件桥；不会接管外部桥")
                if config["bridge_endpoint"] not in bridge.command:
                    raise ConsoleError("所选桥接地址与当前硬件桥不一致")
            if not port_is_open(config["bridge_endpoint"]):
                raise ConsoleError("硬件桥尚未就绪")
        if not config["diagnostics_jsonl"] or config["diagnostics_jsonl"] == "/tmp/openpi_piper_rtc_web.jsonl":
            config["diagnostics_jsonl"] = str(
                ROOT / "outputs/openpi_deployer" / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.jsonl"
            )
        command, env = deployment_command(config)
        with self._lock:
            self._diagnostics_path = (
                Path(config["diagnostics_jsonl"])
                if config["diagnostics_jsonl"]
                else None
            )
            self._runtime_summary = None
        with self._lock:
            if generation != self._generation:
                raise ConsoleError("启动已取消")
            return self._spawn("deployer", command, env)

    def start_stack(self, raw: dict[str, Any] | None) -> None:
        with self._lock:
            requested_generation = self._generation
        config = normalize_config(raw)
        with self._lock:
            if requested_generation != self._generation:
                raise ConsoleError("启动已取消")
        if config.get("offline_dataset_root"):
            with self._lock:
                if requested_generation != self._generation:
                    raise ConsoleError("启动已取消")
                self.start_deployment(config)
            return
        with self._lock:
            if requested_generation != self._generation:
                raise ConsoleError("启动已取消")
            if self._orchestrator is not None and self._orchestrator.is_alive():
                raise ConsoleError("完整启动流程正在进行")
            if self._managed_running("deployer"):
                raise ConsoleError("模型部署已经在运行")
            if self._external_processes():
                raise ConsoleError("检测到外部桥接或部署进程，请先停止后再由 Web 接管")
            if self._closed or self._operation in {"starting_stack", "stopping"}:
                raise ConsoleError("启动或停止操作正在进行")
            generation = self._generation
            self._operation = "starting_stack"
            self._operation_error = None

        def orchestrate() -> None:
            created_bridge = False
            try:
                with self._lock:
                    if generation != self._generation:
                        return
                    bridge = self._processes.get("bridge")
                    if bridge is None or not bridge.running:
                        bridge = self.start_bridge(config["bridge_endpoint"])
                        created_bridge = True
                    elif config["bridge_endpoint"] not in bridge.command:
                        raise ConsoleError("所选桥接地址与当前硬件桥不一致")
                if bridge.state != "ready":
                    self.log("system", "等待硬件桥完成相机初始化、回零和 J6 预置…")
                    deadline = time.monotonic() + 150
                    while not bridge.ready.wait(timeout=0.2):
                        with self._lock:
                            if generation != self._generation:
                                return
                        if time.monotonic() >= deadline:
                            raise ConsoleError("硬件桥 150 秒内未就绪")
                if not bridge.running or bridge.state not in {"ready", "starting"}:
                    raise ConsoleError("硬件桥启动失败，请查看日志")
                with self._lock:
                    if generation != self._generation:
                        return
                    self.start_deployment(config)
                    self._operation = "running"
            except Exception as exc:
                with self._lock:
                    if generation != self._generation:
                        return
                    self._operation = "failed"
                    self._operation_error = str(exc)
                self.log("system", f"完整启动失败：{exc}", "error")
                if created_bridge:
                    self._signal_process("bridge", timeout=90)

        with self._lock:
            if generation != self._generation:
                return
            self._orchestrator = threading.Thread(
                target=orchestrate, daemon=True, name="stack-orchestrator"
            )
            self._orchestrator.start()

    def _signal_process(self, name: str, timeout: float = 12.0) -> None:
        with self._lock:
            managed = self._processes.get(name)
            if managed is None or not managed.running:
                return
            homing = managed.state == "homing"
            managed.state = "stopping"
            pid = managed.process.pid
        if name == "bridge":
            # A model normally sends RPC stop itself. Let that ongoing home
            # finish instead of interrupting it with another SIGINT.
            try:
                managed.process.wait(timeout=timeout if homing else 0.5)
                return
            except subprocess.TimeoutExpired:
                with self._lock:
                    homing = homing or managed.state == "homing"
                if homing:
                    try:
                        managed.process.wait(timeout=timeout)
                        return
                    except subprocess.TimeoutExpired:
                        self.log("system", "硬件桥回零超时，保留进程等待检查", "error")
                        return
        self.log("system", f"向 {name} 发送安全停止信号")
        try:
            os.killpg(pid, signal.SIGINT)
            managed.process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            self.log("system", f"{name} 未响应 SIGINT，发送 SIGTERM", "warning")
        try:
            os.killpg(pid, signal.SIGTERM)
            managed.process.wait(timeout=5)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        self.log("system", f"{name} 未响应 SIGTERM，强制终止", "error")
        try:
            os.killpg(pid, signal.SIGKILL)
            managed.process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def stop_all(self, *, asynchronous: bool = True) -> None:
        with self._lock:
            self._generation += 1
            self._operation = "stopping"
            previous = self._stop_thread
            if previous is None or not previous.is_alive():
                def stop() -> None:
                    self._signal_process("deployer", timeout=90)
                    self._signal_process("bridge", timeout=90)
                    with self._lock:
                        self._operation = "idle"
                previous = threading.Thread(target=stop, daemon=True, name="safe-stop")
                self._stop_thread = previous
                previous.start()
        if not asynchronous:
            previous.join()

    def clear_logs(self) -> None:
        with self._lock:
            self._logs.clear()

    def logs_after(self, after: int) -> dict[str, Any]:
        with self._lock:
            entries = [entry for entry in self._logs if entry["id"] > after]
            latest = self._next_log_id - 1
        return {"entries": entries, "latest_id": latest}

    def _process_status(self, name: str) -> dict[str, Any]:
        managed = self._processes.get(name)
        if managed is None:
            return {
                "state": "stopped",
                "running": False,
                "pid": None,
                "exit_code": None,
                "uptime_s": 0,
                "progress": {"current": 0, "total": 0},
            }
        running = managed.running
        return {
            "state": managed.state,
            "running": running,
            "pid": managed.process.pid,
            "exit_code": managed.exit_code,
            "uptime_s": max(0, time.time() - managed.started_at) if running else 0,
            "progress": {
                "current": managed.progress_current,
                "total": managed.progress_total,
            },
            "command": shlex.join(managed.command),
        }

    def _gpu_status(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        cached_at, cached = self._gpu_cache
        if now - cached_at < 5:
            return cached
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    detail or f"nvidia-smi exited with status {result.returncode}"
                )
            rows = []
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 5:
                    try:
                        index = int(parts[0])
                    except ValueError:
                        continue

                    def metric(value: str) -> int:
                        try:
                            return int(value)
                        except ValueError:
                            return 0

                    rows.append(
                        {
                            "index": index,
                            "name": parts[1],
                            "memory_used_mb": metric(parts[2]),
                            "memory_total_mb": metric(parts[3]),
                            "utilization": metric(parts[4]),
                        }
                    )
            self._gpu_error = None
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            rows = []
            self._gpu_error = str(exc)
        self._gpu_cache = (now, rows)
        return rows

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            path = self._diagnostics_path
            runtime_summary = self._runtime_summary
        points: list[dict[str, Any]] = []
        live_summary: dict[str, Any] = {}
        read_error = None
        if path is not None and path.is_file():
            try:
                with path.open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    size = stream.tell()
                    stream.seek(max(0, size - 512_000))
                    raw = stream.read().decode("utf-8", errors="replace")
                lines = raw.splitlines()
                if size > 512_000 and lines:
                    lines = lines[1:]
                for line in lines[-240:]:
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(item, dict):
                        continue
                    live_summary = {
                        **item.get("counters", {}), **item.get("inference", {}),
                        **({"move_js": item["move_js"]} if "move_js" in item else {}),
                        **({"bridge_stats_error": item["bridge_stats_error"]} if "bridge_stats_error" in item else {}),
                    }
                    action = item.get("action")
                    target = item.get("target")
                    points.append(
                        {
                            "tick": item.get("tick"),
                            "tick_ms": round(float(item.get("tick_s", 0)) * 1000, 3),
                            "observation_ms": round(
                                float(item.get("observation_s", 0)) * 1000, 3
                            ),
                            "engine_ms": round(
                                float(item.get("engine_s", 0)) * 1000, 3
                            ),
                            "queue": item.get("queue_size"),
                            "has_action": action is not None,
                            "action_min": min(action) if action else None,
                            "action_max": max(action) if action else None,
                            "target_min": min(target) if target else None,
                            "target_max": max(target) if target else None,
                            "clipped": bool(item.get("limited", item.get("hardware_limit_clipped", False))),
                            "inference_ms": (
                                round(float(item["inference"]["inference_latency_s_last"]) * 1000, 3)
                                if item.get("inference", {}).get("inference_latency_s_last") is not None else None
                            ),
                        }
                    )
            except (OSError, ValueError, TypeError) as exc:
                read_error = str(exc)
        queues = [point["queue"] for point in points if point["queue"] is not None]
        actions = [point for point in points if point["has_action"]]
        # A missing action before the first successful action is startup
        # prefill.  Only missing actions after that boundary are runtime
        # starvation; the latter is the health signal shown in the console.
        first_action = next((index for index, point in enumerate(points) if point["has_action"]), None)
        if first_action is None:
            prefill_starved = len(points)
            runtime_starved = 0
        else:
            prefill_starved = first_action
            runtime_starved = sum(not point["has_action"] for point in points[first_action:])
        summary = {**live_summary, **(runtime_summary or {})}
        if summary:
            prefill_starved = int(summary.get("prefill_starved_ticks", prefill_starved))
            runtime_starved = int(
                summary.get(
                    "runtime_starved_ticks",
                    summary.get("starved_ticks", runtime_starved),
                )
            )
        return {
            "path": None if path is None else str(path),
            "summary_path": None if path is None else str(path.with_suffix(".summary.json")),
            "read_error": read_error,
            "points": points,
            "latest_queue": queues[-1] if queues else None,
            "minimum_queue": summary.get("minimum_queue", min(queues) if queues else None),
            "action_ticks": summary.get("actions", len(actions)),
            "clipped_ticks": summary.get("hardware_limit_clips", sum(point["clipped"] for point in points)),
            "prefill_starved_ticks": prefill_starved,
            "runtime_starved_ticks": runtime_starved,
            "starved_ticks": prefill_starved + runtime_starved,
            "runtime_summary": runtime_summary,
            "live_summary": live_summary,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            bridge = self._process_status("bridge")
            deployer = self._process_status("deployer")
            operation = self._operation
            operation_error = self._operation_error
            endpoint = self._last_endpoint
        return {
            "bridge": bridge,
            "deployer": deployer,
            "operation": operation,
            "operation_error": operation_error,
            "bridge_endpoint": endpoint,
            "bridge_port_open": port_is_open(endpoint),
            "external_processes": self._external_processes(),
            "gpu": self._gpu_status(),
            "gpu_error": self._gpu_error,
            "diagnostics": self.diagnostics(),
        }


def _numeric_checkpoints(run: Path) -> list[Path]:
    """Return readable numeric checkpoint directories without propagating stale-path errors."""
    try:
        candidates = [
            child
            for child in run.iterdir()
            if child.is_dir() and child.name.isdigit()
        ]
    except OSError:
        return []
    readable: list[Path] = []
    for child in candidates:
        try:
            if (child / "params/_METADATA").is_file():
                readable.append(child)
        except OSError:
            continue
    return sorted(readable, key=lambda item: int(item.name))


def discover_checkpoints() -> list[dict[str, Any]]:
    base = ROOT / "models/finetuned_models"
    items: list[dict[str, Any]] = []
    if not base.is_dir():
        return items
    seen: set[Path] = set()
    try:
        manifests = sorted(base.rglob("training_manifest.json"))
    except OSError:
        return items
    for config_path in manifests:
        try:
            manifest = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("trainer") != "OpenpiTrainer":
            continue
        config = manifest.get("config", {})
        local_run = config_path.parent
        if local_run.name.isdigit():
            local_run = local_run.parent
        configured_run = Path(config.get("output_dir", local_run)).expanduser()
        if not configured_run.is_absolute():
            configured_run = ROOT / configured_run
        candidate_runs = [local_run.resolve()]
        configured_run = configured_run.resolve()
        if configured_run not in candidate_runs:
            candidate_runs.append(configured_run)

        directory = None
        for run in candidate_runs:
            numeric = _numeric_checkpoints(run)
            if numeric:
                directory = numeric[-1]
                break
        if directory is None:
            try:
                local_is_checkpoint = (config_path.parent / "params/_METADATA").is_file()
            except OSError:
                local_is_checkpoint = False
            directory = config_path.parent if local_is_checkpoint else None
        if directory is None or directory in seen:
            continue
        seen.add(directory)
        try:
            label = str(directory.relative_to(base))
        except ValueError:
            label = directory.name
        items.append(
            {
                "path": str(directory),
                "label": label,
                "policy_type": "OpenPI LoRA" if config.get("finetune_mode") == "lora" else "OpenPI Full",
                "chunk_size": config.get("action_horizon", 50),
                "n_action_steps": config.get("n_action_steps", 10),
                "rtc_training_max_delay": config.get("rtc_max_delay", 0),
                "action_mode": config.get("action_mode"),
            }
        )
    return items


class ConsoleHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], supervisor: ProcessSupervisor):
        super().__init__(address, ConsoleHandler)
        self.supervisor = supervisor


class ConsoleHandler(LocalHandlerMixin, BaseHTTPRequestHandler):
    server: ConsoleHTTPServer

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, Any]:
        return read_json(self, ConsoleError)

    def _static(self, path: str) -> None:
        routes = {
            "/": STATIC_DIR / "index.html",
            "/index.html": STATIC_DIR / "index.html",
            "/app.js": STATIC_DIR / "app.js",
            "/styles.css": STATIC_DIR / "styles.css",
        }
        file_path = routes.get(path)
        if file_path is None or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
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
                        "checkpoints": discover_checkpoints(),
                        "status": self.server.supervisor.status(),
                    }
                )
            elif parsed.path == "/api/status":
                self._json(self.server.supervisor.status())
            elif parsed.path == "/api/logs":
                query = parse_qs(parsed.query)
                after = int(query.get("after", ["0"])[0])
                self._json(self.server.supervisor.logs_after(after))
            else:
                self._static(parsed.path)
        except (ConsoleError, ValueError) as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except BrokenPipeError:
            pass
        except Exception as exc:
            self.server.supervisor.log("web", f"GET {parsed.path}: {exc}", "error")
            self._json(
                {"ok": False, "error": f"服务端错误：{exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/preview":
                config = normalize_config(body.get("config"))
                bridge, _ = bridge_command(config["bridge_endpoint"])
                deployer, _ = deployment_command(config)
                self._json(
                    {
                        "ok": True,
                        "bridge": shlex.join(bridge),
                        "deployer": shlex.join(deployer),
                        "config": config,
                    }
                )
            elif parsed.path == "/api/bridge/start":
                endpoint = str(body.get("endpoint", DEFAULT_ENDPOINT))
                managed = self.server.supervisor.start_bridge(endpoint)
                self._json({"ok": True, "pid": managed.process.pid}, HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/deployment/start":
                managed = self.server.supervisor.start_deployment(body.get("config"))
                self._json({"ok": True, "pid": managed.process.pid}, HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/stack/start":
                self.server.supervisor.start_stack(body.get("config"))
                self._json({"ok": True}, HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/stop":
                self.server.supervisor.stop_all(asynchronous=True)
                self._json({"ok": True}, HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/logs/clear":
                self.server.supervisor.clear_logs()
                self._json({"ok": True})
            else:
                self._json({"ok": False, "error": "接口不存在"}, HTTPStatus.NOT_FOUND)
        except ConsoleError as exc:
            self.server.supervisor.log("web", str(exc), "warning")
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.CONFLICT)
        except BrokenPipeError:
            pass
        except Exception as exc:
            self.server.supervisor.log("web", f"POST {parsed.path}: {exc}", "error")
            self._json(
                {"ok": False, "error": f"服务端错误：{exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    for required in (PIPER_PYTHON, OPENPI_PYTHON, BRIDGE_SCRIPT, DEPLOYER_SCRIPT):
        if not required.exists():
            parser.error(f"required path is missing: {required}")
    supervisor = ProcessSupervisor()
    server = ConsoleHTTPServer((args.host, args.port), supervisor)
    shown_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"OpenpiDeployer Web: http://{shown_host}:{server.server_address[1]}", flush=True)
    print("Ctrl-C 会安全停止由本页面启动的 Deployer 和硬件桥。", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n正在关闭 Web 控制台…", flush=True)
    finally:
        server.server_close()
        supervisor._closed = True
        supervisor.stop_all(asynchronous=False)


if __name__ == "__main__":
    main()
