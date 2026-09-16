"""Supervise the unmodified recording workflow through its operator channel."""

import argparse
import json
import math
import mimetypes
import os
import signal
import subprocess
import sys
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .http import LocalHandlerMixin, read_json

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "web/recorder"
SCRIPT = ROOT / "scripts/PiperRecorder/record.py"
PYTHON = ROOT / "thirdParty/.venvs/piper-home/bin/python"
EVENT_PREFIX = "PIPER_RECORDER_EVENT="


def defaults():
    # Import constants from the real script; importing does not open devices.
    sys.path.insert(0, str(SCRIPT.parent)) if str(
        SCRIPT.parent
    ) not in sys.path else None
    import record

    return {
        "output": str(record.OUTPUT_DIR),
        "hz": record.RECORD_HZ,
        "max_task_gap_s": record.MAX_TASK_GAP_S,
        "start_joint_delta_rad": record.START_JOINT_DELTA_RAD,
        "start_gripper_delta_m": record.START_GRIPPER_DELTA_M,
    }


def command_for(raw):
    if not isinstance(raw, dict):
        raise ValueError("采集配置必须为对象")
    config = defaults()
    config.update({key: value for key, value in raw.items() if key in config})
    output = str(config["output"]).strip()
    if not output or "\x00" in output:
        raise ValueError("请填写输出目录")
    path = Path(output).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    command = [str(PYTHON), "-u", str(SCRIPT), "--web-control", "--output", str(path)]
    config["output"] = str(path)
    for key in (
        "hz",
        "max_task_gap_s",
        "start_joint_delta_rad",
        "start_gripper_delta_m",
    ):
        value = config[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} 必须是有限正数")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须是有限正数")
        config[key] = value
        command += ["--" + key.replace("_", "-"), str(value)]
    return command, config


class RecorderSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.logs = deque(maxlen=500)
        self.saved = deque(maxlen=100)
        self.state = {"phase": "idle", "token": 0, "accepts": [], "frames": 0}
        self.config = defaults()
        self.pending = False
        self.stopping = False
        self.generation = 0
        self.closed = False

    def status(self):
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            return {
                **self.state,
                "running": running,
                "pending": self.pending,
                "stopping": self.stopping,
                "session_id": self.generation,
                "pid": self.process.pid if running else None,
                "exit_code": self.process.poll() if self.process else None,
                "config": self.config.copy(),
                "logs": list(self.logs),
                "saved": list(self.saved),
            }

    def start(self, raw):
        command, config = command_for(raw)
        with self.lock:
            if self.closed:
                raise RuntimeError("采集服务正在关闭")
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("采集会话已经在运行")
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            self.generation += 1
            self.config = config
            self.state = {
                "phase": "initializing",
                "token": 0,
                "accepts": [],
                "frames": 0,
                "error": None,
            }
            self.pending = self.stopping = False
            self.logs.clear()
            self.saved.clear()
            threading.Thread(
                target=self._read, args=(self.process,), daemon=True
            ).start()

    def _send(self, action, value=""):
        if self.pending or action not in self.state.get("accepts", []):
            raise RuntimeError("当前阶段不接受该操作，请等待阶段更新")
        message = {"action": action, "value": value, "token": self.state["token"]}
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        self.pending = True

    def input(self, payload):
        with self.lock:
            if not self.process or self.process.poll() is not None:
                raise RuntimeError("请先启动采集会话")
            if self.stopping:
                raise RuntimeError("会话正在结束")
            if (
                payload.get("session_id") != self.generation
                or payload.get("token") != self.state["token"]
            ):
                raise RuntimeError("页面阶段已过期，请等待刷新后再操作")
            value = str(payload.get("value", ""))
            if len(value) > 2000 or any(c in value for c in ("\n", "\r", "\x00")):
                raise ValueError("任务指令应为不超过 2000 字的单行文本")
            self._send(payload.get("action"), value)

    def _advance_stop(self):
        if not self.stopping or self.pending:
            return
        phase = self.state["phase"]
        if phase == "instruction":
            self._send("submit", ":q")
        elif phase == "review":
            self._send("submit", "n")
        elif phase in {"waiting_motion", "recording"}:
            self._send("finish")

    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.stopping = True
                self._advance_stop()

    def _read(self, process):
        try:
            for line in process.stdout:
                line = line.rstrip("\r\n")
                with self.lock:
                    if process is not self.process:
                        continue
                    if line.startswith(EVENT_PREFIX):
                        try:
                            event = json.loads(line[len(EVENT_PREFIX) :])
                            if (
                                not isinstance(event, dict)
                                or not {"token", "phase"} <= event.keys()
                            ):
                                raise ValueError("采集事件缺少阶段字段")
                        except ValueError as exc:
                            self.logs.append(f"忽略无效采集状态：{exc}")
                            continue
                        if event["phase"] == "preposition":
                            self.state["error"] = None
                        if event["token"] != self.state["token"]:
                            self.pending = False
                        self.state.update(event)
                        if event["phase"] == "instruction":
                            self.state["frames"] = 0
                        if event["phase"] == "saved":
                            self.saved.append(
                                {"path": event["path"], "frames": event["frames"]}
                            )
                        self._advance_stop()
                    elif line.strip():
                        self.logs.append(line[-4000:])
            process.wait()
        except (OSError, ValueError) as exc:
            with self.lock:
                self.logs.append(f"读取采集状态失败：{exc}")
        finally:
            process.stdout.close()
            if process.stdin:
                process.stdin.close()
            with self.lock:
                if process is self.process:
                    self.state.update(
                        phase="exited" if process.poll() == 0 else "failed", accepts=[]
                    )
                    self.pending = self.stopping = False

    def shutdown(self):
        with self.lock:
            self.closed = True
        self.stop()
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            process.wait(timeout=150)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                with self.lock:
                    self.logs.append("采集退出超时，请检查设备；保留进程，不中断回零")


class Handler(LocalHandlerMixin, BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def send(self, payload, status=200, content_type="application/json; charset=utf-8"):
        body = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/api/bootstrap":
            self.send({"defaults": defaults(), "status": self.server.session.status()})
        elif route == "/api/status":
            self.send(self.server.session.status())
        elif route == "/api/health":
            self.send({"status": "ok"})
        elif route in {"/", "/index.html", "/styles.css", "/app.js"}:
            path = STATIC / ("index.html" if route == "/" else route[1:])
            self.send(
                path.read_bytes(),
                content_type=(mimetypes.guess_type(path.name)[0] or "text/plain")
                + "; charset=utf-8",
            )
        else:
            self.send({"error": "接口不存在"}, 404)

    def do_POST(self):
        try:
            body = read_json(self)
            route = urlsplit(self.path).path
            if route == "/api/start":
                self.server.session.start(body)
            elif route == "/api/input":
                self.server.session.input(body)
            elif route == "/api/stop":
                self.server.session.stop()
            elif route == "/api/check":
                result = subprocess.run(
                    [str(PYTHON), str(SCRIPT), "--check"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.send(
                    {
                        "ok": result.returncode == 0,
                        "output": result.stdout + result.stderr,
                    }
                )
                return
            elif route == "/api/preview":
                command, config = command_for(body)
                self.send({"command": command, "config": config})
                return
            else:
                self.send({"error": "接口不存在"}, 404)
                return
            self.send({"ok": True})
        except (ValueError, TypeError) as exc:
            self.send({"error": str(exc)}, 400)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            self.send({"error": str(exc)}, 409)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8764)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.session = RecorderSession()
    print(f"PiperRecorder: http://{args.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.session.shutdown()
        server.server_close()
