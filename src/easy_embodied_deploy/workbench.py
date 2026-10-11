"""Local reverse proxy and process owner for the isolated consoles."""

import argparse
import concurrent.futures
import http.client
import json
import mimetypes
import os
import re
import signal
import subprocess
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .http import LocalHandlerMixin, read_json

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "web/workbench"
MODULES = {
    "files": (
        "模型与数据目录", "资源", "DirectoryManager/web_server.py", "piper-home",
        "目录管理 · 路径复制 · 可恢复删除",
    ),
    "recorder": (
        "Piper 采集",
        "采集",
        "PiperRecorder/web_server.py",
        "piper-home",
        "踏板操作 · 双臂示教 · HDF5",
    ),
    "calibration": (
        "相机标定", "采集", "CameraCalibration/web_server.py", "piper-home",
        "左右腕部 · 全局双 base · 自定义标定板",
    ),
    "viewer": (
        "轨迹浏览",
        "数据",
        "PiperViewer/viewer.py",
        "piper-home",
        "三路回放 · 片段融合 · 动作曲线",
    ),
    "processor": (
        "数据转换",
        "数据",
        "LerobotDataProcessor/DataProcessor.py",
        "lerobot-pi05",
        "多目录合并 · 降采样 · LeRobot v3",
    ),
    "analysis": (
        "质量分析",
        "数据",
        "DataAnalysis/DataAnalysis.py",
        "lerobot-pi05",
        "时序诊断 · 数据验证 · 异常清理",
    ),
    "lerobot-trainer": (
        "LeRobot 训练",
        "训练",
        "LerobotTrainer/web_server.py",
        "lerobot-pi05",
        "PI05 · LoRA · RTC · 断点恢复",
    ),
    "openpi-trainer": (
        "OpenPI 训练",
        "训练",
        "OpenpiTrainer/web_server.py",
        "openpi-jax",
        "JAX · FSDP · LoRA · RTC",
    ),
    "lerobot-deployer": (
        "LeRobot 部署",
        "部署",
        "LerobotDeployer/web_server.py",
        "lerobot-pi05",
        "PyTorch 推理 · RTC · 真机控制",
    ),
    "openpi-deployer": (
        "OpenPI 部署",
        "部署",
        "OpenpiDeployer/web_server.py",
        "openpi-jax",
        "JAX 推理 · 离线测试 · 真机控制",
    ),
}


class Service:
    def __init__(self, name):
        self.name = name
        self.lock = threading.RLock()
        self.process = None
        self.port = None
        self.logs = deque(maxlen=100)
        self.ready = threading.Event()
        self.error = None
        self.stopping = False
        self.closing = False

    def snapshot(self):
        title, group, entry, environment, description = MODULES[self.name]
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            state = (
                "stopping"
                if running and self.stopping
                else "ready"
                if running and self.port
                else "starting"
                if running
                else "failed"
                if self.error
                else "stopped"
            )
            return dict(
                id=self.name,
                title=title,
                group=group,
                description=description,
                environment=environment,
                available=(
                    ROOT / f"thirdParty/.venvs/{environment}/bin/python"
                ).exists(),
                state=state,
                pid=self.process.pid if running else None,
                path=f"/modules/{self.name}/",
                error=self.error,
                logs=list(self.logs),
            )

    def start(self):
        with self.lock:
            if self.stopping or self.closing:
                raise RuntimeError("服务正在关闭，请稍后重试")
            if self.process is not None and self.process.poll() is None:
                return
            title, group, entry, environment, description = MODULES[self.name]
            python = ROOT / f"thirdParty/.venvs/{environment}/bin/python"
            if not python.exists():
                self.error = f"缺少 {environment} 环境，请按 thirdParty/README.md 创建"
                raise RuntimeError(self.error)
            self.port = None
            self.error = None
            self.ready.clear()
            self.logs.clear()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.pop("PYTHONPATH", None)
            command = [
                str(python),
                str(ROOT / "scripts" / entry),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
            ]
            if self.name == "viewer":
                command += ["--directory", str(ROOT / "datasets/hdf5")]
            self.process = subprocess.Popen(
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
            threading.Thread(
                target=self._read, args=(self.process,), daemon=True
            ).start()

    def _read(self, process):
        for line in process.stdout:
            with self.lock:
                if process is not self.process:
                    continue
                self.logs.append(line.rstrip()[-2000:])
                match = re.search(r"http://127\.0\.0\.1:(\d+)", line)
                if match and self.port is None:
                    self.port = int(match.group(1))
                    self.ready.set()
        code = process.wait()
        process.stdout.close()
        with self.lock:
            if process is self.process:
                self.port = None
                if code and not self.stopping:
                    self.error = f"服务退出（{code}），请查看服务日志"
                self.stopping = False
                self.ready.set()

    def stop(self, *, closing=False):
        with self.lock:
            if closing:
                self.closing = True
            process = self.process
            if not process or process.poll() is not None:
                return
            self.stopping = True
        # Signal only the console. It owns the graceful shutdown of its children.
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=300)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            with self.lock:
                self.error = "服务清理超时，保留进程以便检查，未强制终止设备控制"
                self.stopping = True


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address):
        self.services = {name: Service(name) for name in MODULES}
        super().__init__(address, Handler)

    def close_services(self):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.services)
        ) as pool:
            futures = [
                pool.submit(service.stop, closing=True)
                for service in self.services.values()
            ]
            for future in futures:
                future.result()


class Handler(LocalHandlerMixin, BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def send(self, payload, status=200, content_type="application/json; charset=utf-8"):
        body = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload, ensure_ascii=False).encode()
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def proxy(self):
        parts = self.path.split("/", 3)
        name = parts[2] if len(parts) > 2 else ""
        service = self.server.services.get(name)
        if service is None:
            self.send({"error": "模块不存在"}, 404)
            return
        if len(parts) < 4:
            self.send_response(307)
            self.send_header("Location", f"/modules/{name}/")
            self.end_headers()
            return
        with service.lock:
            port = service.port
            running = service.process is not None and service.process.poll() is None
        if not running or not port:
            self.send({"error": "模块服务未就绪，请从工作台启动服务"}, 503)
            return
        body = None
        if self.command == "POST":
            # Parse once at the boundary; never forward an unbounded body.
            body = json.dumps(read_json(self), ensure_ascii=False).encode()
        headers = {
            key: self.headers[key]
            for key in ("Host", "Origin", "Sec-Fetch-Site", "Content-Type")
            if self.headers.get(key)
        }
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        try:
            connection.request(self.command, "/" + parts[3], body=body, headers=headers)
            response = connection.getresponse()
            content = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() in {
                    "content-type",
                    "cache-control",
                    "content-security-policy",
                    "content-disposition",
                }:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (OSError, http.client.HTTPException) as exc:
            self.send({"error": f"模块连接失败：{exc}"}, 502)
        finally:
            connection.close()

    def do_GET(self):
        if self.path.startswith("/modules/"):
            self.proxy()
            return
        route = urlsplit(self.path).path
        if route == "/api/status":
            self.send(
                {
                    "modules": [
                        service.snapshot() for service in self.server.services.values()
                    ]
                }
            )
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
            if self.path.startswith("/modules/"):
                self.proxy()
                return
            body = read_json(self)
            service = self.server.services.get(body.get("module"))
            if service is None:
                raise ValueError("模块不存在")
            if self.path == "/api/services/start":
                service.start()
            elif self.path == "/api/services/stop":
                with service.lock:
                    if service.stopping:
                        raise RuntimeError("服务正在关闭")
                    if service.process is not None and service.process.poll() is None:
                        service.stopping = True
                    threading.Thread(target=service.stop, daemon=True).start()
            else:
                self.send({"error": "接口不存在"}, 404)
                return
            self.send({"ok": True, "module": service.snapshot()})
        except (ValueError, TypeError) as exc:
            self.send({"error": str(exc)}, 400)
        except (RuntimeError, OSError) as exc:
            self.send({"error": str(exc)}, 409)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    server = WorkbenchServer((args.host, args.port))
    print(
        f"easyEmbodiedDeploy: http://{args.host}:{server.server_address[1]}", flush=True
    )
    print("打开模块会启动网页服务；设备操作需在模块页面显式触发。", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("正在等待各模块完成退出清理…", flush=True)
    finally:
        server.close_services()
        server.server_close()
