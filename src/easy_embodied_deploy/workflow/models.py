"""Multiple independently warmed models, never owners of physical hardware."""

import copy
import json
import os
import secrets
import subprocess
import threading
import time
from collections import deque
from multiprocessing.connection import Client

import numpy as np

from ..module_http import ROOT


class ModelProcess:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.lock = threading.RLock()
        self.rpc_lock = threading.Lock()
        self.stage, self.detail = "starting", "启动独立推理进程"
        self.port, self.contract = None, None
        self.inference = {}
        self.latencies = deque(maxlen=20)
        self.logs = deque(maxlen=30)
        self.auth = secrets.token_bytes(32)
        self.started = time.monotonic()
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
                   PYTHONUNBUFFERED="1", WORKFLOW_AUTH=self.auth.hex(),
                   XLA_PYTHON_CLIENT_PREALLOCATE="false")
        backend = config["backend"]
        environment = {"mock": "piper-home", "lerobot": "lerobot-pi05", "openpi": "openpi-jax"}[backend]
        if backend == "openpi":
            device = config.get("device", "cuda:0")
            env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if device.startswith("cuda:") else ""
        self.process = subprocess.Popen([str(ROOT / f"thirdParty/.venvs/{environment}/bin/python"),
            "-m", "easy_embodied_deploy.workflow.model_worker", "--config", json.dumps(config)],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace", start_new_session=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            with self.lock:
                if line.startswith("WORKFLOW "):
                    value = json.loads(line[len("WORKFLOW "):])
                    self.stage, self.detail = value["stage"], value.get("detail", "")
                    self.port = value.get("port", self.port)
                    self.contract = value.get("contract", self.contract)
                    self.inference = value.get("inference", self.inference)
                else:
                    self.logs.append(line.strip()[-1000:])
        code = self.process.wait()
        self.process.stdout.close()
        with self.lock:
            if self.stage not in ("failed", "unloaded"):
                self.stage, self.detail = "failed", f"推理进程已退出 ({code})"

    def snapshot(self):
        with self.lock:
            return dict(id=self.config["id"], name=self.config.get("name", self.config["id"]),
                        stage=self.stage, detail=self.detail, elapsed=round(time.monotonic()-self.started, 1),
                        pid=self.process.pid, logs=list(self.logs))

    def infer(self, observation, task, prefix=None, prefix_start=1):
        with self.rpc_lock:
            started = time.monotonic()
            with self.lock:
                if self.stage != "ready" or self.process.poll() is not None:
                    raise RuntimeError(self.detail)
                port = self.port
            with Client(("127.0.0.1", port), authkey=self.auth) as connection:
                connection.send(dict(op="infer", observation=observation, task=task,
                                     prefix=prefix, prefix_start=prefix_start))
                if not connection.poll(120):
                    raise TimeoutError("模型推理超过 120 秒")
                reply = connection.recv()
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error", "模型推理失败"))
            targets = np.asarray(reply["targets"], dtype=float)
            if targets.ndim != 2 or targets.shape[1] != 14 or not len(targets) or not np.isfinite(targets).all():
                raise ValueError("模型返回无效动作 chunk")
            self.latencies.append(time.monotonic()-started)
            return targets

    def close(self):
        with self.lock:
            self.stage = "unloaded"
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(5)
        self.reader.join(2)


class ModelPool:
    def __init__(self):
        self.lock = threading.RLock()
        self.models = {}

    def preload(self, configs):
        with self.lock:
            for config in configs:
                existing = self.models.get(config["id"])
                if existing and existing.config == config and existing.stage != "failed":
                    continue
                if existing:
                    existing.close()
                self.models[config["id"]] = ModelProcess(config)

    def get(self, identifier):
        with self.lock:
            return self.models[identifier]

    def snapshot(self):
        with self.lock:
            return [model.snapshot() for model in self.models.values()]

    def close(self, identifier=None):
        with self.lock:
            ids = [identifier] if identifier else list(self.models)
            for key in ids:
                model = self.models.pop(key, None)
                if model:
                    model.close()
