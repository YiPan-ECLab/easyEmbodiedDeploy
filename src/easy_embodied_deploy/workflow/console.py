"""Local workflow editor API; opening the page never connects hardware."""

import argparse
import base64
import json
import signal
import threading
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from ..calibration import DEFAULT_SERIALS, encode_image
from ..contracts import boolean
from ..http import read_json
from ..module_http import ROOT, ModuleHandler
from ..system import gpu_status
from .plan import camera_point, new_plan, validate, require_models
from .library import SharedPlanStore
from .runtime import WorkflowController


class WorkflowServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, directory=None, controller=None):
        super().__init__(address, Handler)
        self.controller = controller or WorkflowController()
        self.store = SharedPlanStore(directory or ROOT / "datasets/workflows")
        self.operations = threading.RLock()
        self.snapshots = OrderedDict()

    def library(self):
        return self.store.snapshot()['points']

    def save_point(self, point):
        point = dict(point, id=point.get("id") or uuid.uuid4().hex)
        return self.store.save_resource('points', point)


class Handler(ModuleHandler, BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            parsed = urlsplit(self.path)
            query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
            controller = self.server.controller
            if parsed.path == "/api/bootstrap":
                calibrations = [str(p) for p in (ROOT / "datasets/calibration").glob("*/calibration_bundle.json")]
                resources = self.server.store.snapshot()
                self.send(dict(api_version=7, plan=self.server.store.resolve(new_plan()), workflows=self.server.store.listing(),
                               library=resources, points=resources['points'], models=resources['models'],
                               calibrations=calibrations, serials=DEFAULT_SERIALS, state=controller.snapshot()))
            elif parsed.path == "/api/library":
                self.send(self.server.store.snapshot())
            elif parsed.path == "/api/devices":
                self.send(dict(gpus=gpu_status()))
            elif parsed.path == "/api/state":
                self.send(dict(controller.snapshot(), library_revision=self.server.store.revision()))
            elif parsed.path == "/api/workflow":
                self.send(self.server.store.load(query["id"]))
            elif parsed.path == "/api/telemetry":
                if not controller.hardware:
                    raise ValueError("请先连接设备或模拟器")
                self.send(controller.hardware.snapshot())
            elif parsed.path == "/api/frame":
                if not controller.hardware:
                    raise ValueError("请先连接设备")
                rgb, _, _ = controller.hardware.camera(query.get("camera", "agentview"))
                self.send(encode_image(rgb), content_type="image/jpeg")
            elif parsed.path == "/api/health":
                self.send({"status": "ok"})
            else:
                self.static("workflow")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self.fail(exc)

    def do_POST(self):
        try:
            body = read_json(self)
            route = urlsplit(self.path).path
            controller = self.server.controller
            # Stop/pause bypass slow file/model/camera operations.
            if route in ("/api/stop", "/api/emergency", "/api/pause", "/api/resume", "/api/next"):
                if route.endswith("emergency"):
                    controller.stop(emergency=True)
                else:
                    getattr(controller, route.rsplit("/", 1)[-1])()
                self.send(controller.snapshot())
                return
            with self.server.operations:
                if route == "/api/save":
                    identifier = self.server.store.save(body["plan"], body.get("id"))
                    self.send(dict(id=identifier, plan=self.server.store.load(identifier), workflows=self.server.store.listing()))
                elif route == "/api/validate":
                    self.send(self.server.store.resolve(body["plan"]))
                elif route in ("/api/model/validate", "/api/model/save"):
                    check = new_plan()
                    check['settings'] = body.get('settings', {})
                    check['models'] = [body['model']]
                    model = validate(check)['models'][0]
                    require_models([model], [model['id']])
                    if route.endswith('/save'):
                        model = self.server.store.save_resource('models', model, body.get('settings'))
                    self.send(model)
                elif route == "/api/model/delete":
                    if controller.busy():
                        raise ValueError('执行中不能删除模型；请结束执行后再删除')
                    controller.models.close(body['id'])
                    self.server.store.delete_resource('models', body['id'])
                    self.send(self.server.store.snapshot())
                elif route == "/api/connect":
                    controller.connect(boolean(body.get("simulated", True), "模拟模式"), body.get("config", {}))
                    self.send(controller.snapshot())
                elif route == "/api/disconnect":
                    controller.disconnect()
                    self.server.snapshots.clear()
                    self.send(controller.snapshot())
                elif route == "/api/teach":
                    controller.teach(boolean(body["enabled"], "示教模式"), body.get("side", "both"))
                    self.send(controller.snapshot())
                elif route == "/api/home":
                    controller.home(boolean(body.get("close_grippers", False), "回零时闭合夹爪"))
                    self.send(controller.snapshot())
                elif route == "/api/recover":
                    controller.recover()
                    self.send(controller.snapshot())
                elif route == "/api/capture":
                    point = controller.capture(body.get("name", "录制点"), body.get("side", "both"), body.get("space", "joint"))
                    self.send(self.server.save_point(point) if body.get("persist", True) else point)
                elif route == "/api/point/save":
                    self.send(self.server.save_point(body["point"]))
                elif route == "/api/point/delete":
                    self.server.store.delete_resource('points', body['id'])
                    self.send(self.server.store.snapshot())
                elif route == "/api/from-hdf5":
                    # Read recorded feedback, never replay the recorder or mutate a file.
                    import h5py
                    path = Path(body["path"]).expanduser().resolve()
                    with h5py.File(path, "r") as handle:
                        if handle.attrs.get("format") != "pyagxarm_direct_v1":
                            raise ValueError("请选择本系统采集的 HDF5")
                        demos = handle["data"]
                        demo = demos[body.get("demo") or sorted(demos.keys())[0]]
                        index = int(body.get("frame", 0))
                        if not 0 <= index < len(demo["states"]):
                            raise ValueError("帧索引超出范围（从 0 开始）")
                        q = demo["states"][index]
                        space = body.get("space", "joint")
                        arms = {}
                        for side, offset in (("left", 0), ("right", 7)):
                            if body.get("side", "both") not in (side, "both"):
                                continue
                            arms[side] = (q[offset:offset+7].tolist() if space == "joint" else
                                          [*demo[f"obs/{side}_ee_states"][index].tolist(), float(q[offset+6])])
                    point = dict(id=uuid.uuid4().hex, name=body.get("name", "轨迹点"), space=space, arms=arms,
                                 source=dict(kind="hdf5", path=str(path), frame=index))
                    self.send(self.server.save_point(point))
                elif route == "/api/snapshot":
                    if not controller.hardware or controller.busy() or controller.hardware.teaching:
                        raise ValueError("请连接设备，结束示教并停止运动后冻结画面")
                    role = body.get("camera", "agentview")
                    rgb, depth, camera = controller.hardware.camera(role)
                    snapshot = dict(controller.hardware.snapshot(), depth=depth, camera=camera)
                    identifier = uuid.uuid4().hex
                    self.server.snapshots[identifier] = snapshot
                    while len(self.server.snapshots) > 4:
                        self.server.snapshots.popitem(last=False)
                    self.send(dict(id=identifier, camera=camera, has_depth=depth is not None,
                                   image="data:image/jpeg;base64,"+base64.b64encode(encode_image(rgb)).decode()))
                elif route == "/api/pick":
                    snapshot = self.server.snapshots.get(body["snapshot"])
                    if snapshot is None:
                        raise ValueError("快照已过期，请重新冻结画面")
                    path = Path(body["calibration"]).expanduser()
                    bundle = json.loads(path.read_text())
                    point = camera_point(snapshot, body["pixel"], bundle, body["side"],
                                         body.get("tcp_offset", [0, 0, 0]), body.get("offset", [0, 0, .05]))
                    point.update(id=uuid.uuid4().hex, name=body.get("name", "相机选点"), notes="")
                    self.send(self.server.save_point(point))
                elif route == "/api/preload":
                    controller.preload(self.server.store.resolve(body["plan"]), body.get("ids"))
                    self.send(controller.snapshot())
                elif route == "/api/unload":
                    if controller.busy():
                        raise ValueError("执行中不能卸载模型")
                    controller.models.close(body.get("id"))
                    self.send(controller.snapshot())
                elif route == "/api/release-model":
                    identifier = body.get('id')
                    if not isinstance(identifier, str) or not identifier:
                        raise ValueError('请选择要释放的模型')
                    controller.release_model(identifier)
                    self.send(controller.snapshot())
                elif route == "/api/run":
                    controller.start(self.server.store.resolve(body["plan"]), body.get("start", 0), body.get("repetitions", 1),
                                     boolean(body.get("home_before", True), "执行前回零"))
                    self.send(controller.snapshot())
                else:
                    self.send({"error": "接口不存在"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self.fail(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    server = WorkflowServer((args.host, args.port))

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    print(f"Workflow: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.controller.close()
        server.server_close()


if __name__ == "__main__":
    main()
