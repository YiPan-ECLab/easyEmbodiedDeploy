"""Three-camera calibration console; no robot motion commands."""

import argparse
import base64
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import cv2

from .calibration import CalibrationRig, CalibrationStore, DEFAULT_SERIALS, encode_image
from .calibration_math import DEFAULT_BOARD, JOBS, board_config, board_image, detect
from .http import read_json
from .contracts import boolean
from .module_http import ModuleHandler, ROOT


class Handler(ModuleHandler, BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            route = urlsplit(self.path)
            query = {k: v[-1] for k, v in parse_qs(route.query).items()}
            store, rig = self.server.store, self.server.rig
            if route.path == "/api/bootstrap":
                self.send(
                    dict(
                        board=DEFAULT_BOARD,
                        serials=DEFAULT_SERIALS,
                        jobs=JOBS,
                        directory=str(store.directory),
                        sessions=store.listing(),
                        rig=rig.status(),
                    )
                )
            elif route.path == "/api/state":
                self.send(dict(sessions=store.listing(), rig=rig.status()))
            elif route.path == "/api/session":
                self.send(store.load(query.get("id")))
            elif route.path == "/api/frame":
                self.send(encode_image(rig.read()[0]), content_type="image/jpeg")
            elif route.path == "/api/sample":
                session = store.load(query.get("id"))
                sample_id = query.get("sample")
                if sample_id not in [s["id"] for s in session["samples"]]:
                    raise ValueError("样本不存在")
                self.send(
                    (
                        store.directory_for(session["id"]) / (sample_id + ".jpg")
                    ).read_bytes(),
                    content_type="image/jpeg",
                )
            elif route.path == "/api/export":
                path = store.directory_for(query.get("id")) / "calibration_bundle.json"
                if path.is_symlink():
                    raise ValueError("标定包不能是符号链接")
                self.send(path.read_bytes())
            elif route.path == "/api/board.svg":
                board = board_config(query)
                image = board_image(board)
                ok, png = cv2.imencode(".png", image)
                extra = int(board["kind"] == "chessboard")
                width, height = (
                    (board["columns"] + extra) * board["square_mm"],
                    (board["rows"] + extra) * board["square_mm"],
                )
                data = base64.b64encode(png).decode()
                svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width + 20}mm" height="{height + 20}mm" viewBox="0 0 {width + 20} {height + 20}"><rect width="100%" height="100%" fill="white"/><image x="10" y="10" width="{width}" height="{height}" href="data:image/png;base64,{data}" style="image-rendering:pixelated"/></svg>'
                self.send(svg.encode(), content_type="image/svg+xml")
            elif route.path == "/api/health":
                self.send({"ok": True})
            else:
                self.static("calibration")
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            cv2.error,
        ) as exc:
            self.fail(exc)

    def do_POST(self):
        try:
            body = read_json(self)
            route = urlsplit(self.path).path
            store, rig = self.server.store, self.server.rig
            # Serialize hardware operations and their session snapshots, so a
            # second tab cannot switch cameras between pose and image reads.
            with self.server.operations:
                if route == "/api/open":
                    result = rig.open(
                        body.get("camera"), body.get("serial"), body.get("side")
                    )
                elif route == "/api/close":
                    rig.close()
                    result = rig.status()
                elif route == "/api/create":
                    status = rig.status()
                    if not status["open"]:
                        raise ValueError(
                            "请先打开所选相机，读取与视频流对应的 SDK 内参"
                        )
                    result = store.create(
                        body.get("job"),
                        status["camera"],
                        body.get("board", {}),
                        body.get("name"),
                        body.get("mount_id"),
                    )
                elif route == "/api/detect":
                    session = store.load(body.get("id"))
                    rgb, _, camera = rig.read()
                    if (
                        camera["serial"] != session["camera"]["serial"]
                        or camera["role"] != session["camera"]["role"]
                    ):
                        raise ValueError("当前相机与所选记录不一致")
                    objects, pixels, ids, overlay = detect(
                        rgb,
                        session["board"],
                        boolean(body.get("reverse", False), "角点反转"),
                        body.get("rotation_quarters", 0),
                    )
                    result = dict(
                        count=len(pixels),
                        image="data:image/jpeg;base64,"
                        + base64.b64encode(encode_image(overlay)).decode(),
                    )
                elif route == "/api/capture":
                    session = store.load(body.get("id"))
                    job = JOBS[session["job"]]
                    manual = body.get("pose")
                    use_sdk = job["side"] is not None and manual is None
                    rgb, metadata, camera = rig.read(with_pose=use_sdk)
                    if use_sdk and camera["pose_side"] != job["side"]:
                        raise ValueError("当前位姿来源与任务的左/右臂不一致")
                    if body.get("still_confirmed") is not True:
                        raise ValueError("请在机械臂和标定板静止后确认再采样")
                    pose = metadata["pose"] if use_sdk else manual
                    metadata["pose_source"] = (
                        "pyAgxArm_follower_flange"
                        if use_sdk
                        else "manual"
                        if manual is not None
                        else None
                    )
                    result = store.add_sample(
                        session["id"],
                        rgb,
                        metadata,
                        camera,
                        pose,
                        boolean(body.get("reverse", False), "角点反转"),
                        body.get("rotation_quarters", 0),
                    )
                elif route == "/api/remove-sample":
                    result = store.remove_sample(body.get("id"), body.get("sample"))
                elif route == "/api/use-intrinsics":
                    result = store.use_intrinsics(body.get("id"), body.get("source"))
                elif route == "/api/solve":
                    result = store.solve(body.get("id"))
                elif route == "/api/export":
                    result = store.export(
                        body.get("ids"), body.get("accept_warnings") is True
                    )
                else:
                    raise ValueError("未知操作")
            self.send(result)
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            cv2.error,
        ) as exc:
            self.fail(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = False
    server.store = CalibrationStore(ROOT / "datasets/calibration")
    server.rig = CalibrationRig()
    server.operations = threading.RLock()

    # Workbench sends SIGTERM to owned children. Close SDK and lease on exit.
    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    print(f"CameraCalibration: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.rig.close()


if __name__ == "__main__":
    main()
