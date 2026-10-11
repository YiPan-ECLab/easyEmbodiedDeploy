"""HDF5 inspection and non-destructive temporal fusion editor."""

from __future__ import annotations

import json
import mimetypes
import threading
from collections import OrderedDict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import h5py
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from easy_embodied_deploy.http import LocalHandlerMixin, read_json
from easy_embodied_deploy import temporal_fusion


STATIC_ROOT = Path(__file__).resolve().parent.parent / "static"
HDF5_SUFFIXES = {".hdf5", ".h5"}
IMAGE_CACHE_LIMIT = 240


def _plain(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _attrs(obj: h5py.Group | h5py.Dataset | h5py.File) -> dict[str, Any]:
    return {str(key): _plain(value) for key, value in obj.attrs.items()}


def _safe_hdf5_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.suffix.lower() not in HDF5_SUFFIXES:
        raise ValueError("只支持 .hdf5 和 .h5 文件")
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    return path


def _demo_names(handle: h5py.File) -> list[str]:
    if "data" in handle and isinstance(handle["data"], h5py.Group):
        names = [
            name
            for name, value in handle["data"].items()
            if isinstance(value, h5py.Group)
        ]
        if names:
            return sorted(names)
    return ["/"]


def _demo_group(handle: h5py.File, name: str | None) -> tuple[str, h5py.Group]:
    names = _demo_names(handle)
    selected = name if name in names else names[0]
    if selected == "/":
        return selected, handle
    return selected, handle[f"data/{selected}"]


def _frame_count(demo: h5py.Group) -> int:
    if "num_samples" in demo.attrs:
        return int(demo.attrs["num_samples"])
    for path in ("actions", "timestamps", "states"):
        if path in demo and isinstance(demo[path], h5py.Dataset):
            return int(demo[path].shape[0])
    if "obs" in demo:
        for value in demo["obs"].values():
            if isinstance(value, h5py.Dataset) and value.ndim:
                return int(value.shape[0])
    return 0


def _require_direct_schema(handle: h5py.File, demo: h5py.Group) -> None:
    if handle.attrs.get("format") != "pyagxarm_direct_v1":
        raise ValueError("不是 PiperRecorder 的 pyagxarm_direct_v1 文件")
    if bool(handle.attrs.get("normalized", True)):
        raise ValueError("文件被标记为归一化数据，Viewer 仅支持直接控制量")
    required = {
        "actions", "states", "timestamps", "camera_timestamps",
        "action_source_timestamps", "trajectory_phase", "rewards", "dones", "obs",
    }
    missing = required.difference(demo.keys())
    if missing:
        raise ValueError(f"缺少新版数据集: {', '.join(sorted(missing))}")
    frames = _frame_count(demo)
    if demo["actions"].shape != (frames, 14) or demo["states"].shape != (frames, 14):
        raise ValueError("新版 actions 和 states 必须为 (T, 14)")


def _camera_datasets(demo: h5py.Group) -> list[dict[str, Any]]:
    if "obs" not in demo or not isinstance(demo["obs"], h5py.Group):
        return []
    cameras = []
    for name, value in demo["obs"].items():
        if not isinstance(value, h5py.Dataset) or value.ndim != 4:
            continue
        if value.shape[-1] not in (3, 4):
            continue
        label = name.removesuffix("_rgb").replace("_", " ")
        cameras.append(
            {
                "name": name,
                "label": label,
                "path": f"obs/{name}",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        )
    preferred = {"agentview": 0, "agentview_rgb": 0, "left_eye_in_hand": 1,
                 "left_eye_in_hand_rgb": 1, "right_eye_in_hand": 2,
                 "right_eye_in_hand_rgb": 2}
    return sorted(cameras, key=lambda item: (preferred.get(item["name"], 99), item["name"]))


def _instruction(demo: h5py.Group) -> str:
    for key in ("language", "language_instruction", "instruction"):
        if key in demo.attrs:
            return str(_plain(demo.attrs[key]))
    return ""


def _file_summary(path: Path) -> dict[str, Any]:
    stat = path.stat()
    with h5py.File(path, "r") as handle:
        demos = _demo_names(handle)
        counts = []
        first_demo = None
        for name in demos:
            _, demo = _demo_group(handle, name)
            if first_demo is None:
                first_demo = demo
            counts.append(_frame_count(demo))
        assert first_demo is not None
        _require_direct_schema(handle, first_demo)
        attrs = {**_attrs(handle), **_attrs(first_demo)}
        representation = str(
            attrs.get("actions_representation")
            or attrs.get("format")
            or attrs.get("representation")
            or "unknown"
        )
        return {
            "path": str(path),
            "name": path.name,
            "relative_path": path.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
            "demo_count": len(demos),
            "demos": demos,
            "frames": sum(counts),
            "first_demo_frames": counts[0],
            "cameras": len(_camera_datasets(first_demo)),
            "instruction": _instruction(first_demo),
            "representation": representation,
            "normalized": bool(attrs.get("normalized", False)),
        }


def scan_directory(directory: Path) -> dict[str, Any]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"目录不存在: {directory}")
    paths = sorted(
        (path for path in directory.rglob("*") if path.suffix.lower() in HDF5_SUFFIXES),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if len(paths) > 2000:
        raise ValueError("目录内 HDF5 文件超过 2000 个，请指定更具体的目录")
    files = []
    errors = []
    for path in paths:
        try:
            item = _file_summary(path)
            item["relative_path"] = str(path.relative_to(directory))
            files.append(item)
        except Exception as exc:  # Keep valid episodes browsable if one file is corrupt.
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "directory": str(directory),
        "files": files,
        "errors": errors,
        "count": len(files),
        "total_frames": sum(item["frames"] for item in files),
        "total_size": sum(item["size"] for item in files),
    }


def _dataset_array(demo: h5py.Group, path: str) -> Any:
    if path not in demo or not isinstance(demo[path], h5py.Dataset):
        return None
    value = np.asarray(demo[path])
    if value.dtype.kind == "f":
        result = value.astype(object)
        result[~np.isfinite(value)] = None
        return result.tolist()
    return _plain(value)


def _schema(handle: h5py.File) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            items.append(
                {
                    "path": f"/{name}",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": _attrs(obj),
                }
            )

    handle.visititems(visit)
    return items


def read_episode(path: Path, requested_demo: str | None) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        demo_name, demo = _demo_group(handle, requested_demo)
        _require_direct_schema(handle, demo)
        demos = _demo_names(handle)
        cameras = _camera_datasets(demo)
        frame_count = _frame_count(demo)
        dataset_paths = (
            "timestamps",
            "camera_timestamps",
            "trajectory_phase",
            "actions",
            "states",
            "obs/left_joint_states",
            "obs/right_joint_states",
            "obs/left_gripper_states",
            "obs/right_gripper_states",
            "obs/left_ee_states",
            "obs/right_ee_states",
            "obs/left_gripper_force",
            "obs/right_gripper_force",
        )
        series = {
            name.replace("/", "__"): _dataset_array(demo, name)
            for name in dataset_paths
            if name in demo
        }
        action_attrs = _attrs(demo["actions"]) if "actions" in demo else {}
        timestamps = series.get("timestamps") or []
        duration = 0.0
        if len(timestamps) > 1:
            duration = float(timestamps[-1]) - float(timestamps[0])
        elif frame_count > 1:
            env_args = {}
            raw_env = handle.get("data", {}).attrs.get("env_args", "") if "data" in handle else ""
            try:
                env_args = json.loads(_plain(raw_env)) if raw_env else {}
            except (TypeError, ValueError):
                pass
            frequency = env_args.get("env_kwargs", {}).get("control_freq", 0)
            if frequency:
                duration = (frame_count - 1) / float(frequency)
        root_attrs = _attrs(handle)
        demo_attrs = _attrs(demo)
        representation = str(
            action_attrs.get("representation")
            or demo_attrs.get("actions_representation")
            or root_attrs.get("format")
            or "unknown"
        )
        return {
            "path": str(path),
            "name": path.name,
            "demo": demo_name,
            "demos": demos,
            "frames": frame_count,
            "duration": duration,
            "instruction": _instruction(demo),
            "representation": representation,
            "normalized": bool(action_attrs.get("normalized", root_attrs.get("normalized", False))),
            "action_units": action_attrs.get("units") or (
                "rad x6,m per arm" if "direct" in representation else "dimensionless"
            ),
            "root_attrs": root_attrs,
            "demo_attrs": demo_attrs,
            "action_attrs": action_attrs,
            "cameras": cameras,
            "series": series,
            "schema": _schema(handle),
        }


class ImageCache:
    def __init__(self, limit: int = IMAGE_CACHE_LIMIT):
        self.limit = limit
        self.values: OrderedDict[tuple[Any, ...], bytes] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: tuple[Any, ...]) -> bytes | None:
        with self.lock:
            value = self.values.get(key)
            if value is not None:
                self.values.move_to_end(key)
            return value

    def put(self, key: tuple[Any, ...], value: bytes) -> None:
        with self.lock:
            self.values[key] = value
            self.values.move_to_end(key)
            while len(self.values) > self.limit:
                self.values.popitem(last=False)


IMAGE_CACHE = ImageCache()


class FrameBlockCache:
    """Read temporal HDF5 chunks once, without retaining open recording files.

    Older recordings split each image over many chunks spanning multiple frames.
    Reading these frames separately repeatedly decompresses the same data.
    Both the total cache and each read are bounded, including for large images.
    """

    def __init__(self, limit: int = 128 * 1024**2, block_limit: int = 32 * 1024**2):
        self.limit = limit
        self.block_limit = min(block_limit, limit)
        self.values: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self.size = 0
        self.lock = threading.Lock()

    def get(self, identity: tuple[Any, ...], path: Path, demo_name: str | None,
            camera: str, index: int) -> np.ndarray:
        # Coalesce concurrent misses. h5py itself serializes HDF5 operations;
        # JPEG encoding and HTTP delivery take place outside this lock.
        with self.lock:
            for key, block in self.values.items():
                if key[:-1] == identity and key[-1] <= index < key[-1] + len(block):
                    self.values.move_to_end(key)
                    return block[index - key[-1]]
            with h5py.File(path, "r") as handle:
                _, demo = _demo_group(handle, demo_name)
                _require_direct_schema(handle, demo)
                available = {item["name"]: item["path"] for item in _camera_datasets(demo)}
                if camera not in available:
                    raise KeyError(f"相机数据集不存在: {camera}")
                dataset = demo[available[camera]]
                if not len(dataset):
                    raise IndexError("相机数据集为空")
                index = min(max(index, 0), len(dataset) - 1)
                frame_bytes = int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize
                chunk_frames = dataset.chunks[0] if dataset.chunks else 16
                count = max(1, min(chunk_frames, self.block_limit // max(1, frame_bytes)))
                start = index // count * count
                block = np.asarray(dataset[start:min(start + count, len(dataset))])
            if block.nbytes <= self.limit:
                key = (*identity, start)
                previous = self.values.pop(key, None)
                if previous is not None:
                    self.size -= previous.nbytes
                while self.values and self.size + block.nbytes > self.limit:
                    _, removed = self.values.popitem(last=False)
                    self.size -= removed.nbytes
                self.values[key] = block
                self.size += block.nbytes
            return block[index - start]


FRAME_BLOCK_CACHE = FrameBlockCache()


def read_frame(path: Path, demo_name: str | None, camera: str, index: int) -> bytes:
    stat = path.stat()
    identity = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, demo_name, camera)
    cache_key = (*identity, index)
    cached = IMAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rgb = FRAME_BLOCK_CACHE.get(identity, path, demo_name, camera, index)
    conversion = cv2.COLOR_RGBA2BGR if rgb.shape[-1] == 4 else cv2.COLOR_RGB2BGR
    bgr = cv2.cvtColor(rgb, conversion)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError(f"无法编码 {demo_name}/{camera} 第 {index} 帧")
    value = encoded.tobytes()
    IMAGE_CACHE.put(cache_key, value)
    return value


class ViewerHandler(LocalHandlerMixin, BaseHTTPRequestHandler):
    server_version = "PiperViewer/1.0"

    @property
    def default_directory(self) -> Path:
        return self.server.default_directory  # type: ignore[attr-defined]

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(_plain(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc: Exception) -> None:
        status = HTTPStatus.NOT_FOUND if isinstance(exc, (FileNotFoundError, KeyError)) else HTTPStatus.BAD_REQUEST
        self._json({"error": str(exc), "type": type(exc).__name__}, status)

    def _static(self, url_path: str) -> None:
        relative = "index.html" if url_path in ("", "/") else url_path.lstrip("/")
        path = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT.resolve() not in path.parents and path != STATIC_ROOT.resolve():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") or content_type == "application/javascript" else content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/config":
                self._json({"default_directory": str(self.default_directory)})
                return
            if parsed.path == "/api/files":
                raw = query.get("directory", [str(self.default_directory)])[0]
                self._json(scan_directory(Path(raw)))
                return
            if parsed.path == "/api/episode":
                path = _safe_hdf5_path(query.get("path", [""])[0])
                demo = query.get("demo", [None])[0]
                self._json(read_episode(path, demo))
                return
            if parsed.path == "/api/fusion":
                path = _safe_hdf5_path(query.get("path", [""])[0])
                with h5py.File(path, "r") as handle:
                    _, demo = _demo_group(handle, query.get("demo", [None])[0])
                    _require_direct_schema(handle, demo)
                    name = demo.name
                self._json(temporal_fusion.load_plan(path, name))
                return
            if parsed.path == "/api/frame":
                path = _safe_hdf5_path(query.get("path", [""])[0])
                demo = query.get("demo", [None])[0]
                camera = query.get("camera", [""])[0]
                index = int(query.get("index", ["0"])[0])
                body = read_frame(path, demo, camera, index)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(body)
                return
            self._static(parsed.path)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Browsers may cancel a stale image request while scrubbing or
            # closing the page. The response is no longer writable.
            return
        except Exception as exc:  # Return useful API errors without stopping the server.
            if parsed.path.startswith("/api/"):
                self._error(exc)
            else:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        try:
            route = urlparse(self.path).path
            if route not in ("/api/fusion/preview", "/api/fusion/save", "/api/fusion/export"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = read_json(self)
            path = _safe_hdf5_path(body["path"])
            with h5py.File(path, "r") as handle:
                _, demo = _demo_group(handle, body.get("demo"))
                _require_direct_schema(handle, demo)
                name = demo.name
            plan = body["plan"]
            if route.endswith("/export"):
                result = temporal_fusion.export_episode(path, name, plan, body.get("output_directory"))
            elif route.endswith("/save"):
                result = temporal_fusion.save_plan(path, name, plan)
            else:
                result = temporal_fusion.preview(path, name, plan)
            self._json(result)
        except Exception as exc:
            self._error(exc)

    def log_message(self, fmt: str, *args: Any) -> None:
        if args and str(args[1]).startswith("4"):
            super().log_message(fmt, *args)


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], default_directory: Path):
        self.default_directory = default_directory.expanduser().resolve()
        super().__init__(address, ViewerHandler)


def run_server(host: str, port: int, directory: Path) -> int:
    server = ViewerServer((host, port), directory)
    print(f"Piper HDF5 Viewer: http://{host}:{server.server_address[1]}")
    print(f"初始目录: {server.default_directory}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer 已停止")
    finally:
        server.server_close()
    return 0
