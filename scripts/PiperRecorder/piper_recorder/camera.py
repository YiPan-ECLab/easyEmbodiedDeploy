"""Orbbec RGB capture by device serial number."""

import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass

import cv2
import numpy as np

from .camera_errors import camera_start_error

CAMERA_NAMES = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
FRAME_WAIT_TIMEOUT_MS = 100


def resize_with_padding(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize the full image to fit, then add symmetric black padding."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("相机帧必须是 HxWx3 的 uint8 RGB 图像")
    target_width, target_height = size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("目标图像尺寸必须为正数")
    height, width = rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("相机帧尺寸不能为空")
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    image[top : top + resized_height, left : left + resized_width] = resized
    return image


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    timestamp: float


class OrbbecCamera:
    def __init__(
        self,
        name: str,
        serial: str,
        context,
        *,
        width: int,
        height: int,
        fps: int,
        image_size: tuple[int, int],
        max_age_s: float,
    ):
        self.name = name
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.image_size = image_size
        self.max_age_s = max_age_s
        self._context = context
        self._pipeline = None
        self._sensor_type = None
        self._latest: CameraFrame | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, device) -> None:
        try:
            self._start(device)
        except Exception as exc:
            failure = camera_start_error(self.name, self.serial, exc)
            try:
                self.close()
            except Exception as cleanup:
                failure = RuntimeError(f'{failure}；启动失败后的相机清理异常：{cleanup}')
            raise failure from exc

    def _start(self, device) -> None:
        from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline

        if device is None:
            raise RuntimeError(f"相机 {self.name} ({self.serial}) 未找到设备")
        self._stop.clear()
        self._error = None
        self._latest = None
        pipeline = Pipeline(device)
        self._pipeline = pipeline
        sensor_type = OBSensorType.COLOR_SENSOR
        # Stereo models expose separate LEFT/RIGHT_COLOR_SENSOR streams. A
        # regular RGB camera falls back to COLOR_SENSOR when those are absent.
        if self.name in ("left_eye_in_hand", "right_eye_in_hand"):
            preferred_sensor = (
                OBSensorType.LEFT_COLOR_SENSOR
                if self.name == "left_eye_in_hand"
                else OBSensorType.RIGHT_COLOR_SENSOR
            )
            try:
                pipeline.get_stream_profile_list(preferred_sensor)
                sensor_type = preferred_sensor
            except Exception:
                sensor_type = OBSensorType.COLOR_SENSOR
        try:
            profiles = pipeline.get_stream_profile_list(sensor_type)
        except Exception as exc:
            # Some Orbbec stereo/depth-only models (for example Gemini 305)
            # enumerate as devices but do not expose a COLOR_SENSOR.
            model = "unknown"
            try:
                model = device.get_device_info().get_name()
            except Exception:
                pass
            raise RuntimeError(
                f"相机 {self.name} ({self.serial}, {model}) 没有可用的彩色传感器；"
                "请检查设备型号和 SDK 支持。"
            ) from exc
        profile = profiles.get_video_stream_profile(
            self.width, self.height, OBFormat.RGB, self.fps
        )
        config = Config()
        config.enable_stream(profile)
        self._sensor_type = sensor_type
        self._pipeline = pipeline
        pipeline.start(config)
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"orbbec-{self.name}",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            with self._lock:
                frame, error = self._latest, self._error
            if error is not None:
                raise RuntimeError(f"相机 {self.name} 启动失败") from error
            if frame is not None:
                print(f"相机 {self.name}: {self.serial} 已启动")
                return
            time.sleep(0.02)
        raise RuntimeError(f"相机 {self.name}: 6 秒内没有彩色帧")

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                # Keep the SDK wait bounded well below the recorder's 0.35 s
                # task-gap limit. A long SDK wait would hide a stalled camera
                # until several recorder samples had already been missed.
                frames = self._pipeline.wait_for_frames(FRAME_WAIT_TIMEOUT_MS)
                if frames is None:
                    color = None
                elif self._sensor_type.name == "LEFT_COLOR_SENSOR":
                    color = frames.get_left_color_frame()
                elif self._sensor_type.name == "RIGHT_COLOR_SENSOR":
                    color = frames.get_right_color_frame()
                else:
                    color = frames.get_color_frame()
                if color is None:
                    continue
                rgb = np.asarray(color.get_data(), dtype=np.uint8).reshape(
                    color.get_height(), color.get_width(), 3
                )
                image = resize_with_padding(rgb, self.image_size)
                with self._lock:
                    self._latest = CameraFrame(image, time.monotonic())
        except Exception as exc:  # noqa: BLE001 - relay SDK/thread failures to caller
            if not self._stop.is_set():
                with self._lock:
                    self._error = exc

    def read(self) -> CameraFrame:
        with self._lock:
            frame, error = self._latest, self._error
        if error is not None:
            raise RuntimeError(f"相机 {self.name} 彩色流停止") from error
        if frame is None or time.monotonic() - frame.timestamp > self.max_age_s:
            raise RuntimeError(f"相机 {self.name} 没有有效的新帧")
        # A recorder sample may legitimately reuse the latest camera frame
        # when the camera cadence is lower than the robot/control cadence.
        return frame

    def close(self) -> None:
        self._stop.set()
        error = None
        # Stop the pipeline first so wait_for_frames() wakes promptly.
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:
                error = exc
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError(f'相机 {self.name} 采集线程未退出，设备尚未完全释放')
        self._thread = None
        self._pipeline = None
        self._latest = self._error = None
        if error is not None:
            raise RuntimeError(f'相机 {self.name} 停止流失败：{error}') from error


def open_cameras(
    serials: dict[str, str], resources: ExitStack, **stream
) -> dict[str, OrbbecCamera]:
    from pyorbbecsdk import Context, OBLogLevel

    if set(serials) != set(CAMERA_NAMES) or len(set(serials.values())) != len(CAMERA_NAMES):
        raise ValueError(f"相机配置必须恰好包含 {CAMERA_NAMES} 且序列号不能重复")
    Context.set_logger_to_console(OBLogLevel.ERROR)
    Context.set_logger_to_file(OBLogLevel.NONE, "/tmp")
    context = Context()
    devices = context.query_devices()
    cameras = {}
    for name, serial in serials.items():
        camera = OrbbecCamera(name, serial, context, **stream)
        resources.callback(camera.close)
        camera.start(devices.get_device_by_serial_number(serial))
        cameras[name] = camera
    return cameras
