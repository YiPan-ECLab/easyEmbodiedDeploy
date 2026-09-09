"""Orbbec RGB capture by device serial number."""

import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass

import cv2
import numpy as np

CAMERA_NAMES = ("agentview", "left_eye_in_hand", "right_eye_in_hand")


def resize_with_padding(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize the full image to fit, then add symmetric black padding."""
    target_width, target_height = size
    height, width = rgb.shape[:2]
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
        from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline

        pipeline = Pipeline(device)
        sensor_type = OBSensorType.COLOR_SENSOR
        # Gemini 305 exposes its stereo color cameras as LEFT/RIGHT_COLOR_SENSOR
        # instead of the single COLOR_SENSOR used by RGB cameras such as 335L.
        if self.name in ("left_eye_in_hand", "right_eye_in_hand"):
            try:
                pipeline.get_stream_profile_list(sensor_type)
            except Exception:
                sensor_type = OBSensorType.LEFT_COLOR_SENSOR
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
                frames = self._pipeline.wait_for_frames(500)
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
        return frame

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._pipeline is not None:
            self._pipeline.stop()
        self._thread = None
        self._pipeline = None


def open_cameras(
    serials: dict[str, str], resources: ExitStack, **stream
) -> dict[str, OrbbecCamera]:
    from pyorbbecsdk import Context, OBLogLevel

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
