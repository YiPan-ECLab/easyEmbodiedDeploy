"""Persistent calibration sessions and an explicitly opened, read-only rig."""

import json
import sys
import threading
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import cv2

from .calibration_math import (
    JOBS,
    board_config,
    detect,
    flange_transform,
    solve_extrinsics,
    solve_intrinsics,
)
from .module_http import ROOT
from .resources import hardware_lease

DEFAULT_SERIALS = dict(
    agentview="CP2AB530005P",
    left_eye_in_hand="CV2L360000KH",
    right_eye_in_hand="CV2L3600003K",
)
CAMERAS = tuple(DEFAULT_SERIALS)


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def encode_image(rgb, extension=".jpg"):
    ok, data = cv2.imencode(extension, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("图像编码失败")
    return data.tobytes()


def sdk_intrinsics(profile):
    intrinsic, distortion = profile.get_intrinsic(), profile.get_distortion()
    raw = {
        key: float(getattr(distortion, key))
        for key in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
    }
    name = distortion.model.name
    if name == "NONE":
        model, coefficients = "brown", [0.0] * 5
    elif name in ("BROWN_CONRADY", "BROWN_CONRADY_K6"):
        model = "brown"
        coefficients = [
            raw[k] for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
        ]
        if name == "BROWN_CONRADY":
            coefficients = coefficients[:5]
    elif name == "KANNALA_BRANDT4":
        model, coefficients = "fisheye", [raw[k] for k in ("k1", "k2", "k3", "k4")]
    else:
        model, coefficients = "unsupported", []
    return dict(
        K=[[intrinsic.fx, 0, intrinsic.cx], [0, intrinsic.fy, intrinsic.cy], [0, 0, 1]],
        D=coefficients,
        model=model,
        sdk_model=name,
        sdk_coefficients=raw,
        source="orbbec_selected_video_profile",
        width=intrinsic.width,
        height=intrinsic.height,
    )


class CalibrationRig:
    """Reuse recorder capture, but never instantiate its motion controller."""

    def __init__(self):
        self.lock = threading.RLock()
        self.resources = None
        self.camera = None
        self.arm = None
        self.metadata = None

    def open(self, camera_name, serial, side=None):
        with self.lock:
            if self.resources:
                raise ValueError("请先关闭当前相机再切换")
            if (
                camera_name not in CAMERAS
                or not isinstance(serial, str)
                or not serial.strip()
            ):
                raise ValueError("请选择相机并填写序列号")
            if side not in (None, "left", "right"):
                raise ValueError("机械臂必须为 left 或 right")
            if str(ROOT / "scripts/PiperRecorder") not in sys.path:
                sys.path.insert(0, str(ROOT / "scripts/PiperRecorder"))
            from piper_recorder.camera import OrbbecCamera
            from pyorbbecsdk import Context, OBFormat, OBLogLevel

            with ExitStack() as resources:
                resources.enter_context(hardware_lease())
                Context.set_logger_to_console(OBLogLevel.ERROR)
                Context.set_logger_to_file(OBLogLevel.NONE, "/tmp")
                context = Context()
                device = context.query_devices().get_device_by_serial_number(
                    serial.strip()
                )
                camera = OrbbecCamera(
                    camera_name,
                    serial.strip(),
                    context,
                    width=640,
                    height=480,
                    fps=30,
                    image_size=(640, 480),
                    max_age_s=0.5,
                )
                resources.callback(camera.close)
                camera.start(device)
                profile = camera._pipeline.get_stream_profile_list(
                    camera._sensor_type
                ).get_video_stream_profile(640, 480, OBFormat.RGB, 30)
                metadata = dict(
                    role=camera_name,
                    serial=serial.strip(),
                    sensor=camera._sensor_type.name,
                    width=640,
                    height=480,
                    fps=30,
                    pixel_format="RGB",
                    intrinsics=sdk_intrinsics(profile),
                )
                if (
                    metadata["intrinsics"]["width"] != 640
                    or metadata["intrinsics"]["height"] != 480
                ):
                    raise ValueError("SDK 返回的内参与当前视频分辨率不一致")
                arm = None
                if side:
                    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
                    from pyAgxArm import PiperFW

                    arm = AgxArmFactory.create_arm(
                        create_agx_arm_config(
                            robot=ArmModel.PIPER,
                            firmeware_version=PiperFW.V189,
                            channel="piper-lo" if side == "left" else "piper-ro",
                        )
                    )
                    resources.callback(arm.disconnect)
                    arm.connect()
                self.camera, self.arm = camera, arm
                self.metadata = {**metadata, "pose_side": side}
                self.resources = resources.pop_all()
                return self.metadata

    def close(self):
        with self.lock:
            resources = self.resources
            self.resources = self.camera = self.arm = self.metadata = None
            if resources:
                resources.close()

    def read(self, with_pose=False):
        with self.lock:
            if self.camera is None:
                raise ValueError("请先打开相机")
            frame = self.camera.read()
            pose, timestamp = None, None
            if with_pose:
                if self.arm is None:
                    raise ValueError(
                        "当前未连接位姿读取，可关闭后勾选只读位姿，或手动输入位姿"
                    )
                message = self.arm.get_flange_pose()
                if message is None:
                    raise ValueError(
                        "SDK 尚未提供法兰位姿，请稍后采样；标定模块不会切换机械臂模式"
                    )
                pose, timestamp = list(message.msg), float(message.timestamp)
                flange_transform(pose)
            return (
                frame.rgb,
                dict(
                    camera_received_monotonic_s=frame.timestamp,
                    sample_wall_time_s=time.time(),
                    pose_sdk_timestamp_s=timestamp,
                    pose=pose,
                ),
                dict(self.metadata),
            )

    def status(self):
        with self.lock:
            return dict(open=self.resources is not None, camera=self.metadata)


class CalibrationStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.RLock()

    def directory_for(self, identifier):
        if (
            not isinstance(identifier, str)
            or len(identifier) != 32
            or any(c not in "0123456789abcdef" for c in identifier)
        ):
            raise ValueError("无效标定记录 ID")
        if self.directory.is_symlink():
            raise ValueError("标定目录不能是符号链接")
        path = self.directory / identifier
        if path.is_symlink():
            raise ValueError("标定记录不能是符号链接")
        return path

    def load(self, identifier):
        path = self.directory_for(identifier) / "session.json"
        if path.is_symlink():
            raise ValueError("标定记录不能是符号链接")
        return json.loads(path.read_text())

    def save(self, session):
        atomic_json(self.directory_for(session["id"]) / "session.json", session)

    def create(self, job, camera, board, name, mount_id):
        with self.lock:
            if job not in JOBS or camera["role"] not in CAMERAS:
                raise ValueError("未知标定任务或相机")
            if JOBS[job]["camera"] not in (None, camera["role"]):
                raise ValueError("所选相机与标定任务不匹配")
            if not isinstance(name, str) or not name.strip() or len(name) > 120:
                raise ValueError("请填写 1–120 字的标定名称")
            if (
                not isinstance(mount_id, str)
                or not mount_id.strip()
                or len(mount_id) > 120
            ):
                raise ValueError("请填写安装版本；相机或机械臂基座移动后须更换版本")
            board = board_config(board)
            identifier = uuid.uuid4().hex
            path = self.directory_for(identifier)
            path.mkdir(parents=True)
            session = dict(
                schema_version=1,
                id=identifier,
                name=name.strip(),
                job=job,
                mount_id=mount_id.strip(),
                camera=camera,
                board=board,
                created_at=time.time(),
                samples=[],
                result=None,
            )
            self.save(session)
            return session

    def listing(self):
        with self.lock:
            result = []
            if not self.directory.exists():
                return result
            for path in self.directory.iterdir():
                try:
                    session = self.load(path.name)
                    result.append(
                        {
                            key: session[key]
                            for key in (
                                "id",
                                "name",
                                "job",
                                "mount_id",
                                "created_at",
                                "camera",
                            )
                        }
                        | dict(
                            sample_count=len(session["samples"]),
                            solved=session["result"] is not None,
                        )
                    )
                except (OSError, ValueError, KeyError):
                    continue
            return sorted(result, key=lambda row: row["created_at"], reverse=True)

    def add_sample(
        self,
        identifier,
        rgb,
        metadata,
        camera,
        pose=None,
        reverse=False,
        rotation_quarters=0,
    ):
        with self.lock:
            session = self.load(identifier)
            for key in ("role", "serial", "sensor", "width", "height"):
                if session["camera"][key] != camera[key]:
                    raise ValueError("相机身份或视频配置与此记录不一致，请新建标定记录")
            if rgb.shape != (camera["height"], camera["width"], 3):
                raise ValueError("图像尺寸与内参不一致")
            objects, pixels, ids, overlay = detect(
                rgb, session["board"], reverse, rotation_quarters
            )
            if session["job"] != "intrinsics":
                matrix = flange_transform(pose)
            else:
                matrix = None
            sample_id = uuid.uuid4().hex
            sample = dict(
                id=sample_id,
                objects=objects.tolist(),
                pixels=pixels.tolist(),
                corner_ids=ids,
                reverse_corners=bool(reverse),
                rotation_quarters=rotation_quarters,
                T_base_flange=matrix.tolist() if matrix is not None else None,
                pose_xyz_rpy=pose,
                metadata=metadata,
            )
            directory = self.directory_for(identifier)
            (directory / f"{sample_id}.png").write_bytes(encode_image(rgb, ".png"))
            (directory / f"{sample_id}.jpg").write_bytes(encode_image(overlay))
            session["samples"].append(sample)
            session["result"] = None
            self.save(session)
            return session

    def remove_sample(self, identifier, sample_id):
        with self.lock:
            session = self.load(identifier)
            if sample_id not in [s["id"] for s in session["samples"]]:
                raise ValueError("样本不存在")
            session["samples"] = [s for s in session["samples"] if s["id"] != sample_id]
            session["result"] = None
            self.save(session)
            for suffix in (".jpg", ".png"):
                (self.directory_for(identifier) / (sample_id + suffix)).unlink(
                    missing_ok=True
                )
            return session

    def use_intrinsics(self, identifier, source_id):
        with self.lock:
            session, source = self.load(identifier), self.load(source_id)
            if source["job"] != "intrinsics" or not source["result"]:
                raise ValueError("请选择已求解的内参记录")
            for key in ("role", "serial", "sensor", "width", "height"):
                if source["camera"][key] != session["camera"][key]:
                    raise ValueError("内参记录的相机或分辨率不匹配")
            session["camera"]["intrinsics"] = {
                **source["result"]["intrinsics"],
                "session_id": source_id,
            }
            session["result"] = None
            self.save(session)
            return session

    def solve(self, identifier):
        with self.lock:
            session = self.load(identifier)
            if session["job"] == "intrinsics":
                result = solve_intrinsics(
                    session["samples"],
                    session["camera"]["width"],
                    session["camera"]["height"],
                )
            else:
                job = JOBS[session["job"]]
                result = solve_extrinsics(
                    session["samples"], session["camera"]["intrinsics"], job["mounting"]
                )
                result["transform"] = job["transform"]
            session["result"] = {**result, "solved_at": time.time(), "accepted": False}
            self.save(session)
            return session

    def export(self, identifiers, accept_warnings=False):
        with self.lock:
            if not isinstance(identifiers, list) or not 1 <= len(identifiers) <= 7:
                raise ValueError("请选择 1–7 项已求解的记录")
            sessions = [self.load(i) for i in identifiers]
            seen = set()
            for session in sessions:
                result = session["result"]
                key = (session["job"], session["camera"]["role"])
                if key in seen:
                    raise ValueError("同一任务不能导出两份结果")
                seen.add(key)
                if not result:
                    raise ValueError("请先求解所有选中的记录")
                if result["warnings"] and not accept_warnings:
                    raise ValueError("结果包含质量提示，请检查后勾选接受提示再导出")
            extrinsics = [s for s in sessions if s["job"] != "intrinsics"]
            if len({s["mount_id"] for s in extrinsics}) > 1:
                raise ValueError("安装版本不一致，不可合并为同一标定包")
            global_sessions = [s for s in sessions if s["job"].startswith("global_")]
            if len(global_sessions) == 2 and any(
                global_sessions[0]["camera"][k] != global_sessions[1]["camera"][k]
                for k in ("serial", "sensor", "width", "height")
            ):
                raise ValueError(
                    "左右 base 的全局相机标定必须使用同一台相机、同一视频配置"
                )
            bundle = dict(
                schema="easy_embodied_camera_calibration",
                schema_version=1,
                created_at=time.time(),
                convention="T_A_B maps B to A; meters; rotation matrices stored by rows",
                status="operator_exported_not_hardware_certified",
                training_integration=True,
                missing_jobs=[
                    j
                    for j in JOBS
                    if j != "intrinsics" and j not in {s["job"] for s in sessions}
                ],
                calibrations=[
                    {
                        k: s[k]
                        for k in (
                            "id",
                            "name",
                            "job",
                            "mount_id",
                            "camera",
                            "board",
                            "result",
                        )
                    }
                    for s in sessions
                ],
            )
            for calibration in bundle["calibrations"]:
                calibration["result"] = {
                    **calibration["result"],
                    "accepted": True,
                    "warnings_accepted": bool(calibration["result"]["warnings"]),
                }
            export_id = uuid.uuid4().hex
            folder = self.directory_for(export_id)
            folder.mkdir(parents=True)
            destination = folder / "calibration_bundle.json"
            atomic_json(destination, bundle)
            return dict(path=str(destination), bundle=bundle, export_id=export_id)
