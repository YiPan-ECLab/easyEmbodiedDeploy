"""Persistent calibration sessions and an explicitly opened, read-only rig."""

import json
import hashlib
import sys
import threading
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from .calibration_math import (
    JOBS,
    board_config,
    detect,
    flange_transform,
    solve_extrinsics,
    solve_intrinsics,
    transform_error,
    base_transform,
    inverse,
    validate_observation,
    compare_board_observations,
)
from .module_http import ROOT
from .resources import hardware_lease

DEFAULT_SERIALS = dict(
    agentview="CP2AB530005P",
    left_eye_in_hand="CV2L360000KH",
    right_eye_in_hand="CV2L3600003K",
)
CAMERAS = tuple(DEFAULT_SERIALS)
SAMPLING_LIMITS = dict(window_s=.35, max_age_s=.25, max_skew_s=.10,
                       translation_mm=2.0, rotation_deg=.5, corner_rms_px=2.0)


def check_sampling_window(frames, board, reverse=False, rotation_quarters=0):
    """Check actual feedback and board observations; never infer stillness from a checkbox."""
    if len({m["camera_received_monotonic_s"] for _, m in frames}) < 3:
        raise ValueError("采样窗口内不足三张新图像，请检查相机是否冻结")
    points = []
    for rgb, metadata in (frames[0], frames[len(frames)//2], frames[-1]):
        _, pixels, ids, _ = detect(rgb, board, reverse, rotation_quarters)
        points.append(dict(zip(ids, pixels)))
    drift = 0.0
    for i, a in enumerate(points):
        for b in points[i+1:]:
            common = sorted(a.keys() & b.keys())
            if len(common) < 6:
                raise ValueError("静止检查中共同可见角点不足，请保持板面可见且静止")
            drift = max(drift, float(np.sqrt(np.mean([
                np.sum((a[k]-b[k])**2) for k in common]))))
    quality = dict(limits=dict(SAMPLING_LIMITS), image_count=len(frames),
                   window_s=frames[-1][1]["sample_monotonic_s"]-frames[0][1]["sample_monotonic_s"],
                   corner_drift_rms_px=drift, timing_basis="host_receive_not_exposure",
                   pose_verified=False)
    if drift > SAMPLING_LIMITS["corner_rms_px"]:
        raise ValueError(f"标定板图像仍在移动（角点漂移 {drift:.2f} px），请静止后重试")
    poses = [m for _, m in frames if m.get("pose") is not None]
    if poses:
        if len({m["pose_sdk_timestamp_s"] for m in poses}) < 3:
            raise ValueError("采样窗口内法兰反馈未持续更新，请检查机械臂反馈")
        matrices = [flange_transform(m["pose"]) for m in poses]
        errors = [transform_error(a, b) for i, a in enumerate(matrices) for b in matrices[i+1:]]
        translation = max(e["translation_mm"] for e in errors)
        rotation = max(e["rotation_deg"] for e in errors)
        if translation > SAMPLING_LIMITS["translation_mm"] or rotation > SAMPLING_LIMITS["rotation_deg"]:
            raise ValueError(f"法兰仍在移动（{translation:.2f} mm / {rotation:.2f}°），请静止后重试")
        quality.update(pose_verified=True, translation_span_mm=translation, rotation_span_deg=rotation,
                       max_pose_age_s=max(m["pose_age_s"] for m in poses),
                       max_image_pose_skew_s=max(m["image_pose_skew_s"] for m in poses))
    quality["max_image_age_s"] = max(m["image_age_s"] for _, m in frames)
    return quality


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
            monotonic_before = time.monotonic()
            wall = time.time()
            monotonic_after = time.monotonic()
            now = (monotonic_before + monotonic_after) / 2
            image_age = now - frame.timestamp
            pose_age = None if timestamp is None else wall - timestamp
            skew = None if pose_age is None else abs(now - pose_age - frame.timestamp)
            if with_pose:
                if not np.isfinite(pose_age) or not -.02 <= pose_age <= SAMPLING_LIMITS["max_age_s"]:
                    raise ValueError("法兰反馈已过期或时间戳异常，请等待新反馈；不会使用缓存位姿采样")
                if skew > SAMPLING_LIMITS["max_skew_s"]:
                    raise ValueError(f"图像与法兰反馈接收时间差 {skew*1000:.0f} ms 超过 100 ms，请静止后重试")
            return (
                frame.rgb,
                dict(
                    camera_received_monotonic_s=frame.timestamp,
                    sample_wall_time_s=wall,
                    sample_monotonic_s=now,
                    pose_sdk_timestamp_s=timestamp,
                    pose=pose,
                    image_age_s=image_age,
                    pose_age_s=pose_age,
                    image_pose_skew_s=skew,
                    clock_mapping_uncertainty_s=monotonic_after-monotonic_before,
                ),
                dict(self.metadata),
            )

    def capture(self, with_pose=False):
        with self.lock:
            frames = []
            started = time.monotonic()
            while True:
                rgb, metadata, camera = self.read(with_pose)
                age = metadata["image_age_s"]
                if not np.isfinite(age) or not -.02 <= age <= SAMPLING_LIMITS["max_age_s"]:
                    raise ValueError("图像已过期或时间戳异常，请等待新帧")
                frames.append((rgb.copy(), metadata))
                if time.monotonic()-started >= SAMPLING_LIMITS["window_s"]:
                    return frames[-1][0], dict(metadata), camera, frames
                time.sleep(.05)

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
        session = json.loads(path.read_text())
        session.setdefault("validation_samples", [])
        return session

    @staticmethod
    def solution_id(session):
        result = session.get("result")
        if not result:
            return None
        return result.get("solution_id") or hashlib.sha256(
            json.dumps(result, sort_keys=True).encode()).hexdigest()

    def validation_summary(self, session):
        key = self.solution_id(session)
        current = [s for s in session.get("validation_samples", [])
                   if key is not None and s.get("validation", {}).get("solution_id") == key]
        reports = [s["validation"] for s in current if s.get("purpose") != "dual_validation"]
        return dict(status=("unvalidated" if not reports else "passed" if all(
                        r["within_thresholds"] for r in reports) else "needs_review"),
                    sample_count=len(reports),
                    dual_observation_count=sum(s.get("purpose") == "dual_validation" for s in current),
                    stale_sample_count=len(session.get("validation_samples", []))-len(current),
                    novel_pose_count=sum(bool(r.get("novel_pose")) for r in reports),
                    max_translation_mm=max((r.get("translation_mm", 0) for r in reports), default=None),
                    max_rotation_deg=max((r.get("rotation_deg", 0) for r in reports), default=None),
                    max_reprojection_rms_px=max((r["reprojection_rms_px"] for r in reports), default=None))

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
                validation_samples=[],
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
                            validation=self.validation_summary(session),
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
        purpose="calibration",
        scene_id="",
    ):
        with self.lock:
            session = self.load(identifier)
            if purpose not in ("calibration", "validation", "dual_validation"):
                raise ValueError("请选择求解样本、独立验证或双臂检查观测")
            if purpose != "calibration" and not session["result"]:
                raise ValueError("请先求解，再采集不参与求解的新验证样本")
            if purpose == "dual_validation" and session["job"] not in ("left_wrist", "right_wrist"):
                raise ValueError("双臂检查观测须使用左腕或右腕记录")
            if not isinstance(scene_id, str) or len(scene_id) > 120:
                raise ValueError("验证场景名称须为不超过 120 字的文本")
            if purpose == "dual_validation" and not scene_id.strip():
                raise ValueError("请填写双臂验证场景名称，方便匹配左右对同一固定板的观测")
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
                purpose=purpose,
                scene_id=scene_id.strip(),
            )
            if purpose != "calibration":
                camera_intrinsics = (session["result"]["intrinsics"] if session["job"] == "intrinsics"
                                     else session["camera"]["intrinsics"])
                report = validate_observation(sample, session["samples"], session["result"],
                                              camera_intrinsics, None if purpose == "dual_validation" else JOBS[session["job"]]["mounting"])
                sample["validation"] = dict(report, solution_id=self.solution_id(session), evaluated_at=time.time())
            directory = self.directory_for(identifier)
            (directory / f"{sample_id}.png").write_bytes(encode_image(rgb, ".png"))
            (directory / f"{sample_id}.jpg").write_bytes(encode_image(overlay))
            if purpose != "calibration":
                session["validation_samples"].append(sample)
            else:
                session["samples"].append(sample)
                session["result"] = None
            self.save(session)
            return session

    def remove_sample(self, identifier, sample_id):
        with self.lock:
            session = self.load(identifier)
            collection = next((key for key in ("samples", "validation_samples")
                               if any(s["id"] == sample_id for s in session[key])), None)
            if collection is None:
                raise ValueError("样本不存在")
            session[collection] = [s for s in session[collection] if s["id"] != sample_id]
            if collection == "samples":
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
            session["result"] = {**result, "solved_at": time.time(), "solution_id": uuid.uuid4().hex, "accepted": False}
            self.save(session)
            return session

    def compare_bases(self, identifiers, left_sample, right_sample, board_fixed=False):
        with self.lock:
            if board_fixed is not True:
                raise ValueError("请确认两次观测是同一块保持固定的标定板，且角点原点一致")
            sessions = self._export_sessions(identifiers, True)
            jobs = {s["job"]: s for s in sessions}
            required = ("left_wrist", "right_wrist", "global_left", "global_right")
            if any(job not in jobs for job in required):
                raise ValueError("双臂独立一致性检查需勾选左右腕和全局对左右 base 四项结果")
            samples = []
            for job, sample_id in (("left_wrist", left_sample), ("right_wrist", right_sample)):
                session = jobs[job]
                sample = next((s for s in session["validation_samples"] if s["id"] == sample_id), None)
                if sample is None or sample["validation"]["solution_id"] != self.solution_id(session):
                    raise ValueError("请选择当前求解结果下新采集的腕部验证样本；旧验证样本已失效")
                samples.append(sample)
            left, right = samples
            if jobs["left_wrist"]["board"] != jobs["right_wrist"]["board"]:
                raise ValueError("左右验证必须使用相同标定板参数")
            if not left.get("scene_id") or left["scene_id"] != right.get("scene_id"):
                raise ValueError("左右验证样本须填写同一场景名称，并观测同一块固定标定板")
            relation = base_transform(jobs["global_left"]["result"]["matrix"], jobs["global_right"]["result"]["matrix"])
            report = compare_board_observations(
                left["T_base_flange"], jobs["left_wrist"]["result"]["matrix"], left["validation"]["T_camera_board"],
                right["T_base_flange"], jobs["right_wrist"]["result"]["matrix"], right["validation"]["T_camera_board"], relation)
            report["max_reprojection_rms_px"] = max(s["validation"]["reprojection_rms_px"] for s in samples)
            report["thresholds"]["reprojection_rms_px"] = 2
            report["within_thresholds"] &= report["max_reprojection_rms_px"] <= 2
            report.update(schema="easy_embodied_dual_base_validation", id=uuid.uuid4().hex,
                          created_at=time.time(), scene_id=left["scene_id"],
                          source_sessions={job: jobs[job]["id"] for job in required},
                          source_solutions={job: self.solution_id(jobs[job]) for job in required},
                          source_samples=dict(left=left_sample, right=right_sample),
                          method="independent_wrist_observations_of_same_fixed_board",
                          T_left_base_right_base=relation.tolist())
            directory = self.directory_for(report["id"])
            directory.mkdir(parents=True)
            atomic_json(directory / "dual_base_validation.json", report)
            return report

    def _export_sessions(self, identifiers, accept_warnings):
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
            if (result["warnings"] or self.validation_summary(session)["status"] == "needs_review") and not accept_warnings:
                raise ValueError("结果包含质量提示或独立验证超差，请检查后勾选接受提示再导出")
        extrinsics = [s for s in sessions if s["job"] != "intrinsics"]
        if len({s["mount_id"] for s in extrinsics}) > 1:
            raise ValueError("安装版本不一致，不可合并为同一标定包")
        globals_ = [s for s in sessions if s["job"].startswith("global_")]
        if len(globals_) == 2 and any(globals_[0]["camera"][k] != globals_[1]["camera"][k]
                                    for k in ("serial", "sensor", "width", "height")):
            raise ValueError("左右 base 的全局相机标定必须使用同一台相机、同一视频配置")
        return sessions

    def export(self, identifiers, accept_warnings=False):
        with self.lock:
            sessions = self._export_sessions(identifiers, accept_warnings)
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
            jobs = {s["job"]: s for s in sessions}
            bundle["derived_transforms"] = {}
            if "global_left" in jobs and "global_right" in jobs:
                relation = base_transform(jobs["global_left"]["result"]["matrix"], jobs["global_right"]["result"]["matrix"])
                for name, matrix, direction, method in (
                    ("T_left_base_right_base", relation, "right_base -> left_base",
                     "T_left_base_camera @ inverse(T_right_base_camera)"),
                    ("T_right_base_left_base", inverse(relation), "left_base -> right_base",
                     "T_right_base_camera @ inverse(T_left_base_camera)"),
                ):
                    bundle["derived_transforms"][name] = dict(matrix=matrix.tolist(), units="m", maps=direction,
                        mount_id=jobs["global_left"]["mount_id"],
                        source_sessions=[jobs[j]["id"] for j in ("global_left", "global_right")],
                        method=method,
                        independently_validated=False)
            current_ids = {s["job"]: s["id"] for s in sessions}
            current_solutions = {s["job"]: self.solution_id(s) for s in sessions}
            checks = []
            for path in self.directory.glob("*/dual_base_validation.json"):
                if path.is_symlink() or path.parent.is_symlink():
                    continue
                report = json.loads(path.read_text())
                if all(current_ids.get(j) == i for j, i in report["source_sessions"].items()) and all(
                        current_solutions.get(j) == i for j, i in report["source_solutions"].items()):
                    # Deleting an observation also retires reports made from it.
                    if all(any(s["id"] == report["source_samples"][side]
                               for s in jobs[f"{side}_wrist"]["validation_samples"])
                           for side in ("left", "right")):
                        checks.append(report)
            bundle["dual_base_validation"] = sorted(checks, key=lambda r: r["created_at"])
            if any(not report["within_thresholds"] for report in checks) and not accept_warnings:
                raise ValueError("双臂独立一致性检查存在超差结果，请检查后勾选接受提示再导出")
            for value in bundle["derived_transforms"].values():
                value["independently_validated"] = bool(checks) and all(r["within_thresholds"] for r in checks)
            for calibration, source in zip(bundle["calibrations"], sessions):
                calibration["validation_summary"] = self.validation_summary(source)
                calibration["validation_samples"] = source["validation_samples"]
            for calibration in bundle["calibrations"]:
                calibration["result"] = {
                    **calibration["result"],
                    "accepted": True,
                    "warnings_accepted": bool(calibration["result"]["warnings"] or calibration["validation_summary"]["status"] == "needs_review"),
                }
            export_id = uuid.uuid4().hex
            folder = self.directory_for(export_id)
            folder.mkdir(parents=True)
            destination = folder / "calibration_bundle.json"
            atomic_json(destination, bundle)
            return dict(path=str(destination), bundle=bundle, export_id=export_id)
