"""Write direct pyAgxArm observations and actions to HDF5."""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from .camera import CAMERA_NAMES, CameraFrame
from .robot import (
    GRIPPER_LIMITS_M,
    HARDWARE_TOLERANCE_M,
    JOINT_LIMITS_RAD,
    SIDES,
    ArmSample,
)

TASK_PHASE = 0
HOME_PHASE = 1
ACTION_LAYOUT = (
    "left_j1_rad,left_j2_rad,left_j3_rad,left_j4_rad,left_j5_rad,left_j6_rad,"
    "left_gripper_m,right_j1_rad,right_j2_rad,right_j3_rad,right_j4_rad,"
    "right_j5_rad,right_j6_rad,right_gripper_m"
)
_JOINT_LIMITS_ARRAY = np.asarray(JOINT_LIMITS_RAD, dtype=np.float64)
JOINT_BOUNDARY_TOLERANCE_RAD = np.deg2rad(1.0)
GRIPPER_BOUNDARY_TOLERANCE_M = HARDWARE_TOLERANCE_M


@dataclass
class EpisodeData:
    recording_metadata: dict = field(default_factory=dict, kw_only=True)
    timestamps: list[float] = field(default_factory=list)
    phases: list[int] = field(default_factory=list)
    camera_timestamps: list[list[float]] = field(default_factory=list)
    scheduled_timestamps: list[float] = field(default_factory=list)
    capture_durations: list[float] = field(default_factory=list)
    camera_durations: list[float] = field(default_factory=list)
    robot_durations: list[float] = field(default_factory=list)
    images: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {name: [] for name in CAMERA_NAMES}
    )
    samples: dict[str, list[ArmSample]] = field(
        default_factory=lambda: {side: [] for side in SIDES}
    )

    def append(
        self,
        frames: dict[str, CameraFrame],
        samples: dict[str, ArmSample],
        phase: int,
        *,
        timestamp: float | None = None,
        scheduled_timestamp: float | None = None,
        capture_duration_s: float | None = None,
        camera_duration_s: float | None = None,
        robot_duration_s: float | None = None,
    ) -> None:
        self.timestamps.append(time.monotonic() if timestamp is None else float(timestamp))
        self.phases.append(phase)
        self.camera_timestamps.append([frames[name].timestamp for name in CAMERA_NAMES])
        if scheduled_timestamp is not None:
            self.scheduled_timestamps.append(float(scheduled_timestamp))
        if capture_duration_s is not None:
            self.capture_durations.append(float(capture_duration_s))
        if camera_duration_s is not None:
            self.camera_durations.append(float(camera_duration_s))
        if robot_duration_s is not None:
            self.robot_durations.append(float(robot_duration_s))
        for name in CAMERA_NAMES:
            self.images[name].append(frames[name].rgb)
        for side in SIDES:
            self.samples[side].append(samples[side])

    def __len__(self) -> int:
        return len(self.timestamps)


def _clip_side_values(
    values: np.ndarray, name: str
) -> tuple[np.ndarray, int]:
    lower = np.concatenate(
        (_JOINT_LIMITS_ARRAY[:, 0], [GRIPPER_LIMITS_M[0]])
    )
    upper = np.concatenate(
        (_JOINT_LIMITS_ARRAY[:, 1], [GRIPPER_LIMITS_M[1]])
    )
    tolerance = np.asarray(
        [JOINT_BOUNDARY_TOLERANCE_RAD] * 6
        + [GRIPPER_BOUNDARY_TOLERANCE_M],
        dtype=np.float64,
    )
    invalid = (values < lower - tolerance) | (values > upper + tolerance)
    if np.any(invalid):
        frame, axis = np.argwhere(invalid)[0]
        label = f"j{axis + 1}" if axis < 6 else "gripper"
        unit = "rad" if axis < 6 else "m"
        raise ValueError(
            f"{name} 超出允许截断范围: frame={frame}, {label}="
            f"{values[frame, axis]:.6f} {unit}, "
            f"边界=[{lower[axis]:.6f},{upper[axis]:.6f}]"
        )
    clipped = np.clip(values, lower, upper)
    return clipped, int(np.count_nonzero(clipped != values))


def _side_arrays(
    samples: list[ArmSample], side: str
) -> dict[str, np.ndarray | int]:
    joints = np.asarray([sample.joints for sample in samples], dtype=np.float64)
    poses = np.asarray([sample.flange_pose for sample in samples], dtype=np.float64)
    grippers = np.asarray(
        [[sample.gripper_width] for sample in samples], dtype=np.float64
    )
    forces = np.asarray(
        [[sample.gripper_force] for sample in samples], dtype=np.float64
    )
    actions = np.asarray(
        [(*sample.action.joints, sample.action.gripper_width) for sample in samples],
        dtype=np.float64,
    )
    states, state_clip_count = _clip_side_values(
        np.concatenate((joints, grippers), axis=1), f"{side} state"
    )
    actions, action_clip_count = _clip_side_values(actions, f"{side} action")
    joints = states[:, :6]
    grippers = states[:, 6:7]
    action_timestamps = np.asarray(
        [sample.action.timestamp for sample in samples], dtype=np.float64
    )
    return {
        "joints": joints,
        "poses": poses,
        "grippers": grippers,
        "forces": forces,
        "states": states,
        "actions": actions,
        "action_timestamps": action_timestamps,
        "clip_count": state_clip_count + action_clip_count,
    }


def _validate_episode(episode: EpisodeData) -> None:
    n = len(episode)
    if n < 2:
        raise ValueError("episode 至少需要两帧")
    if len(episode.phases) != n or len(episode.camera_timestamps) != n:
        raise ValueError("episode 的时间、阶段和相机记录长度不一致")
    if episode.scheduled_timestamps and len(episode.scheduled_timestamps) != n:
        raise ValueError("scheduled_timestamps 长度不一致")
    for name, values in (("capture_durations", episode.capture_durations),
                         ("camera_durations", episode.camera_durations),
                         ("robot_durations", episode.robot_durations)):
        if values and len(values) != n:
            raise ValueError(f"{name} 长度不一致")
    timestamps = np.asarray(episode.timestamps, dtype=np.float64)
    if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps 必须有限且严格递增")
    phases = np.asarray(episode.phases)
    if not np.isin(phases, (TASK_PHASE, HOME_PHASE)).all() or phases[0] != TASK_PHASE:
        raise ValueError("trajectory_phase 必须从 task(0) 开始且只含 0/1")
    home = np.flatnonzero(phases == HOME_PHASE)
    if len(home) and not np.all(phases[home[0] :] == HOME_PHASE):
        raise ValueError("trajectory_phase 必须是连续 task 前缀和 home 后缀")
    camera_timestamps = np.asarray(episode.camera_timestamps, dtype=np.float64)
    if camera_timestamps.shape != (n, len(CAMERA_NAMES)) or not np.isfinite(camera_timestamps).all():
        raise ValueError("camera_timestamps 必须是有限的 (T,3)")
    if episode.scheduled_timestamps:
        scheduled = np.asarray(episode.scheduled_timestamps, dtype=np.float64)
        if not np.isfinite(scheduled).all() or not np.all(np.diff(scheduled) > 0):
            raise ValueError("scheduled_timestamps 必须有限且严格递增")
    for name, values in (("capture_durations", episode.capture_durations),
                         ("camera_durations", episode.camera_durations),
                         ("robot_durations", episode.robot_durations)):
        if values:
            durations = np.asarray(values, dtype=np.float64)
            if not np.isfinite(durations).all() or np.any(durations < 0):
                raise ValueError(f"{name} 必须是有限的非负数")
    for camera in CAMERA_NAMES:
        if len(episode.images[camera]) != n:
            raise ValueError(f"{camera} 图像数量与 episode 不一致")
        images = episode.images[camera]
        shape = images[0].shape
        if len(shape) != 3 or shape[-1] != 3 or images[0].dtype != np.uint8:
            raise ValueError(f"{camera} 图像必须是 HxWx3 uint8")
        if any(
            image.shape != shape or image.dtype != np.uint8 for image in images
        ):
            raise ValueError(f"{camera} 图像尺寸或类型不一致")
    for side in SIDES:
        samples = episode.samples[side]
        if len(samples) != n:
            raise ValueError(f"{side} 机械臂采样数量与 episode 不一致")
        for index, sample in enumerate(samples):
            joints = np.asarray(sample.joints, dtype=np.float64)
            pose = np.asarray(sample.flange_pose, dtype=np.float64)
            action_joints = np.asarray(sample.action.joints, dtype=np.float64)
            values = np.asarray((*joints, sample.gripper_width), dtype=np.float64)
            actions = np.asarray((*action_joints, sample.action.gripper_width), dtype=np.float64)
            if joints.shape != (6,) or pose.shape != (6,) or action_joints.shape != (6,):
                raise ValueError(f"{side} 第 {index} 帧反馈维度错误")
            if (
                not np.isfinite(pose).all()
                or not np.isfinite(values).all()
                or not np.isfinite(actions).all()
            ):
                raise ValueError(f"{side} 第 {index} 帧反馈含 NaN/Inf")
            if not np.isfinite(float(sample.action.timestamp)):
                raise ValueError(f"{side} 第 {index} 帧动作时间戳无效")


def _create_dataset(group, name: str, data: np.ndarray):
    return group.create_dataset(name, data=data, compression="lzf", shuffle=True)


def _set_format(dataset, *, units: str, layout: str = "") -> None:
    dataset.attrs["normalized"] = False
    dataset.attrs["units"] = units
    if layout:
        dataset.attrs["layout"] = layout


def _create_formatted_dataset(
    group, name: str, data: np.ndarray, *, units: str, layout: str = ""
):
    dataset = _create_dataset(group, name, data)
    _set_format(dataset, units=units, layout=layout)
    return dataset


def write_episode(
    episode: EpisodeData,
    instruction: str,
    output_dir: Path,
    *,
    pyagxarm_version: str,
    firmware_profile: str,
    firmware_version: str,
    gripper_force: float,
    record_hz: float | None = None,
) -> Path:
    if not np.isfinite(gripper_force):
        raise ValueError("夹爪控制力必须是有限值")
    if record_hz is not None and (not np.isfinite(record_hz) or record_hz <= 0):
        raise ValueError("record_hz 必须是有限的正数")
    _validate_episode(episode)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"pyagxarm_dual_{stamp}.hdf5"
    suffix = 1
    while path.exists() or path.with_suffix(".hdf5.tmp").exists():
        path = output_dir / f"pyagxarm_dual_{stamp}_{suffix}.hdf5"
        suffix += 1
    temporary = path.with_suffix(".hdf5.tmp")
    t0 = episode.timestamps[0]
    sides = {side: _side_arrays(episode.samples[side], side) for side in SIDES}
    clip_count = sum(int(sides[side]["clip_count"]) for side in SIDES)
    if clip_count:
        print(f"保存时已将 {clip_count} 个轻微越界值截断到 Piper 边界")

    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = "pyagxarm_direct_v1"
            handle.attrs["control_library"] = "pyAgxArm"
            handle.attrs["pyagxarm_version"] = pyagxarm_version
            handle.attrs["firmware_profile"] = firmware_profile
            handle.attrs["firmware_version"] = firmware_version
            handle.attrs["normalized"] = False
            handle.attrs["boundary_clipped_values"] = clip_count
            handle.attrs["joint_boundary_tolerance_rad"] = (
                JOINT_BOUNDARY_TOLERANCE_RAD
            )
            handle.attrs["gripper_boundary_tolerance_m"] = (
                GRIPPER_BOUNDARY_TOLERANCE_M
            )

            data = handle.create_group("data")
            demo = data.create_group("demo_0")
            demo.attrs["num_samples"] = len(episode)
            demo.attrs["language"] = instruction
            for key, value in episode.recording_metadata.items():
                demo.attrs[key] = value
            demo.attrs["gripper_force_n"] = gripper_force
            if record_hz is not None:
                demo.attrs["record_hz"] = float(record_hz)

            obs = demo.create_group("obs")
            for name in CAMERA_NAMES:
                image_dataset = _create_dataset(
                    obs, name, np.stack(episode.images[name])
                )
                _set_format(image_dataset, units="uint8", layout="height,width,rgb")

            for side in SIDES:
                joint_dataset = _create_dataset(
                    obs, f"{side}_joint_states", sides[side]["joints"]
                )
                _set_format(joint_dataset, units="rad", layout="j1,j2,j3,j4,j5,j6")
                pose_dataset = _create_dataset(
                    obs, f"{side}_ee_states", sides[side]["poses"]
                )
                _set_format(
                    pose_dataset,
                    units="m,m,m,rad,rad,rad",
                    layout="x,y,z,roll,pitch,yaw",
                )
                gripper_dataset = _create_dataset(
                    obs, f"{side}_gripper_states", sides[side]["grippers"]
                )
                _set_format(gripper_dataset, units="m", layout="width")
                force_dataset = _create_dataset(
                    obs, f"{side}_gripper_force", sides[side]["forces"]
                )
                _set_format(force_dataset, units="N", layout="force")

            joint_states = np.concatenate(
                [sides[side]["joints"] for side in SIDES], axis=1
            )
            ee_states = np.concatenate([sides[side]["poses"] for side in SIDES], axis=1)
            gripper_states = np.concatenate(
                [sides[side]["grippers"] for side in SIDES], axis=1
            )
            states = np.concatenate([sides[side]["states"] for side in SIDES], axis=1)
            joint_dataset = _create_dataset(obs, "joint_states", joint_states)
            _set_format(joint_dataset, units="rad", layout="left_j1..j6,right_j1..j6")
            pose_dataset = _create_dataset(obs, "ee_states", ee_states)
            _set_format(
                pose_dataset,
                units="m,m,m,rad,rad,rad per arm",
                layout="left_xyzrpy,right_xyzrpy",
            )
            gripper_dataset = _create_dataset(obs, "gripper_states", gripper_states)
            _set_format(gripper_dataset, units="m", layout="left_width,right_width")
            states_dataset = _create_dataset(demo, "states", states)
            _set_format(
                states_dataset,
                units="rad x6,m per arm",
                layout="left_j1..j6,left_gripper,right_j1..j6,right_gripper",
            )
            demo["robot_states"] = states_dataset

            for side in SIDES:
                side_actions = _create_dataset(
                    demo, f"{side}_actions", sides[side]["actions"]
                )
                _set_format(
                    side_actions,
                    units="rad,rad,rad,rad,rad,rad,m",
                    layout="j1,j2,j3,j4,j5,j6,gripper_width",
                )
            actions = _create_dataset(
                demo,
                "actions",
                np.concatenate([sides[side]["actions"] for side in SIDES], axis=1),
            )
            _set_format(
                actions,
                units="rad x6,m per arm",
                layout=ACTION_LAYOUT,
            )
            actions.attrs["replay"] = (
                "move_j(action[:6]); move_gripper_m(action[6], force=gripper_force_n)"
            )

            _create_formatted_dataset(
                demo,
                "timestamps",
                np.asarray(episode.timestamps) - t0,
                units="s",
                layout="seconds_since_first_frame",
            )
            _create_formatted_dataset(
                demo,
                "camera_timestamps",
                np.asarray(episode.camera_timestamps, dtype=np.float64) - t0,
                units="s",
                layout="agentview,left_eye_in_hand,right_eye_in_hand",
            )
            _create_formatted_dataset(
                demo,
                "action_source_timestamps",
                np.column_stack(
                    [sides[side]["action_timestamps"] - t0 for side in SIDES]
                ),
                units="s",
                layout="left,right",
            )
            if episode.scheduled_timestamps:
                _create_formatted_dataset(
                    demo,
                    "scheduled_timestamps",
                    np.asarray(episode.scheduled_timestamps, dtype=np.float64) - t0,
                    units="s",
                    layout="monotonic_clock_schedule",
                )
            if episode.capture_durations:
                _create_formatted_dataset(
                    demo,
                    "capture_durations",
                    np.asarray(episode.capture_durations, dtype=np.float64),
                    units="s",
                    layout="per_frame_capture_duration",
                )
            for name, values, layout in (
                ("camera_durations", episode.camera_durations, "per_frame_camera_read_duration"),
                ("robot_durations", episode.robot_durations, "per_frame_robot_read_duration"),
            ):
                if values:
                    _create_formatted_dataset(
                        demo, name, np.asarray(values, dtype=np.float64),
                        units="s", layout=layout,
                    )
            _create_formatted_dataset(
                demo,
                "trajectory_phase",
                np.asarray(episode.phases, dtype=np.int8),
                units="enum",
                layout="0=task,1=home",
            )
            _create_formatted_dataset(
                demo,
                "rewards",
                np.zeros(len(episode), dtype=np.float32),
                units="dimensionless",
                layout="per_frame",
            )
            dones = np.zeros(len(episode), dtype=np.bool_)
            dones[-1] = True
            _create_formatted_dataset(
                demo,
                "dones",
                dones,
                units="boolean",
                layout="per_frame",
            )
            handle.flush()
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path
