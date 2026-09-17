"""Piper flange geometry; no hardware access or model-state augmentation.

T_A_B maps B coordinates into A. Camera deltas follow Qwen Eq. (5), with
one frozen camera orientation for the entire prediction chunk.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .action_contract import POSE_MODES


def finite(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须为有限的 {shape} 数组")
    return result


def rpy_matrix(value):
    roll, pitch, yaw = finite(value, (3,), "RPY")
    sr, sp, sy = np.sin([roll, pitch, yaw])
    cr, cp, cy = np.cos([roll, pitch, yaw])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def validate_rotation(value):
    matrix = finite(value, (3, 3), "rotation")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(matrix), 1, atol=1e-6):
        raise ValueError("旋转矩阵必须属于 SO(3)")
    return matrix


def rotation_6d(matrix):
    return validate_rotation(matrix)[:, :2].T.reshape(6).copy()


def matrix_6d(value):
    value = finite(value, (6,), "rotation6d")
    a, b = value[:3], value[3:]
    if np.linalg.norm(a) < 1e-10:
        raise ValueError("6D 旋转第一列退化")
    a = a / np.linalg.norm(a)
    b = b - a * np.dot(a, b)
    if np.linalg.norm(b) < 1e-10:
        raise ValueError("6D 旋转两列共线")
    b = b / np.linalg.norm(b)
    return np.column_stack((a, b, np.cross(a, b)))


def rotvec_matrix(value):
    import cv2
    return cv2.Rodrigues(finite(value, (3,), "rotvec"))[0]


def matrix_rotvec(value):
    import cv2
    # The explicit 6D round trip fixes column order and orthonormalization.
    return cv2.Rodrigues(matrix_6d(rotation_6d(value)))[0].reshape(3)


def matrix_rpy(value):
    matrix = validate_rotation(value)
    pitch = np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0]))
    if abs(np.cos(pitch)) > 1e-8:
        roll, yaw = np.arctan2(matrix[2, 1], matrix[2, 2]), np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll, yaw = np.arctan2(-matrix[1, 2], matrix[1, 1]), 0.0
    return np.array([roll, pitch, yaw])


def flange_matrix(pose):
    pose = finite(pose, (6,), "flange xyz/RPY")
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rpy_matrix(pose[3:]), pose[:3]
    return matrix


def validate_transform(value):
    matrix = finite(value, (4, 4), "T_A_B")
    validate_rotation(matrix[:3, :3])
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("齐次矩阵最后一行必须为 [0,0,0,1]")
    return matrix


def calibration_for_mode(bundle, mode, mount_id=None):
    if mode == "absolute_base_pose":
        return {}
    jobs = {"camera_delta_wrist": ("left_wrist", "right_wrist"),
            "camera_delta_global": ("global_left", "global_right")}.get(mode)
    if not jobs or not isinstance(bundle, dict) or bundle.get("schema") != "easy_embodied_camera_calibration":
        raise ValueError("camera delta 需要有效的已导出标定包")
    result, mounts, cameras = {}, set(), []
    for side, job in zip(("left", "right"), jobs):
        matches = [c for c in bundle.get("calibrations", []) if c.get("job") == job]
        if len(matches) != 1:
            raise ValueError(f"标定包必须恰好包含一项 {job}")
        item = matches[0]
        solved = item.get("result") or {}
        expected = f"T_{side}_flange_{side}_camera" if mode == "camera_delta_wrist" else f"T_{side}_base_global_camera"
        role = f"{side}_eye_in_hand" if mode == "camera_delta_wrist" else "agentview"
        if solved.get("transform") != expected or not solved.get("accepted"):
            raise ValueError(f"{job} 方向错误或尚未确认导出")
        if item.get("camera", {}).get("role") != role or not item["camera"].get("serial"):
            raise ValueError(f"{job} 相机身份不完整")
        if not item.get("mount_id"):
            raise ValueError(f"{job} 缺少安装版本")
        mounts.add(item["mount_id"])
        cameras.append(item["camera"])
        result[side] = validate_transform(solved.get("matrix"))
    if len(mounts) != 1 or (mount_id is not None and mounts != {mount_id}):
        raise ValueError("标定包安装版本与数据采集安装版本不一致")
    if mode == "camera_delta_global" and any(cameras[0].get(k) != cameras[1].get(k) for k in ("serial", "sensor", "width", "height")):
        raise ValueError("左右 base 外参必须来自同一全局相机和视频配置")
    return result


def load_calibration(path, mode, mount_id):
    if mode == "absolute_base_pose":
        return None
    if not path or not mount_id:
        raise ValueError("camera delta 须选择标定包并确认采集时安装版本")
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    calibration_for_mode(bundle, mode, mount_id)
    return bundle


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class PoseCodec:
    def __init__(self, mode, calibration=None):
        if mode not in POSE_MODES:
            raise ValueError(f"未知位姿动作模式：{mode}")
        self.mode = mode
        self.extrinsics = calibration_for_mode(calibration, mode)

    def _camera_rotation(self, anchor, side):
        calibration = self.extrinsics[side][:3, :3]
        return anchor @ calibration if self.mode == "camera_delta_wrist" else calibration

    def encode(self, anchor_pose, target_pose, grippers):
        anchor = finite(anchor_pose, (12,), "当前双臂法兰位姿")
        target = finite(target_pose, (12,), "目标双臂法兰位姿")
        grip = finite(grippers, (2,), "夹爪")
        if np.any((grip < -1e-8) | (grip > .1+1e-8)):
            raise ValueError("夹爪绝对宽度必须在 [0,0.1] m")
        grip = np.clip(grip, 0, .1)
        action = np.empty(14)
        for index, side in enumerate(("left", "right")):
            p, q = anchor[index*6:index*6+6], target[index*6:index*6+6]
            current_r, future_r = rpy_matrix(p[3:]), rpy_matrix(q[3:])
            if self.mode == "absolute_base_pose":
                xyz, rotation = q[:3], future_r
            else:
                camera_r = self._camera_rotation(current_r, side)
                xyz = camera_r.T @ (q[:3]-p[:3])
                rotation = camera_r.T @ future_r @ current_r.T @ camera_r
            action[index*7:index*7+7] = np.r_[xyz, matrix_rotvec(rotation), grip[index]]
        return action

    def decode(self, anchor_pose, action):
        anchor = finite(anchor_pose, (12,), "当前双臂法兰位姿")
        action = finite(action, (14,), "位姿 action")
        poses = np.empty(12)
        for index, side in enumerate(("left", "right")):
            current = anchor[index*6:index*6+6]
            value = action[index*7:index*7+7]
            rotation = rotvec_matrix(value[3:6])
            if self.mode == "absolute_base_pose":
                xyz = value[:3]
            else:
                current_r = rpy_matrix(current[3:])
                camera_r = self._camera_rotation(current_r, side)
                xyz = current[:3] + camera_r @ value[:3]
                rotation = camera_r @ rotation @ camera_r.T @ current_r
            poses[index*6:index*6+6] = np.r_[xyz, matrix_rpy(rotation)]
        return poses, action[[6, 13]].copy()

    def chunk(self, poses, grippers, index, horizon=50):
        poses = np.asarray(poses, dtype=np.float64)
        grippers = np.asarray(grippers, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 12 or grippers.shape != (len(poses), 2):
            raise ValueError("位姿 episode 须为 [T,12]，夹爪为 [T,2]")
        if not 0 <= index < len(poses)-1 or horizon < 1:
            raise ValueError("位姿锚点必须至少有一个有效未来目标")
        future = index + np.arange(1, horizon+1)
        valid = future < len(poses)
        future = np.minimum(future, len(poses)-1)
        actions = np.stack([self.encode(poses[index], poses[i], grippers[i]) for i in future])
        return actions.astype(np.float32), valid
