"""Offline pose diagnostics. Heuristic warnings never modify the trajectory."""

from dataclasses import asdict, dataclass

import numpy as np

from .pose_actions import matrix_rotvec, rpy_matrix
from .pose_conversion import read_raw_poses


@dataclass(frozen=True)
class PoseThresholds:
    max_speed_m_s: float = 1.0
    max_angular_speed_rad_s: float = 6.0
    idle_translation_m: float = 0.0001
    idle_rotation_rad: float = 0.001

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("位姿检测阈值必须为有限正数")


def edge_count(mask, tail=False):
    count = 0
    for flag in (mask[::-1] if tail else mask):
        if not flag:
            break
        count += 1
    return count


def analyze_poses(poses, timestamps, states, *, thresholds=None):
    limits = thresholds or PoseThresholds()
    poses, ts, states = np.asarray(poses), np.asarray(timestamps), np.asarray(states)
    if poses.shape != (len(ts), 12) or states.shape != (len(ts), 14):
        raise ValueError("位姿检测需要 (T,12) pose、(T,) 时间和 (T,14) state")
    if not np.isfinite(poses).all() or not np.isfinite(ts).all() or not np.isfinite(states).all():
        raise ValueError("位姿检测数组含 NaN/Inf")
    dt = np.diff(ts)
    if np.any(dt <= 0):
        raise ValueError("位姿时间戳必须严格递增")
    warnings, sides = [], {}
    for side, offset, joint in (("left", 0, 0), ("right", 6, 7)):
        values = poses[:, offset:offset+6]
        rotations = np.asarray([rpy_matrix(v[3:]) for v in values])
        distances = np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1)
        angles = np.asarray([np.linalg.norm(matrix_rotvec(b @ a.T))
                             for a, b in zip(rotations[:-1], rotations[1:])])
        speed, angular = distances/dt, angles/dt
        idle = (distances <= limits.idle_translation_m) & (angles <= limits.idle_rotation_rad)
        idle &= np.abs(np.diff(states[:, joint+6])) <= 0.0001
        joint_motion = np.max(np.abs(np.diff(states[:, joint:joint+6], axis=0)), axis=1)
        frozen = (distances < 1e-10) & (angles < 1e-10) & (joint_motion > 0.005)
        excessive = (speed > limits.max_speed_m_s) | (angular > limits.max_angular_speed_rad_s)
        rotation_vectors = np.asarray([matrix_rotvec(r) for r in rotations])
        branch = ((np.linalg.norm(np.diff(rotation_vectors, axis=0), axis=1) > np.pi)
                  & (angles < .2))
        metric = {
            "max_translation_step_m": float(distances.max(initial=0)),
            "max_rotation_step_rad": float(angles.max(initial=0)),
            "max_speed_m_s": float(speed.max(initial=0)),
            "max_angular_speed_rad_s": float(angular.max(initial=0)),
            "leading_near_idle_frames": edge_count(idle),
            "trailing_near_idle_frames": edge_count(idle, True),
            "speed_outlier_frames": (np.flatnonzero(excessive)+1).tolist(),
            "frozen_pose_moving_joint_frames": (np.flatnonzero(frozen)+1).tolist(),
            "rotvec_branch_crossing_frames": (np.flatnonzero(branch)+1).tolist(),
        }
        sides[side] = metric
        for key, description in (("speed_outlier_frames", "位姿速度超过提示阈值"),
                                 ("frozen_pose_moving_joint_frames", "关节运动而位姿完全重复，请检查反馈同步"),
                                 ("rotvec_branch_crossing_frames", "绝对旋转向量跨 ±π 分支；真实旋转连续")):
            if metric[key]:
                warnings.append(f"{side}: {description}，{len(metric[key])} 处")
        for edge in ("leading", "trailing"):
            if metric[f"{edge}_near_idle_frames"] >= 5:
                warnings.append(f"{side}: {edge} 近静止 {metric[f'{edge}_near_idle_frames']} 帧（不自动裁剪）")
    return {"thresholds": asdict(limits), "sides": sides, "warnings": warnings}


def analyze_raw_pose(demo, thresholds=None):
    poses = read_raw_poses(demo)
    selected = np.flatnonzero(np.asarray(demo["trajectory_phase"]) == 0)
    return analyze_poses(poses[selected], np.asarray(demo["timestamps"])[selected],
                         np.asarray(demo["states"])[selected], thresholds=thresholds)
