"""Descriptive joint-label diagnostics, without claiming training causality."""

import numpy as np


def joint_label_metrics(commands, states, horizon=50):
    commands, states = np.asarray(commands), np.asarray(states)
    if commands.ndim != 2 or commands.shape[1] != 14 or states.shape != commands.shape or len(commands) < 2:
        raise ValueError("关节分析需要至少两帧 (T,14) command/state")
    if not np.isfinite(commands).all() or not np.isfinite(states).all():
        raise ValueError("关节分析包含 NaN/Inf")
    joints = np.r_[0:6, 7:13]
    delta = np.diff(commands, axis=0)
    lag_rmse = []
    for lag in range(min(11, max(1, len(commands)//4))):
        a, b = (commands, states) if lag == 0 else (commands[:-lag], states[lag:])
        lag_rmse.append(float(np.sqrt(np.mean((a[:, joints]-b[:, joints])**2))))
    rows = len(delta)
    valid = sum(min(horizon, rows-i) for i in range(rows))
    return {
        "delta_joint_abs_median_rad": float(np.median(np.abs(delta[:, joints]))),
        "delta_joint_abs_p99_rad": float(np.quantile(np.abs(delta[:, joints]), .99)),
        "delta_joint_near_zero_fraction": float(np.mean(np.abs(delta[:, joints]) < 1e-5)),
        "last_delta_joint_l2_rad": float(np.linalg.norm(delta[-1, joints])),
        "last_delta_gripper_m": delta[-1, [6, 13]].tolist(),
        "legacy_repeat_padding_fraction": float(1-valid/(rows*horizon)),
        "initial_command_feedback_joint_l2_rad": float(np.linalg.norm((commands[0]-states[0])[joints])),
        "command_feedback_rmse_rad_by_lag": lag_rmse,
        "minimum_error_lag_frames": int(np.argmin(lag_rmse)),
        "note": "时延只是描述性匹配；不能据此断言绝对/相对训练优劣。旧 padding 比例按 horizon 计算。",
    }
