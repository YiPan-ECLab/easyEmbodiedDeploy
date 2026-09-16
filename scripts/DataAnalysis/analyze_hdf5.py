#!/usr/bin/env python3
"""Analyze Piper Recorder HDF5 files and fail on training-data defects."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
JOINT_LIMITS = np.asarray(
    [
        (-2.617994, 2.617994),
        (0.0, 3.141593),
        (-2.967060, 0.0),
        (-1.745330, 1.745330),
        (-1.221730, 1.221730),
        (-2.094396, 2.094396),
    ],
    dtype=np.float64,
)
PREPOSITION = np.asarray(
    [0, 0, 0, 0, 0, -np.pi / 4, 0, 0, 0, 0, 0, 0, np.pi / 4, 0],
    dtype=np.float64,
)
EXPECTED_LAYOUT = (
    "left_j1_rad,left_j2_rad,left_j3_rad,left_j4_rad,left_j5_rad,left_j6_rad,"
    "left_gripper_m,right_j1_rad,right_j2_rad,right_j3_rad,right_j4_rad,"
    "right_j5_rad,right_j6_rad,right_gripper_m"
)
EXPECTED_STATE_LAYOUT = "left_j1..j6,left_gripper,right_j1..j6,right_gripper"


def plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def text_attr(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def edge_repeat_count(actions: np.ndarray, states: np.ndarray, *, tail: bool) -> int:
    same = np.all(actions[1:] == actions[:-1], axis=1) & np.all(
        states[1:] == states[:-1], axis=1
    )
    count = 0
    for repeated in (same[::-1] if tail else same):
        if not repeated:
            break
        count += 1
    return count


def trailing_action_repeat_count(actions: np.ndarray) -> int:
    """Count the final unchanged leader targets that the processor removes."""
    count = 0
    for repeated in np.all(actions[1:] == actions[:-1], axis=1)[::-1]:
        if not repeated:
            break
        count += 1
    return count


def range_errors(values: np.ndarray, label: str) -> list[str]:
    errors: list[str] = []
    for offset, side in ((0, "left"), (7, "right")):
        joints = values[:, offset : offset + 6]
        bad = (joints < JOINT_LIMITS[:, 0]) | (joints > JOINT_LIMITS[:, 1])
        if np.any(bad):
            frame, joint = np.argwhere(bad)[0]
            errors.append(
                f"{label}: frame {frame} {side} J{joint + 1}="
                f"{joints[frame, joint]:.6f} rad 超限"
            )
        gripper = values[:, offset + 6]
        bad_gripper = (gripper < 0) | (gripper > 0.1)
        if np.any(bad_gripper):
            frame = int(np.flatnonzero(bad_gripper)[0])
            errors.append(
                f"{label}: frame {frame} {side} gripper={gripper[frame]:.6f} m 超限"
            )
    return errors


def analyze_file(path: Path, expected_fps: float, gap_factor: float) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {"file": path.name}
    try:
        with h5py.File(path, "r") as handle:
            if text_attr(handle.attrs.get("format", "")) != "pyagxarm_direct_v1":
                errors.append("根属性 format 不是 pyagxarm_direct_v1")
            if bool(handle.attrs.get("normalized", False)):
                errors.append("源 HDF5 不应被归一化")
            if "data/demo_0" not in handle:
                raise ValueError("缺少 data/demo_0")
            demo = handle["data/demo_0"]
            required = ("actions", "states", "timestamps", "trajectory_phase", "dones")
            missing = [name for name in required if name not in demo]
            if missing:
                raise ValueError(f"缺少数据集: {missing}")

            actions = np.asarray(demo["actions"], dtype=np.float64)
            states = np.asarray(demo["states"], dtype=np.float64)
            timestamps = np.asarray(demo["timestamps"], dtype=np.float64)
            phases = np.asarray(demo["trajectory_phase"])
            frames = len(actions)
            metrics["frames"] = frames
            if actions.shape != (frames, 14) or states.shape != (frames, 14):
                errors.append("actions/states 必须为 (T,14)")
            if frames < 2 or timestamps.shape != (frames,) or phases.shape != (frames,):
                errors.append("帧数、timestamps 或 trajectory_phase 维度错误")
                raise ValueError("核心数组维度不合法")
            if int(demo.attrs.get("num_samples", frames)) != frames:
                errors.append("num_samples 与实际帧数不一致")
            for name in ("actions", "states"):
                dataset = demo[name]
                if text_attr(dataset.attrs.get("units", "")) != "rad x6,m per arm":
                    errors.append(f"{name} units 错误")
                layout = text_attr(dataset.attrs.get("layout", ""))
                expected_layout = EXPECTED_LAYOUT if name == "actions" else EXPECTED_STATE_LAYOUT
                if layout != expected_layout:
                    errors.append(f"{name} layout 错误")
            if not np.isfinite(actions).all() or not np.isfinite(states).all():
                errors.append("actions/states 含 NaN 或 Inf")
            else:
                errors.extend(range_errors(actions, "actions"))
                errors.extend(range_errors(states, "states"))

            if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
                errors.append("timestamps 必须有限且严格递增")
                task_dt = np.asarray([], dtype=np.float64)
            else:
                if phases[0] != 0 or np.any(~np.isin(phases, (0, 1))):
                    errors.append("trajectory_phase 必须从 task(0) 开始且只能包含 0/1")
                home_positions = np.flatnonzero(phases == 1)
                if len(home_positions) and not np.all(phases[home_positions[0] :] == 1):
                    errors.append("trajectory_phase 不是连续 task 前缀与 home 后缀")
                task_positions = np.flatnonzero(phases == 0)
                home_positions = np.flatnonzero(phases == 1)
                metrics["task_frames"] = int(len(task_positions))
                metrics["home_frames"] = int(len(home_positions))
                task_dt = np.diff(timestamps[task_positions])
                nominal = 1 / expected_fps
                measured = float(1 / np.median(task_dt)) if len(task_dt) else 0.0
                bad_gap = (task_dt < nominal / gap_factor) | (task_dt > nominal * gap_factor)
                metrics.update(
                    measured_fps=measured,
                    task_dt_min_ms=float(task_dt.min() * 1000) if len(task_dt) else 0.0,
                    task_dt_median_ms=float(np.median(task_dt) * 1000) if len(task_dt) else 0.0,
                    task_dt_max_ms=float(task_dt.max() * 1000) if len(task_dt) else 0.0,
                    timing_outliers=int(np.count_nonzero(bad_gap)),
                )
                if measured and abs(measured - expected_fps) / expected_fps > 0.02:
                    errors.append(f"实测 {measured:.3f} Hz 偏离期望 {expected_fps:g} Hz 超过 2%")
                if np.any(bad_gap):
                    errors.append(
                        f"task 有 {np.count_nonzero(bad_gap)} 个显著时间异常，"
                        f"范围 {task_dt.min() * 1000:.3f}..{task_dt.max() * 1000:.3f} ms"
                    )

                task_actions = actions[task_positions]
                task_states = states[task_positions]
                leading_idle = edge_repeat_count(task_actions, task_states, tail=False)
                trailing_idle = trailing_action_repeat_count(task_actions)
                baseline = np.asarray(demo.attrs.get("start_baseline", PREPOSITION), dtype=np.float64)
                joint_threshold = float(demo.attrs.get("start_joint_delta_rad", 0.01))
                gripper_threshold = float(demo.attrs.get("start_gripper_delta_m", 0.002))
                first_delta = np.abs(task_actions[0] - baseline)
                first_started = bool(
                    np.max(first_delta[np.r_[0:6, 7:13]]) >= joint_threshold - 1e-9
                    or np.max(first_delta[[6, 13]]) >= gripper_threshold - 1e-9
                )
                metrics.update(
                    leading_idle_frames=leading_idle,
                    trailing_idle_frames=trailing_idle,
                    first_frame_operator_started=first_started,
                    max_task_action_joint_step_rad=float(
                        np.max(np.abs(np.diff(task_actions[:, np.r_[0:6, 7:13]], axis=0)))
                    ) if len(task_actions) > 1 else 0.0,
                    max_task_state_joint_step_rad=float(
                        np.max(np.abs(np.diff(task_states[:, np.r_[0:6, 7:13]], axis=0)))
                    ) if len(task_states) > 1 else 0.0,
                )
                if leading_idle:
                    errors.append(f"task 开头有 {leading_idle} 个完全静止的相邻帧")
                if not first_started:
                    errors.append("task 第一帧未达到 Recorder 的人工接管阈值")
                if trailing_idle:
                    warnings.append(
                        f"task 尾部有 {trailing_idle} 个重复示教目标，DataProcessor 将自动裁掉"
                    )

            if "left_actions" in demo and "right_actions" in demo:
                split = np.concatenate(
                    [np.asarray(demo["left_actions"]), np.asarray(demo["right_actions"])], axis=1
                )
                if not np.array_equal(actions, split):
                    errors.append("actions 与 left/right_actions 不一致")
            state_parts = (
                "obs/left_joint_states", "obs/left_gripper_states",
                "obs/right_joint_states", "obs/right_gripper_states",
            )
            if all(name in demo for name in state_parts):
                split = np.concatenate([np.asarray(demo[name]) for name in state_parts], axis=1)
                if not np.array_equal(states, split):
                    errors.append("states 与左右 observation 数据不一致")

            camera_timestamps = np.asarray(demo.get("camera_timestamps", []), dtype=np.float64)
            if camera_timestamps.shape != (frames, len(CAMERAS)) or not np.isfinite(camera_timestamps).all():
                errors.append("camera_timestamps 必须为有限的 (T,3)")
            else:
                backwards = np.sum(np.diff(camera_timestamps, axis=0) < 0, axis=0)
                reused = np.sum(np.diff(camera_timestamps, axis=0) == 0, axis=0)
                skew = float(np.max(np.abs(camera_timestamps - timestamps[:, None])))
                metrics["camera_reused_frames"] = dict(zip(CAMERAS, map(int, reused)))
                metrics["max_camera_skew_ms"] = skew * 1000
                if np.any(backwards):
                    errors.append(f"相机时间戳倒退: {backwards.tolist()}")
                if skew > 0.5:
                    errors.append(f"相机与采样时间最大偏差 {skew:.3f}s 超过 0.5s")
            image_shapes = {}
            for camera in CAMERAS:
                key = f"obs/{camera}"
                if key not in demo:
                    errors.append(f"缺少相机 {camera}")
                    continue
                image = demo[key]
                image_shapes[camera] = list(image.shape)
                if image.shape != (frames, 224, 224, 3) or image.dtype != np.uint8:
                    errors.append(f"{camera} 必须为 (T,224,224,3) uint8")
            metrics["image_shapes"] = image_shapes

            dones = np.asarray(demo["dones"], dtype=bool)
            if dones.shape != (frames,) or not np.array_equal(np.flatnonzero(dones), [frames - 1]):
                errors.append("dones 必须只在 episode 最后一帧为 true")
            if "action_source_timestamps" in demo:
                source_ts = np.asarray(demo["action_source_timestamps"], dtype=np.float64)
                valid_source_ts = (
                    source_ts.shape == (frames, 2)
                    and np.isfinite(source_ts).all()
                    and np.all(np.diff(source_ts, axis=0) >= 0)
                    and np.all(source_ts <= timestamps[:, None] + 1e-9)
                )
                if not valid_source_ts:
                    errors.append("action_source_timestamps 时钟基准或顺序错误")
            metrics["boundary_clipped_values"] = int(handle.attrs.get("boundary_clipped_values", 0))
    except Exception as exc:
        errors.append(f"读取失败: {type(exc).__name__}: {exc}")
    return {"file": path.name, "valid": not errors, "errors": errors,
            "warnings": warnings, "metrics": metrics}


def resolve_files(raw: str) -> tuple[Path, list[Path]]:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path.is_file():
        return path.parent, [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted((*path.glob("*.hdf5"), *path.glob("*.h5")))
    if not files:
        raise ValueError(f"目录中没有 HDF5: {path}")
    return path, files


def print_report(report: dict[str, Any]) -> None:
    print(f"Piper HDF5 quality report: {report['root']}")
    print(f"Expected frequency: {report['expected_fps']:g} Hz")
    print("-")
    print(f"{'RESULT':7} {'TASK':>6} {'HOME':>6} {'TAIL':>6} {'FPS':>8} {'DT MAX':>9}  FILE")
    for item in report["files"]:
        m = item["metrics"]
        print(
            f"{('PASS' if item['valid'] else 'FAIL'):7} "
            f"{m.get('task_frames', 0):6d} {m.get('home_frames', 0):6d} "
            f"{m.get('trailing_idle_frames', 0):6d} {m.get('measured_fps', 0):8.3f} "
            f"{m.get('task_dt_max_ms', 0):8.3f}ms  {item['file']}"
        )
        for error in item["errors"]:
            print(f"  ERROR: {error}")
        for warning in item["warnings"]:
            print(f"  WARNING: {warning}")
    s = report["summary"]
    print("-")
    print(
        f"{report['status']}: {s['valid_files']}/{s['files']} files, "
        f"task={s['task_frames']}, home={s['home_frames']}, "
        f"processor_tail_trim={s['trailing_idle_frames']}, "
        f"timing_outliers={s['timing_outliers']}, errors={s['errors']}"
    )
    print(
        f"fps_range={s['measured_fps_min']:.3f}..{s['measured_fps_max']:.3f}, "
        f"max_dt={s['max_task_dt_ms']:.3f} ms, "
        f"max_camera_skew={s['max_camera_skew_ms']:.3f} ms, "
        f"boundary_clipped_values={s['boundary_clipped_values']}"
    )
    print(
        f"max task joint step: action={s['max_task_action_joint_step_rad']:.6f} rad, "
        f"state={s['max_task_state_joint_step_rad']:.6f} rad"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="HDF5 file or directory")
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument(
        "--gap-factor", type=float, default=1.5,
        help="fail intervals outside [period/factor, period*factor] (default: 1.5)",
    )
    parser.add_argument("--json", dest="json_path", help="write the full JSON report")
    args = parser.parse_args()
    if (not math.isfinite(args.expected_fps) or not math.isfinite(args.gap_factor)
            or args.expected_fps <= 0 or args.gap_factor <= 1):
        parser.error("--expected-fps must be positive and --gap-factor must exceed 1")
    root, files = resolve_files(args.path)
    results = [analyze_file(path, args.expected_fps, args.gap_factor) for path in files]
    measured_fps = [item["metrics"].get("measured_fps", 0.0) for item in results]
    summary = {
        "files": len(results),
        "valid_files": sum(item["valid"] for item in results),
        "frames": sum(item["metrics"].get("frames", 0) for item in results),
        "task_frames": sum(item["metrics"].get("task_frames", 0) for item in results),
        "home_frames": sum(item["metrics"].get("home_frames", 0) for item in results),
        "trailing_idle_frames": sum(item["metrics"].get("trailing_idle_frames", 0) for item in results),
        "timing_outliers": sum(item["metrics"].get("timing_outliers", 0) for item in results),
        "errors": sum(len(item["errors"]) for item in results),
        "warnings": sum(len(item["warnings"]) for item in results),
        "measured_fps_min": min(measured_fps),
        "measured_fps_max": max(measured_fps),
        "max_task_dt_ms": max(item["metrics"].get("task_dt_max_ms", 0.0) for item in results),
        "max_camera_skew_ms": max(item["metrics"].get("max_camera_skew_ms", 0.0) for item in results),
        "max_task_action_joint_step_rad": max(
            item["metrics"].get("max_task_action_joint_step_rad", 0.0) for item in results
        ),
        "max_task_state_joint_step_rad": max(
            item["metrics"].get("max_task_state_joint_step_rad", 0.0) for item in results
        ),
        "boundary_clipped_values": sum(
            item["metrics"].get("boundary_clipped_values", 0) for item in results
        ),
    }
    report = {
        "status": "PASS" if summary["errors"] == 0 else "FAIL",
        "root": str(root),
        "expected_fps": args.expected_fps,
        "gap_factor": args.gap_factor,
        "summary": summary,
        "files": results,
    }
    print_report(report)
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=plain), encoding="utf-8"
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
