#!/usr/bin/env python3
"""Analyze a converted Piper LeRobot v3 bundle and verify source semantics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from easy_embodied_deploy.statistics import statistics_close
CAMERAS = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
AXES = [
    f"{side}_{axis}"
    for side in ("left", "right")
    for axis in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
]
JOINT_LIMITS = np.asarray(
    [
        (-2.617994, 2.617994), (0.0, 3.141593), (-2.967060, 0.0),
        (-1.745330, 1.745330), (-1.221730, 1.221730), (-2.094396, 2.094396),
    ], dtype=np.float64,
)
STATS = ("min", "max", "mean", "std", "q01", "q99")


def resolve(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def lerobot_root(raw: str) -> tuple[Path, Path]:
    bundle = resolve(raw)
    root = bundle / "lerobot" if (bundle / "lerobot/meta/info.json").is_file() else bundle
    if not (root / "meta/info.json").is_file():
        raise FileNotFoundError(f"找不到 LeRobot meta/info.json: {bundle}")
    return root.parent if root.name == "lerobot" else root, root


def parquet_table(root: Path) -> pa.Table:
    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise ValueError("缺少 data parquet")
    tables = [pq.read_table(path) for path in files]
    try:
        return pa.concat_tables(tables, promote_options="default")
    except TypeError:
        return pa.concat_tables(tables, promote=True)


def edge_repeats(actions: np.ndarray, states: np.ndarray, tail: bool) -> int:
    same = np.all(actions[1:] == actions[:-1], axis=1) & np.all(
        states[1:] == states[:-1], axis=1
    )
    count = 0
    for value in (same[::-1] if tail else same):
        if not value:
            break
        count += 1
    return count


def validate_ranges(values: np.ndarray, label: str) -> list[str]:
    errors = []
    for offset, side in ((0, "left"), (7, "right")):
        joints = values[:, offset : offset + 6]
        bad = (joints < JOINT_LIMITS[:, 0] - 1e-6) | (joints > JOINT_LIMITS[:, 1] + 1e-6)
        if np.any(bad):
            frame, joint = np.argwhere(bad)[0]
            errors.append(f"{label}: row {frame} {side} J{joint + 1} 超限")
        gripper = values[:, offset + 6]
        if np.any((gripper < -1e-6) | (gripper > 0.1 + 1e-6)):
            errors.append(f"{label}: {side} gripper 超出 [0,0.1] m")
    return errors


def source_indices(demo: h5py.Group, stride: int) -> tuple[np.ndarray, int]:
    phase = np.asarray(demo["trajectory_phase"])
    home = np.flatnonzero(phase == 1)
    task_end = int(home[0]) if len(home) else len(phase)
    actions = demo["actions"]
    last_active = task_end - 1
    while (
        last_active > 0
        and np.array_equal(actions[last_active], actions[last_active - 1])
    ):
        last_active -= 1
    selected = np.arange(0, len(phase), stride)
    selected = selected[phase[selected] == 0]
    endpoint = int(np.searchsorted(selected, last_active, side="left"))
    if endpoint < len(selected):
        selected = selected[: endpoint + 1]
    return selected, task_end - last_active - 1


def source_paths(
    source_raw: str | list[str] | None,
    plan_files: list[dict],
    source_override: str | None,
) -> tuple[list[Path], list[Path]]:
    """Resolve source HDF5 files for single- or multi-directory manifests."""
    raw_dirs = source_override or source_raw
    if isinstance(raw_dirs, str):
        raw_dirs = raw_dirs.splitlines()
    directories = [resolve(value) for value in (raw_dirs or []) if str(value).strip()]
    if not directories:
        return [], []
    if plan_files and not source_override and any(item.get("source_dir") for item in plan_files):
        files = []
        for item in plan_files:
            base = resolve(item.get("source_dir") or directories[0])
            path = base / str(item.get("name", ""))
            if path.is_file():
                files.append(path)
        return files, directories
    by_name = {}
    for directory in directories:
        for path in sorted((*directory.glob("*.hdf5"), *directory.glob("*.h5"))):
            if path.name in by_name and by_name[path.name] != path:
                raise ValueError(f"源目录中有同名文件 {path.name}，不能按文件名推断 episode 来源；请保留 processing.json 中的 source_dir")
            by_name[path.name] = path
    expected_names = [item.get("name") for item in plan_files if item.get("name")]
    if expected_names:
        return [by_name[name] for name in expected_names if name in by_name], directories
    return sorted(by_name.values()), directories


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)],
        check=False, capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise ValueError(f"ffprobe 失败: {path}: {result.stderr.strip()}")
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": float(Fraction(stream["r_frame_rate"])),
        "frames": None if stream.get("nb_frames") in (None, "N/A") else int(stream["nb_frames"]),
    }


def _analyze(raw: str, source_override: str | None) -> dict[str, Any]:
    bundle, root = lerobot_root(raw)
    errors: list[str] = []
    warnings: list[str] = []
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    report_path = bundle / "processing.json"
    processing = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    settings = processing.get("settings", {})
    plan = processing.get("plan", {})
    mode = "absolute_joint" if settings.get("action_mode") == "absolute" else settings.get("action_mode")
    fps = int(info.get("fps", 0))
    total_frames = int(info.get("total_frames", 0))
    total_episodes = int(info.get("total_episodes", 0))
    total_tasks = int(info.get("total_tasks", 0))
    if fps <= 0 or total_frames <= 0 or total_episodes <= 0:
        errors.append("fps、total_frames 和 total_episodes 必须为正数")
    if total_tasks != 1:
        errors.append("当前单任务转换结果的 total_tasks 必须为 1")
    if info.get("codebase_version") != "v3.0":
        errors.append("codebase_version 不是 v3.0")
    if info.get("robot_type") != "piper_dual":
        errors.append("robot_type 不是 piper_dual")
    if mode not in ("absolute_joint", "delta_joint"):
        errors.append("processing.json 缺少明确 action_mode")
    for key in ("observation.state", "action"):
        feature = info.get("features", {}).get(key, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [14]:
            errors.append(f"{key} feature 必须为 float32[14]")
        if feature.get("names", {}).get("axes") != AXES:
            errors.append(f"{key} axes 顺序错误")
        values = stats.get(key, {})
        for stat in STATS:
            array = np.asarray(values.get(stat, []), dtype=np.float64)
            if array.shape != (14,) or not np.isfinite(array).all():
                errors.append(f"stats {key}/{stat} 不是有限 14 维数组")
        if np.asarray(values.get("count", [])).tolist() != [total_frames]:
            errors.append(f"stats {key}/count 与总帧数不一致")
        if all(stat in values for stat in ("min", "q01", "q99", "max")):
            minimum, q01, q99, maximum = (
                np.asarray(values[name], dtype=np.float64) for name in ("min", "q01", "q99", "max")
            )
            if np.any(minimum > q01 + 1e-6) or np.any(q01 > q99 + 1e-6) or np.any(q99 > maximum + 1e-6):
                errors.append(f"stats {key} 分位数顺序错误")

    table = parquet_table(root)
    required = {"observation.state", "action", "timestamp", "frame_index",
                "episode_index", "index", "task_index"}
    if missing := required.difference(table.column_names):
        errors.append(f"parquet 缺少列: {sorted(missing)}")
        return {
            "status": "FAIL", "bundle": str(bundle),
            "summary": {
                "episodes": total_episodes, "frames": total_frames, "fps": fps,
                "action_mode": mode, "leading_idle_frames": 0,
                "trailing_idle_frames": 0, "max_action_joint_step_rad": 0.0,
                "max_action_gripper_step_m": 0.0, "max_state_joint_step_rad": 0.0,
                "errors": len(errors),
                "warnings": len(warnings),
            },
            "source_verification": None, "videos": {}, "statistics": {},
            "errors": errors, "warnings": warnings, "episodes": [],
        }
    columns = {name: np.asarray(table[name].combine_chunks().to_pylist()) for name in required}
    actions = np.asarray(columns["action"], dtype=np.float32)
    states = np.asarray(columns["observation.state"], dtype=np.float32)
    episode = columns["episode_index"].astype(np.int64)
    frame = columns["frame_index"].astype(np.int64)
    index = columns["index"].astype(np.int64)
    task_index = columns["task_index"].astype(np.int64)
    timestamps = columns["timestamp"].astype(np.float64)
    rows = len(table)
    if rows != total_frames:
        errors.append(f"parquet {rows} 行与 info.total_frames={total_frames} 不一致")
    if actions.shape != (rows, 14) or states.shape != (rows, 14):
        errors.append("action/state parquet 维度错误")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        errors.append("action/state 含 NaN 或 Inf")
    else:
        errors.extend(validate_ranges(states, "observation.state"))
        if mode == "absolute_joint":
            errors.extend(validate_ranges(actions, "action"))
    if not np.array_equal(index, np.arange(rows)):
        errors.append("全局 index 不连续")
    if not np.array_equal(np.unique(episode), np.arange(total_episodes)):
        errors.append("episode_index 不连续")
    if not np.array_equal(np.unique(task_index), np.asarray([0])):
        errors.append("单任务数据的 task_index 必须全部为 0")

    statistics = {}
    for key, values in (("action", actions), ("observation.state", states)):
        if values.shape != (rows, 14) or not len(values) or not np.isfinite(values).all():
            continue
        actual = {
            "min": values.min(axis=0), "max": values.max(axis=0),
            "mean": values.astype(np.float64).mean(axis=0),
            "std": values.astype(np.float64).std(axis=0),
        }
        maximum_error = 0.0
        for name, calculated in actual.items():
            stored = np.asarray(stats.get(key, {}).get(name, []), dtype=np.float64)
            if stored.shape != (14,):
                continue
            difference = float(np.max(np.abs(stored - calculated)))
            maximum_error = max(maximum_error, difference)
            if not statistics_close(stored, calculated, name, values):
                errors.append(f"stats {key}/{name} 与 parquet 全量重算不一致，最大误差 {difference:g}")
        statistics[key] = {"max_recompute_error": maximum_error}

    task_path = root / "meta/tasks.parquet"
    if not task_path.is_file():
        errors.append("缺少 meta/tasks.parquet")
    else:
        tasks = pq.read_table(task_path).to_pydict()
        task_texts = tasks.get("task", [])
        valid_task_text = (
            len(task_texts) == 1
            and isinstance(task_texts[0], str)
            and bool(task_texts[0].strip())
        )
        if tasks.get("task_index") != [0] or not valid_task_text:
            errors.append("tasks.parquet 必须包含唯一的非空 task_index=0")
        elif settings.get("task") and task_texts[0] != settings["task"]:
            errors.append("tasks.parquet 的任务文本与 processing.json 不一致")

    episode_meta_files = sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not episode_meta_files:
        errors.append("缺少 episode 元数据 parquet")
    else:
        episode_meta = pa.concat_tables(
            [pq.read_table(path) for path in episode_meta_files], promote_options="default"
        ).to_pydict()
        meta_indices = np.asarray(episode_meta.get("episode_index", []), dtype=np.int64)
        lengths = np.asarray(episode_meta.get("length", []), dtype=np.int64)
        starts = np.asarray(episode_meta.get("dataset_from_index", []), dtype=np.int64)
        stops = np.asarray(episode_meta.get("dataset_to_index", []), dtype=np.int64)
        expected_lengths = np.bincount(episode, minlength=total_episodes)
        expected_starts = np.cumsum(np.r_[0, expected_lengths[:-1]])
        if not np.array_equal(meta_indices, np.arange(total_episodes)):
            errors.append("episode 元数据索引不连续")
        if not np.array_equal(lengths, expected_lengths):
            errors.append("episode 元数据 length 与 parquet 不一致")
        if not np.array_equal(starts, expected_starts) or not np.array_equal(stops, expected_starts + expected_lengths):
            errors.append("episode 元数据全局起止索引与 parquet 不一致")

    episode_results = []
    for ep in range(total_episodes):
        positions = np.flatnonzero(episode == ep)
        ep_errors = []
        if not len(positions) or positions[-1] - positions[0] + 1 != len(positions):
            ep_errors.append("parquet 中不是连续区块")
        expected_frame = np.arange(len(positions))
        if not np.array_equal(frame[positions], expected_frame):
            ep_errors.append("frame_index 不连续")
        if not np.allclose(timestamps[positions], expected_frame / fps, rtol=0, atol=2e-5):
            ep_errors.append("timestamp 未严格落在固定 FPS 网格")
        ep_actions, ep_states = actions[positions], states[positions]
        leading = edge_repeats(ep_actions, ep_states, False) if len(positions) > 1 else 0
        trailing = edge_repeats(ep_actions, ep_states, True) if len(positions) > 1 else 0
        if leading:
            ep_errors.append(f"开头有 {leading} 个完全静止相邻帧")
        if trailing:
            ep_errors.append(f"尾部有 {trailing} 个完全静止相邻帧")
        action_steps = ep_actions if mode == "delta_joint" else np.diff(ep_actions, axis=0)
        state_steps = np.diff(ep_states, axis=0)
        episode_results.append({
            "episode": ep, "rows": len(positions), "leading_idle_frames": leading,
            "trailing_idle_frames": trailing,
            "max_action_joint_step_rad": float(
                np.max(np.abs(action_steps[:, np.r_[0:6, 7:13]]))
            ) if len(action_steps) else 0.0,
            "max_action_gripper_step_m": float(
                np.max(np.abs(action_steps[:, [6, 13]]))
            ) if len(action_steps) else 0.0,
            "max_state_joint_step_rad": float(
                np.max(np.abs(state_steps[:, np.r_[0:6, 7:13]]))
            ) if len(state_steps) else 0.0,
            "errors": ep_errors,
        })
        errors.extend(f"episode {ep}: {error}" for error in ep_errors)

    source_metrics = None
    source_raw = settings.get("input_dirs") or settings.get("input_dir") or plan.get("input_dirs") or plan.get("input_dir")
    if source_raw:
        plan_files = plan.get("files", [])
        source_files, source_dirs = source_paths(source_raw, plan_files, source_override)
        if len(source_files) != total_episodes:
            errors.append(f"源 HDF5 数量 {len(source_files)} 与 episode 数 {total_episodes} 不一致")
        else:
            global_action_error = 0.0
            global_state_error = 0.0
            max_source_dt = 0.0
            trimmed_idle = 0
            home_frames_in_output = 0
            for ep, path in enumerate(source_files):
                positions = np.flatnonzero(episode == ep)
                with h5py.File(path, "r") as handle:
                    demo = handle["data/demo_0"]
                    file_plan = plan_files[ep] if ep < len(plan_files) else {}
                    source_fps = int(file_plan.get("source_fps") or settings.get("source_fps") or 0)
                    if source_fps <= 0:
                        source_timestamps = np.asarray(demo["timestamps"], dtype=np.float64)
                        source_fps = int(round(1 / np.median(np.diff(source_timestamps))))
                    if source_fps <= 0 or source_fps % fps:
                        errors.append(f"episode {ep}: 源 FPS 无法整除目标 FPS")
                        continue
                    stride = source_fps // fps
                    selected, idle = source_indices(demo, stride)
                    trimmed_idle += idle
                    raw_actions = np.asarray(demo["actions"][selected], dtype=np.float32)
                    expected_actions = raw_actions
                    expected_selected = selected
                    if mode == "delta_joint":
                        expected_actions = np.diff(raw_actions, axis=0)
                        expected_selected = selected[:-1]
                    expected_states = np.asarray(demo["states"][expected_selected], dtype=np.float32)
                    if len(positions) != len(expected_selected):
                        errors.append(
                            f"episode {ep}: 输出 {len(positions)} 帧，按源数据应为 {len(expected_selected)} 帧"
                        )
                        continue
                    action_error = float(np.max(np.abs(actions[positions] - expected_actions)))
                    state_error = float(np.max(np.abs(states[positions] - expected_states)))
                    global_action_error = max(global_action_error, action_error)
                    global_state_error = max(global_state_error, state_error)
                    if action_error > 1e-6 or state_error > 1e-6:
                        errors.append(
                            f"episode {ep}: 源数据语义不一致 action={action_error:g}, state={state_error:g}"
                        )
                    phase = np.asarray(demo["trajectory_phase"])[expected_selected]
                    home_count = int(np.count_nonzero(phase != 0))
                    home_frames_in_output += home_count
                    if home_count:
                        errors.append(f"episode {ep}: 输出包含 home 帧")
                    source_ts = np.asarray(demo["timestamps"], dtype=np.float64)[selected]
                    if len(source_ts) > 1:
                        ep_max_dt = float(np.max(np.diff(source_ts)))
                        max_source_dt = max(max_source_dt, ep_max_dt)
                        if ep_max_dt > 1.5 / fps:
                            errors.append(f"episode {ep}: 相邻源采样间隔 {ep_max_dt:.6f}s 过大")
                    else:
                        ep_max_dt = 0.0
                    episode_results[ep].update(
                        source_file=path.name, source_fps=source_fps,
                        max_source_dt_ms=ep_max_dt * 1000,
                        action_error=action_error, state_error=state_error,
                        home_frames=home_count, trimmed_source_idle_frames=idle,
                    )
            source_metrics = {
                "directory": str(source_dirs[0]) if len(source_dirs) == 1 else None,
                "directories": [str(path) for path in source_dirs], "files": len(source_files),
                "trimmed_idle_frames": trimmed_idle,
                "max_action_error": global_action_error,
                "max_state_error": global_state_error,
                "max_selected_source_dt_ms": max_source_dt * 1000,
                "home_frames_in_output": home_frames_in_output,
            }
    else:
        warnings.append("未找到源 HDF5，跳过 absolute/delta 数值语义与 home 回溯")

    videos = {}
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        feature = info.get("features", {}).get(key, {})
        paths = sorted((root / "videos" / key).glob("chunk-*/file-*.mp4"))
        if feature.get("dtype") != "video" or not paths:
            errors.append(f"缺少视频特征 {key}")
            continue
        probes = [probe_video(path) for path in paths]
        frames_found = sum(item["frames"] or 0 for item in probes)
        if any(item["frames"] is None for item in probes):
            warnings.append(f"{key} 有视频未提供 nb_frames")
        elif frames_found != total_frames:
            errors.append(f"{key} 视频 {frames_found} 帧，与表数据 {total_frames} 不一致")
        expected_shape = feature.get("shape", [])
        for item in probes:
            if expected_shape and [item["height"], item["width"]] != expected_shape[:2]:
                errors.append(f"{key} 视频尺寸错误")
            if abs(item["fps"] - fps) > 0.05:
                errors.append(f"{key} 视频 FPS 错误")
        videos[key] = {"files": len(paths), "frames": frames_found}

    # Source checks also belong to episode rows, including deletion eligibility.
    for item in episode_results:
        prefix = f"episode {item['episode']}: "
        item["errors"] = list(dict.fromkeys([
            *item["errors"], *(error[len(prefix):] for error in errors if error.startswith(prefix))
        ]))
    summary = {
        "episodes": total_episodes, "frames": total_frames, "fps": fps,
        "action_mode": mode,
        "leading_idle_frames": sum(item["leading_idle_frames"] for item in episode_results),
        "trailing_idle_frames": sum(item["trailing_idle_frames"] for item in episode_results),
        "max_action_joint_step_rad": max(
            (item["max_action_joint_step_rad"] for item in episode_results), default=0.0
        ),
        "max_action_gripper_step_m": max(
            (item["max_action_gripper_step_m"] for item in episode_results), default=0.0
        ),
        "max_state_joint_step_rad": max(
            (item["max_state_joint_step_rad"] for item in episode_results), default=0.0
        ),
        "errors": len(errors), "warnings": len(warnings),
    }
    return {
        "status": "PASS" if not errors else "FAIL", "bundle": str(bundle),
        "summary": summary, "source_verification": source_metrics,
        "videos": videos, "statistics": statistics,
        "errors": errors, "warnings": warnings,
        "episodes": episode_results,
    }


def analyze(raw: str, source_override: str | None) -> dict[str, Any]:
    """Malformed datasets produce a usable failure report, including strict JSON."""
    from easy_embodied_deploy.http import json_safe
    try:
        return json_safe(_analyze(raw, source_override))
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, OSError) as exc:
        return {
            "status": "FAIL", "bundle": str(Path(raw).expanduser().resolve()),
            "summary": {"episodes": 0, "frames": 0, "fps": 0, "action_mode": None,
                        "leading_idle_frames": 0, "trailing_idle_frames": 0,
                        "max_action_joint_step_rad": 0.0, "max_action_gripper_step_m": 0.0,
                        "max_state_joint_step_rad": 0.0, "errors": 1, "warnings": 0},
            "source_verification": None, "videos": {}, "statistics": {},
            "errors": [f"数据结构无法解析：{type(exc).__name__}: {exc}"],
            "warnings": [], "episodes": [],
        }


def print_report(report: dict[str, Any]) -> None:
    s = report["summary"]
    print(f"LeRobot quality report: {report['bundle']}")
    print(
        f"mode={s['action_mode']} fps={s['fps']} episodes={s['episodes']} "
        f"frames={s['frames']}"
    )
    source = report.get("source_verification")
    if source:
        print(
            f"source exactness: action={source['max_action_error']:.3g}, "
            f"state={source['max_state_error']:.3g}, home_in_output={source['home_frames_in_output']}, "
            f"max_dt={source['max_selected_source_dt_ms']:.3f} ms, "
            f"trimmed_tail={source['trimmed_idle_frames']}"
        )
    for key, values in report.get("statistics", {}).items():
        print(f"stats recompute: {key} max_error={values['max_recompute_error']:.3g}")
    for key, values in report.get("videos", {}).items():
        print(f"video: {key} files={values['files']} frames={values['frames']}")
    print(
        f"edge idle: leading={s['leading_idle_frames']}, trailing={s['trailing_idle_frames']}"
    )
    print(
        f"max step: action_joint={s['max_action_joint_step_rad']:.6f} rad, "
        f"action_gripper={s['max_action_gripper_step_m']:.6f} m, "
        f"state_joint={s['max_state_joint_step_rad']:.6f} rad"
    )
    for error in report["errors"]:
        print(f"ERROR: {error}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"{report['status']}: errors={s['errors']}, warnings={s['warnings']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="converter bundle or inner lerobot directory")
    parser.add_argument("--source-dir", help="override source HDF5 directory")
    parser.add_argument("--json", dest="json_path", help="write the full JSON report")
    args = parser.parse_args()
    report = analyze(args.path, args.source_dir)
    print_report(report)
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
