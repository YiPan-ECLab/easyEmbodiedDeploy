"""Validated Piper HDF5 -> LeRobot conversion.

The converter deliberately owns the Piper action semantics instead of relying on
filename conventions: raw actions are absolute leader targets, task/home phases
are explicit, and delta actions are built only after task selection/resampling.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import time
import threading
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
AXES = [f"{side}_{axis}" for side in ("left", "right")
        for axis in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")]
REPORT_NAME = "processing.json"
TASK_PHASE = 0
HOME_PHASE = 1
ACTION_MODES = ("absolute_joint", "delta_joint")
GAP_POLICIES = ("compact", "reject")
SOURCE_ACTION_LAYOUT = (
    "left_j1_rad,left_j2_rad,left_j3_rad,left_j4_rad,left_j5_rad,left_j6_rad,"
    "left_gripper_m,right_j1_rad,right_j2_rad,right_j3_rad,right_j4_rad,"
    "right_j5_rad,right_j6_rad,right_gripper_m"
)
JOINT_LIMITS_RAD = np.asarray(
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
GRIPPER_LIMITS_M = (0.0, 0.1)
HARDWARE_TOLERANCE = 1e-6
MAX_GAP_FACTOR = 1.5
MAX_CAMERA_SKEW_S = 0.5
STATS_KEYS = ("q01", "q99", "mean", "std", "min", "max")
# Keep HDF5 reads bounded while avoiding one compressed-image read per frame.
# A block is large enough to amortize HDF5 hyperslab overhead, but small enough
# for the largest expected Piper episode on the CPU-only conversion path.
READ_BLOCK_SIZE = 256
MAX_READ_SPAN = 512


class ProcessingCancelled(RuntimeError):
    """Raised when the user cancels between safe processing checkpoints."""


def available_cpu_count() -> int:
    """Return the number of CPUs available to this process.

    ``os.cpu_count()`` reports host CPUs even when a service is restricted by a
    cpuset/container.  Prefer the process affinity when available and always
    return at least one usable worker.
    """
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, int(count))


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@dataclass
class Settings:
    input_dir: str
    output_root: str | None = None
    task: str = "Remove the shipping label and place it on the package on the table"
    target_fps: int = 10
    source_fps: int | None = None
    action_mode: str = "delta_joint"
    repo_id: str | None = None
    limit: int = 0
    gap_policy: str = "reject"
    input_dirs: list[str] | None = None


def canonical_action_mode(value: str) -> str:
    """Accept the old spelling while writing one unambiguous representation."""
    return "absolute_joint" if value == "absolute" else value


def canonical_settings(settings: Settings) -> Settings:
    raw_dirs = settings.input_dirs
    if isinstance(raw_dirs, str):
        raw_dirs = raw_dirs.splitlines()
    if raw_dirs is None:
        raw_dirs = [settings.input_dir]
    dirs = []
    seen = set()
    for value in raw_dirs:
        value = str(value).strip()
        if value and value not in seen:
            dirs.append(value)
            seen.add(value)
    if not dirs:
        dirs = [settings.input_dir]
    return replace(
        settings,
        input_dir=dirs[0],
        input_dirs=dirs,
        action_mode=canonical_action_mode(settings.action_mode),
    )


def input_directories(settings: Settings) -> list[Path]:
    """Resolve the ordered, de-duplicated HDF5 directories in a setting."""
    settings = canonical_settings(settings)
    paths = []
    seen = set()
    for raw in (settings.input_dirs or [settings.input_dir]):
        path = resolve_path(raw)
        if path not in seen:
            paths.append(path)
            seen.add(path)
    for path in paths:
        if not path.is_dir():
            raise ValueError(f"输入目录不存在：{path}")
    return paths


def _text_attr(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _bool_attr(value) -> bool:
    if isinstance(value, (str, bytes)):
        return _text_attr(value).strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _task_sample_indices(demo: h5py.Group, stride: int) -> tuple[np.ndarray, int, int]:
    """Select task frames, excluding trailing idle and homing frames."""
    indices = np.arange(0, len(demo["actions"]), stride)
    if "trajectory_phase" not in demo:
        raise ValueError(
            "缺少 trajectory_phase；为避免把回零过程混入训练，"
            "只接受显式标注 0=task,1=home 的录制文件"
        )

    phase = np.asarray(demo["trajectory_phase"])
    if phase.shape != (len(demo["actions"]),):
        raise ValueError("trajectory_phase 必须为 (T,)")
    sampled_phase = phase[indices]
    invalid = np.setdiff1d(np.unique(phase), (TASK_PHASE, HOME_PHASE))
    if len(invalid):
        raise ValueError(f"trajectory_phase 含未知值：{invalid.tolist()}")
    if phase[0] != TASK_PHASE:
        raise ValueError("trajectory_phase 必须从 task(0) 开始")
    home = np.flatnonzero(phase == HOME_PHASE)
    if len(home) and not np.all(phase[home[0] :] == HOME_PHASE):
        raise ValueError("trajectory_phase 必须是连续 task 前缀和可选 home 后缀")
    task_indices = indices[sampled_phase == TASK_PHASE]
    excluded_home_frames = len(indices) - len(task_indices)
    if not len(task_indices):
        raise ValueError("trajectory_phase 中没有 task(0) 帧")
    if len(task_indices) > 1 and np.any(np.diff(task_indices) != stride):
        raise ValueError("task phase 必须是连续区间，不能跨非任务片段计算 action")

    task_end = int(home[0]) if len(home) else len(phase)
    last_active = task_end - 1
    actions = demo["actions"]
    while (
        last_active > 0
        and np.array_equal(actions[last_active], actions[last_active - 1])
    ):
        last_active -= 1
    trimmed_idle_frames = task_end - last_active - 1
    # Keep the first fixed-grid sample at or after the final target change. This
    # preserves the endpoint at lower FPS without retaining the later wait.
    endpoint = int(np.searchsorted(task_indices, last_active, side="left"))
    if endpoint < len(task_indices):
        task_indices = task_indices[: endpoint + 1]
    return task_indices, excluded_home_frames, trimmed_idle_frames


def _validate_piper_range(values: np.ndarray, name: str) -> None:
    for offset, side in ((0, "left"), (7, "right")):
        joints = values[:, offset : offset + 6]
        bad = (joints < JOINT_LIMITS_RAD[:, 0] - HARDWARE_TOLERANCE) | (
            joints > JOINT_LIMITS_RAD[:, 1] + HARDWARE_TOLERANCE
        )
        if np.any(bad):
            frame, joint = np.argwhere(bad)[0]
            raise ValueError(
                f"{name} 超出 Piper 限位：frame={frame}, {side}_j{joint + 1}="
                f"{joints[frame, joint]:.6f} rad"
            )
    grippers = values[:, [6, 13]]
    bad = (grippers < GRIPPER_LIMITS_M[0] - HARDWARE_TOLERANCE) | (
        grippers > GRIPPER_LIMITS_M[1] + HARDWARE_TOLERANCE
    )
    if np.any(bad):
        frame, side = np.argwhere(bad)[0]
        raise ValueError(
            f"{name} 夹爪超出 [0,0.1] m：frame={frame}, "
            f"{('left', 'right')[side]}={grippers[frame, side]:.6f} m"
        )


def _validate_source_contract(demo: h5py.Group) -> None:
    actions = demo["actions"]
    states = demo["states"]
    if _bool_attr(actions.attrs.get("normalized", False)) or _bool_attr(states.attrs.get("normalized", False)):
        raise ValueError("actions/states 必须是未归一化的 Piper 物理量")
    layout = _text_attr(actions.attrs.get("layout", ""))
    if not layout:
        raise ValueError("actions 缺少 layout 属性，无法确认 14 维轴顺序")
    if layout != SOURCE_ACTION_LAYOUT:
        raise ValueError(f"actions 轴顺序不符合双臂 Piper 契约：{layout}")
    units = _text_attr(actions.attrs.get("units", ""))
    if units != "rad x6,m per arm":
        raise ValueError(f"actions units 必须为 rad x6,m per arm：{units}")
    state_layout = _text_attr(states.attrs.get("layout", ""))
    expected_state = "left_j1..j6,left_gripper,right_j1..j6,right_gripper"
    if not state_layout:
        raise ValueError("states 缺少 layout 属性，无法确认 14 维轴顺序")
    if state_layout != expected_state:
        raise ValueError(f"states 轴顺序不符合双臂 Piper 契约：{state_layout}")
    state_units = _text_attr(states.attrs.get("units", ""))
    if state_units != "rad x6,m per arm":
        raise ValueError(f"states units 必须为 rad x6,m per arm：{state_units}")


def _joint_l2(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values[:, np.r_[0:6, 7:13]], axis=1)


def inspect_file(path: Path, settings: Settings) -> dict:
    settings = canonical_settings(settings)
    with h5py.File(path, "r") as f:
        demo = f["data/demo_0"]
        actions, states = demo["actions"], demo["states"]
        n = len(actions)
        if n < 2 or actions.shape != (n, 14) or states.shape != (n, 14):
            raise ValueError(f"{path.name}: 需要至少两帧，且 actions/states 均为 (T,14)")
        if int(demo.attrs.get("num_samples", n)) != n:
            raise ValueError(f"{path.name}: num_samples 与实际帧数不符")
        if _bool_attr(f.attrs.get("normalized", False)):
            raise ValueError(f"{path.name}: 需要未归一化的原始控制量")
        _validate_source_contract(demo)
        representation = _text_attr(demo["actions"].attrs.get("representation", ""))
        if "delta" in representation or "delta" in _text_attr(f.attrs.get("actions_representation", "")):
            raise ValueError(f"{path.name}: 输入已经是增量动作，不能再次差分")
        t = np.asarray(demo["timestamps"], dtype=np.float64)
        if t.shape != (n,) or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
            raise ValueError(f"{path.name}: timestamps 必须有限且严格递增")
        dt = np.diff(t)
        measured = float(1 / np.median(dt))
        source_fps = settings.source_fps or int(round(measured))
        if source_fps <= 0 or abs(measured - source_fps) / source_fps > .02:
            raise ValueError(f"{path.name}: 时间戳估计 {measured:.3f} Hz，与源帧率 {source_fps} 不符")
        if settings.target_fps > source_fps or source_fps % settings.target_fps:
            raise ValueError(f"{path.name}: 目标 {settings.target_fps} Hz 必须整除源帧率 {source_fps} Hz，不支持升采样")
        arrays = {}
        for key in ("actions", "states"):
            arrays[key] = np.asarray(demo[key], dtype=np.float64)
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{path.name}: {key} 含 NaN/Inf")
            _validate_piper_range(arrays[key], f"{path.name}/{key}")
        shapes = []
        for camera in CAMERAS:
            image = demo[f"obs/{camera}"]
            if image.ndim != 4 or image.shape[0] != n or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{path.name}: {camera} 需要 (T,H,W,3) uint8")
            shapes.append(list(image.shape[1:]))
        if "camera_timestamps" in demo:
            camera_timestamps = np.asarray(demo["camera_timestamps"], dtype=np.float64)
            if camera_timestamps.shape != (n, len(CAMERAS)) or not np.isfinite(camera_timestamps).all():
                raise ValueError(f"{path.name}: camera_timestamps 必须为有限的 (T,3)")
            if np.any(np.diff(camera_timestamps, axis=0) < 0):
                raise ValueError(f"{path.name}: camera_timestamps 不能倒退")
            max_camera_skew = float(np.max(np.abs(t[:, None] - camera_timestamps)))
            if max_camera_skew > MAX_CAMERA_SKEW_S:
                raise ValueError(
                    f"{path.name}: 相机与机器人时间差 {max_camera_skew:.3f}s "
                    f"超过 {MAX_CAMERA_SKEW_S:g}s"
                )
        else:
            max_camera_skew = None
        stride = source_fps // settings.target_fps
        selected, excluded_home_frames, trimmed_idle_frames = _task_sample_indices(demo, stride)
        sampled_frames = len(selected)
        output_frames = sampled_frames - int(settings.action_mode == "delta_joint")
        if output_frames < 1:
            raise ValueError(f"{path.name}: 降采样后不足以计算增量")
        selected_dt = np.diff(t[selected])
        nominal_dt = 1 / settings.target_fps
        gap_positions = np.flatnonzero(
            (selected_dt < nominal_dt / MAX_GAP_FACTOR)
            | (selected_dt > nominal_dt * MAX_GAP_FACTOR)
        )
        gap_count = len(gap_positions)
        max_gap = float(selected_dt[gap_positions].max()) if gap_count else 0.0
        action_steps = np.diff(arrays["actions"][selected], axis=0)
        max_gap_action = (
            float(_joint_l2(action_steps[gap_positions]).max()) if gap_count else 0.0
        )
        if gap_count and settings.gap_policy == "reject":
            raise ValueError(
                f"{path.name}: task 段有 {gap_count} 个采样间隔异常，最大 {max_gap:.3f}s；"
                "严格模式拒绝改变这些间隔"
            )
        warnings = []
        if gap_count:
            warnings.append(
                f"task 段 {gap_count} 个采样间隔异常将被映射到固定 {settings.target_fps} Hz；"
                f"最大间隔 {max_gap:.3f}s，跨异常间隔动作 {max_gap_action:.4f} rad L2"
            )
        return dict(name=path.name, frames=n, source_fps=source_fps,
                    measured_fps=round(measured, 4), stride=stride,
                    sampled_frames=sampled_frames, output_frames=output_frames,
                    excluded_home_frames=excluded_home_frames,
                    trimmed_idle_frames=trimmed_idle_frames,
                    source_task_frames=int(np.count_nonzero(np.asarray(demo["trajectory_phase"]) == TASK_PHASE)),
                    source_home_frames=int(np.count_nonzero(np.asarray(demo["trajectory_phase"]) == HOME_PHASE)),
                    camera_shapes=shapes, max_camera_skew_s=max_camera_skew,
                    gaps=gap_count, max_gap_s=round(max_gap, 6),
                    max_gap_action_joint_l2=round(max_gap_action, 6), warnings=warnings)


def scan(settings: Settings) -> dict:
    settings = canonical_settings(settings)
    if settings.target_fps <= 0 or settings.limit < 0:
        raise ValueError("目标帧率必须为正整数，文件上限不能为负数")
    if settings.action_mode not in ACTION_MODES:
        raise ValueError(f"动作模式必须为 {' 或 '.join(ACTION_MODES)}")
    if settings.gap_policy not in GAP_POLICIES:
        raise ValueError(f"时间空档策略必须为 {' 或 '.join(GAP_POLICIES)}")
    if not settings.task.strip():
        raise ValueError("请填写任务描述")
    sources = input_directories(settings)
    paths = [
        (source, path)
        for source in sources
        for path in sorted(source.iterdir())
        if path.is_file() and path.suffix.lower() in (".h5", ".hdf5")
    ]
    if settings.limit:
        paths = paths[:settings.limit]
    if not paths:
        raise ValueError("输入目录中没有 .hdf5/.h5 文件（只扫描各目录当前层）")
    files = []
    for source, path in paths:
        info = inspect_file(path, settings)
        info["source_dir"] = portable_path(source)
        files.append(info)
    if any(x["camera_shapes"] != files[0]["camera_shapes"] for x in files):
        raise ValueError("不同 episode 的相机尺寸不一致")
    source_label = sources[0].name if len(sources) == 1 else f"{sources[0].name}_merged"
    default_out = Path("datasets/lerobot") / f"{source_label}_{settings.target_fps}hz_{settings.action_mode}"
    out = resolve_path(settings.output_root or default_out)
    if any(out == source or out in source.parents or source in out.parents for source in sources):
        raise ValueError("输出目录必须与输入目录分开，不能互相包含")
    warnings = [f"{item['name']}: {warning}" for item in files for warning in item["warnings"]]
    return dict(input_dir=portable_path(sources[0]), input_dirs=[portable_path(path) for path in sources],
                output_root=portable_path(out), files=files,
                episodes=len(files), input_frames=sum(x["frames"] for x in files),
                output_frames=sum(x["output_frames"] for x in files),
                gaps=sum(x["gaps"] for x in files),
                warnings=warnings,
                repo_id=settings.repo_id or f"local/piper_{settings.target_fps}hz_{settings.action_mode}")


def make_features(shapes: list) -> dict:
    features = {key: {"dtype": "float32", "shape": (14,), "names": {"axes": AXES}}
                for key in ("observation.state", "action")}
    for camera, shape in zip(CAMERAS, shapes):
        features[f"observation.images.{camera}"] = dict(dtype="video", shape=tuple(shape),
                                                       names=["height", "width", "channels"])
    return features


def _read_parquet_rows(data_root: Path) -> tuple[dict[str, np.ndarray], int]:
    """Read the small scalar/vector columns needed for structural validation."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(data_root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise ValueError("LeRobot 输出缺少 data/*.parquet")
    tables = [pq.read_table(path) for path in files]
    try:
        table = pa.concat_tables(tables, promote_options="default")
    except TypeError:  # pyarrow < 14
        table = pa.concat_tables(tables, promote=True)
    required = {"observation.state", "action", "timestamp", "frame_index",
                "episode_index", "index", "task_index"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"LeRobot parquet 缺少列：{sorted(missing)}")
    values = {key: np.asarray(table[key].combine_chunks().to_pylist())
              for key in required}
    return values, table.num_rows


def _ffprobe_video(path: Path) -> dict:
    if shutil.which("ffprobe") is None:
        return {"available": False}
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
               "-of", "json", str(path)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"无法读取视频 {path.name}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"视频没有视频流：{path}")
    stream = streams[0]
    fps = float(Fraction(stream.get("r_frame_rate", "0/1")))
    frames = stream.get("nb_frames")
    return {"available": True, "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)), "fps": fps,
            "frames": int(frames) if frames not in (None, "N/A") else None}


def validate_lerobot_bundle(bundle_root: str | Path, *, expected: dict | None = None) -> dict:
    """Validate a finalized local LeRobot bundle without loading a model/GPU.

    Quantiles are checked for presence and ordering only. Their values are left
    to LeRobot's official ``finalize`` implementation, including its
    episode-wise quantile aggregation semantics.
    """
    root = resolve_path(bundle_root)
    if (root / "lerobot").is_dir() and (root / "lerobot/meta/info.json").is_file():
        root = root / "lerobot"
    if expected is None:
        # A converter output stores the action representation next to the
        # dataset.  Use it for standalone validation when available.
        report_path = root.parent / REPORT_NAME
        if report_path.is_file():
            with contextlib.suppress(Exception):
                report = json.loads(report_path.read_text(encoding="utf-8"))
                expected = dict(report.get("settings", {}))
    info_path, stats_path = root / "meta/info.json", root / "meta/stats.json"
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict = {}
    if not info_path.is_file() or not stats_path.is_file():
        return {"valid": False, "errors": ["缺少 meta/info.json 或 meta/stats.json"],
                "warnings": [], "metrics": {}}
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        fps = int(info["fps"])
        total_episodes = int(info["total_episodes"])
        total_frames = int(info["total_frames"])
        if fps <= 0 or total_episodes <= 0 or total_frames <= 0:
            raise ValueError("info 中 fps/episode/frame 总数必须为正数")
        if info.get("robot_type") != "piper_dual":
            errors.append("robot_type 必须为 piper_dual")
        if int(info.get("total_tasks", 1)) != 1:
            errors.append("当前处理器只支持单任务数据集")
        features = info.get("features", {})
        for key in ("observation.state", "action"):
            feature = features.get(key, {})
            if feature.get("dtype") != "float32" or feature.get("shape") != [14]:
                errors.append(f"{key} feature 不是 float32[14]")
            axes = feature.get("names", {}).get("axes")
            if axes != AXES:
                errors.append(f"{key} 的 axes 顺序不符合 Piper 契约")
            stat = stats.get(key)
            if not isinstance(stat, dict):
                errors.append(f"stats 缺少 {key}")
                continue
            for name in STATS_KEYS:
                values = np.asarray(stat.get(name, []), dtype=np.float64)
                if values.shape != (14,) or not np.isfinite(values).all():
                    errors.append(f"{key}/{name} 必须是有限的 14 维数组")
            if np.asarray(stat.get("count", []), dtype=np.int64).tolist() != [total_frames]:
                errors.append(f"{key}/count 不等于 total_frames={total_frames}")
            try:
                minimum, q01, q99, maximum = (np.asarray(stat[name], dtype=np.float64)
                                              for name in ("min", "q01", "q99", "max"))
                if np.any(q01 < minimum - 1e-6) or np.any(q99 > maximum + 1e-6):
                    errors.append(f"{key} 分位数超出 min/max 范围")
                if np.any(q01 > q99 + 1e-6):
                    errors.append(f"{key} q01 大于 q99")
                quantile_names = ("min", "q01", "q10", "q50", "q90", "q99", "max")
                if all(name in stat for name in quantile_names):
                    quantiles = [np.asarray(stat[name], dtype=np.float64) for name in quantile_names]
                    if any(np.any(left > right + 1e-6)
                           for left, right in zip(quantiles, quantiles[1:])):
                        errors.append(f"{key} 分位数顺序不递增")
            except (KeyError, ValueError):
                pass
        columns, rows = _read_parquet_rows(root)
        metrics["rows"] = rows
        if rows != total_frames:
            errors.append(f"parquet 行数 {rows} != info.total_frames {total_frames}")
        for key in ("observation.state", "action"):
            values = np.asarray(columns[key], dtype=np.float64)
            if values.shape != (rows, 14) or not np.isfinite(values).all():
                errors.append(f"{key} 数据不是有限的 [rows,14]")
            elif key == "observation.state":
                try:
                    _validate_piper_range(values, key)
                except ValueError as exc:
                    errors.append(str(exc))
        index = columns["index"].astype(np.int64, copy=False)
        episode = columns["episode_index"].astype(np.int64, copy=False)
        frame = columns["frame_index"].astype(np.int64, copy=False)
        task_index = columns["task_index"].astype(np.int64, copy=False)
        timestamp = columns["timestamp"].astype(np.float64, copy=False)
        if not np.array_equal(index, np.arange(rows, dtype=np.int64)):
            errors.append("index 不是从 0 连续递增")
        if not np.array_equal(np.unique(episode), np.arange(total_episodes, dtype=np.int64)):
            errors.append("episode_index 不是连续的 0..total_episodes-1")
        if rows and np.any(np.diff(episode) < 0):
            errors.append("parquet 中 episode_index 顺序倒退")
        if not np.array_equal(np.unique(task_index), np.asarray([0], dtype=np.int64)):
            errors.append("单任务数据集的 task_index 必须全部为 0")
        if not np.isfinite(timestamp).all():
            errors.append("timestamp 含 NaN/Inf")
        for ep in range(total_episodes):
            positions = np.flatnonzero(episode == ep)
            expected_frames = np.arange(len(positions), dtype=np.int64)
            if len(positions) and positions[-1] - positions[0] + 1 != len(positions):
                errors.append(f"episode {ep} 在 parquet 中不是连续区块")
            if not np.array_equal(frame[positions], expected_frames):
                errors.append(f"episode {ep} frame_index 不是从 0 连续递增")
            if len(positions) and not np.allclose(timestamp[positions], expected_frames / fps,
                                                  rtol=0, atol=2e-5):
                errors.append(f"episode {ep} timestamp 与 fps={fps} 不一致")
        action_mode = canonical_action_mode(str((expected or {}).get("action_mode", "")))
        action_values = np.asarray(columns["action"], dtype=np.float64)
        if action_mode == "absolute_joint" and action_values.shape == (rows, 14):
            try:
                _validate_piper_range(action_values, "action")
            except ValueError as exc:
                errors.append(str(exc))
        if action_values.shape == (rows, 14):
            metrics["max_action_joint_l2"] = float(np.max(_joint_l2(action_values)))
        statistics = {}
        for key in ("observation.state", "action"):
            values = np.asarray(columns[key], dtype=np.float64)
            if values.shape != (rows, 14) or not len(values) or not np.isfinite(values).all():
                continue
            calculated = {
                "min": values.min(axis=0), "max": values.max(axis=0),
                "mean": values.mean(axis=0), "std": values.std(axis=0),
            }
            maximum_error = 0.0
            for name, expected_values in calculated.items():
                stored = np.asarray(stats.get(key, {}).get(name, []), dtype=np.float64)
                if stored.shape != (14,):
                    continue
                difference = float(np.max(np.abs(stored - expected_values)))
                maximum_error = max(maximum_error, difference)
                if not np.allclose(stored, expected_values, rtol=2e-5, atol=2e-5):
                    errors.append(f"{key}/{name} 与 parquet 全量重算不一致")
            statistics[key] = {"max_recompute_error": maximum_error}
        metrics["statistics"] = statistics
        video_features = [key for key, feature in features.items()
                          if key.startswith("observation.images.") and feature.get("dtype") == "video"]
        expected_video_features = {f"observation.images.{camera}" for camera in CAMERAS}
        if set(video_features) != expected_video_features:
            errors.append("LeRobot 输出必须包含 agentview、left_eye_in_hand、right_eye_in_hand 三路视频")
        video_results = {}
        for key in video_features:
            feature = features[key]
            shape = feature.get("shape", [])
            video_dir = root / "videos" / key
            files = sorted(video_dir.glob("chunk-*/file-*.mp4"))
            if not files:
                errors.append(f"{key} 没有视频文件")
                continue
            counts = []
            for path in files:
                probe = _ffprobe_video(path)
                if not probe.get("available"):
                    warnings.append("ffprobe 不可用，跳过视频帧数验收")
                    break
                counts.append(probe.get("frames"))
                if probe.get("frames") is None:
                    warnings.append(f"{path.name} 没有 nb_frames，跳过该文件帧数验收")
                if shape and (probe.get("height"), probe.get("width")) != tuple(shape[:2]):
                    errors.append(f"{key}/{path.name} 尺寸与 info 不一致")
                if abs(probe.get("fps", fps) - fps) > 0.05:
                    errors.append(f"{key}/{path.name} FPS 与 info 不一致")
            if counts and all(value is not None for value in counts):
                video_results[key] = {"files": len(files), "frames": int(sum(counts))}
                if sum(counts) != total_frames:
                    errors.append(f"{key} 视频帧数 {sum(counts)} != total_frames {total_frames}")
        metrics["videos"] = video_results
        if expected:
            for key in ("total_frames", "episodes"):
                expected_key = "total_frames" if key == "total_frames" else "total_episodes"
                if expected.get(key) is not None and int(expected[key]) != int(info[expected_key]):
                    errors.append(f"输出 {expected_key} 与处理计划不一致")
    except Exception as exc:
        errors.append(f"验收读取失败：{type(exc).__name__}: {exc}")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "metrics": metrics}


def run(settings: Settings, progress: Callable[[dict], None] | None = None,
        cancel_event: threading.Event | None = None,
        prepared_plan: dict | None = None) -> dict:
    settings = canonical_settings(settings)
    start_time = time.monotonic()
    def emit(stage, **fields):
        if progress:
            progress(dict(stage=stage, elapsed=round(time.monotonic() - start_time, 1), **fields))

    def checkpoint():
        if cancel_event is not None and cancel_event.is_set():
            raise ProcessingCancelled("用户取消了处理")

    emit("checking", message="检查全部输入文件与帧率")
    plan = prepared_plan if prepared_plan is not None else scan(settings)
    output = resolve_path(plan["output_root"])
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请换一个新目录：{output}")
    # Never overwrite existing output, including an empty directory.
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(settings=asdict(settings), plan=plan, status="running",
                    processing="streaming: task phase -> resample -> optional delta -> LeRobot",
                    lerobot="lerobot")
    def save_report():
        temporary = output / f".{REPORT_NAME}.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / REPORT_NAME)
    save_report()
    dataset = None
    try:
        count = plan["episodes"]
        emit("loading", message="加载 LeRobot，准备写入数据集")
        from lerobot.datasets import LeRobotDataset
        cpu_count = available_cpu_count()
        emit("workers", cpu_count=cpu_count,
             message=f"使用 {cpu_count} 个可用 CPU worker 写入图像；视频编码由 LeRobot 编码器自动调度")
        dataset = LeRobotDataset.create(
            repo_id=plan["repo_id"], root=output / "lerobot", fps=settings.target_fps,
            features=make_features(plan["files"][0]["camera_shapes"]),
            robot_type="piper_dual", use_videos=True,
            image_writer_threads=cpu_count,
        )
        for index, info in enumerate(plan["files"]):
            checkpoint()
            source_dir = resolve_path(info.get("source_dir") or plan["input_dir"])
            with h5py.File(source_dir / info["name"], "r") as f:
                demo = f["data/demo_0"]
                # Stream one HDF5 directly into the current LeRobot episode.
                # A bounded contiguous read block is held while frames are sent
                # to LeRobot's asynchronous image writer.
                selected, _, _ = _task_sample_indices(demo, info["stride"])
                raw_actions = np.asarray(demo["actions"][selected], dtype=np.float32)
                if settings.action_mode == "delta_joint":
                    actions = np.diff(raw_actions, axis=0)
                    selected = selected[:-1]
                else:
                    actions = raw_actions
                n = len(selected)
                emit("resampling", episode=index, episodes=count, file=info["name"],
                     frame=0, frames=n, message="降采样并直接写入 LeRobot（不保存中间 HDF5）")
                # Read contiguous index blocks once.  The previous per-frame
                # HDF5 indexing caused three independent hyperslab reads for
                # every sample and left most CPU cores idle.
                block_start = 0
                while block_start < n:
                    block_end = min(block_start + READ_BLOCK_SIZE, n)
                    # HDF5's fancy indexing repeatedly decompresses the same
                    # chunks for strided indices.  Read one contiguous span,
                    # while bounding the raw span for high source/target FPS
                    # ratios so memory remains predictable.
                    while (
                        block_end > block_start + 1
                        and int(selected[block_end - 1] - selected[block_start]) + 1 > MAX_READ_SPAN
                    ):
                        block_end -= 1
                    block_indices = selected[block_start:block_end]
                    source_start, source_stop = int(block_indices[0]), int(block_indices[-1]) + 1
                    local_indices = block_indices - source_start
                    states = np.asarray(demo["states"][source_start:source_stop], dtype=np.float32)
                    images = {
                        camera: np.asarray(demo[f"obs/{camera}"][source_start:source_stop])
                        for camera in CAMERAS
                    }
                    for block_offset, action in enumerate(actions[block_start:block_end]):
                        i = block_start + block_offset
                        checkpoint()
                        dataset.add_frame({
                            "observation.state": states[local_indices[block_offset]],
                            "action": action,
                            **{
                                f"observation.images.{camera}": images[camera][local_indices[block_offset]]
                                for camera in CAMERAS
                            },
                            "task": settings.task,
                        })
                        if i % 20 == 0 or i == n - 1:
                            emit("writing", episode=index, episodes=count, frame=i + 1, frames=n,
                                 file=info["name"], message="写入图像与状态")
                    block_start = block_end
            emit("encoding", episode=index, episodes=count, file=info["name"],
                 message="编码三路视频并统计本 episode，完成后进入下一条")
            dataset.save_episode()
        emit("finalizing", message="保存 LeRobot 元数据与官方统计量")
        dataset.finalize()
        stats = dataset.meta.stats
        for key in ("action", "observation.state"):
            for stat in ("q01", "q99", "mean", "std", "min", "max"):
                values = np.asarray(stats[key][stat])
                if values.shape != (14,) or not np.isfinite(values).all():
                    raise ValueError(f"无效统计量：{key}/{stat}")
        validation = validate_lerobot_bundle(
            output / "lerobot",
            expected={"action_mode": settings.action_mode,
                      "total_frames": plan["output_frames"], "episodes": plan["episodes"]},
        )
        manifest["validation"] = validation
        if not validation["valid"]:
            raise ValueError("输出验收失败：" + "；".join(validation["errors"]))
        manifest["status"] = "complete"
        manifest["elapsed_seconds"] = round(time.monotonic() - start_time, 1)
        save_report()
        emit("complete", episode=count, episodes=count, message="处理完成", result=manifest)
        return manifest
    except ProcessingCancelled as exc:
        manifest.update(status="cancelled", error=str(exc), elapsed_seconds=round(time.monotonic() - start_time, 1))
        with contextlib.suppress(Exception):
            save_report()
        emit("cancelled", message=str(exc), result=manifest)
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), elapsed_seconds=round(time.monotonic() - start_time, 1))
        with contextlib.suppress(Exception):
            save_report()
        raise
    finally:
        if dataset is not None:
            # Close handles even after failure, but do not label partial output complete.
            with contextlib.suppress(Exception):
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                dataset.finalize()
