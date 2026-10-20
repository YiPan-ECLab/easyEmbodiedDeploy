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
import sys
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from easy_embodied_deploy.statistics import statistics_close
from easy_embodied_deploy.action_contract import ACTION_MODES, POSE_MODES, action_axes, read_pose_contract
from easy_embodied_deploy.trajectory_files import revision, hold_trajectories
from easy_embodied_deploy.task_metadata import validate_tasks
from easy_embodied_deploy.pose_cleaning import clean_pose_indices, episode_sources, DEFAULT_SINGULARITY_RATIO
CAMERAS = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
AXES = [f"{side}_{axis}" for side in ("left", "right")
        for axis in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")]
REPORT_NAME = "processing.json"
TASK_PHASE = 0
HOME_PHASE = 1
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


class TimingError(ValueError):
    def __init__(self, info, errors):
        self.info = info
        self.errors = errors
        super().__init__("；".join(errors))


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
    task: str = ""
    target_fps: int = 10
    source_fps: int | None = None
    action_mode: str = "delta_joint"
    repo_id: str | None = None
    limit: int = 0
    gap_policy: str = "reject"
    input_dirs: list[str] | None = None
    calibration_bundle: str | None = None
    calibration_mount_id: str = ""
    timing_overrides: dict[str, str] | None = None
    task_source: str = "hdf5"
    pose_singularity_filter: bool = False
    pose_singularity_threshold: float = DEFAULT_SINGULARITY_RATIO


def canonical_action_mode(value: str) -> str:
    """Accept the old spelling while writing one unambiguous representation."""
    return "absolute_joint" if value == "absolute" else value


def canonical_settings(settings: Settings) -> Settings:
    if not isinstance(settings.pose_singularity_filter, bool):
        raise ValueError("位姿奇异清洗开关必须为布尔值")
    cleaning_enabled = settings.pose_singularity_filter and settings.action_mode in POSE_MODES
    threshold = settings.pose_singularity_threshold if cleaning_enabled else DEFAULT_SINGULARITY_RATIO
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("奇异比值阈值须在 0 与 1 之间")
    if settings.task_source not in ("hdf5", "override"):
        raise ValueError("任务指令来源须为 hdf5 或 override")
    if not isinstance(settings.task, str):
        raise ValueError("任务指令须为文本")
    if settings.timing_overrides is not None and (
        not isinstance(settings.timing_overrides, dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in settings.timing_overrides.items())
    ):
        raise ValueError("时序放行记录必须为文件路径与版本的映射")
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
        pose_singularity_filter=cleaning_enabled,
        pose_singularity_threshold=threshold,
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


def _task_end(demo: h5py.Group) -> int:
    """Validate the phase boundary before using task timestamps or targets."""
    if "trajectory_phase" not in demo:
        raise ValueError(
            "缺少 trajectory_phase；为避免把回零过程混入训练，"
            "只接受显式标注 0=task,1=home 的录制文件"
        )

    phase = np.asarray(demo["trajectory_phase"])
    if phase.shape != (len(demo["actions"]),):
        raise ValueError("trajectory_phase 必须为 (T,)")
    invalid = np.setdiff1d(np.unique(phase), (TASK_PHASE, HOME_PHASE))
    if len(invalid):
        raise ValueError(f"trajectory_phase 含未知值：{invalid.tolist()}")
    if phase[0] != TASK_PHASE:
        raise ValueError("trajectory_phase 必须从 task(0) 开始")
    home = np.flatnonzero(phase == HOME_PHASE)
    if len(home) and not np.all(phase[home[0] :] == HOME_PHASE):
        raise ValueError("trajectory_phase 必须是连续 task 前缀和可选 home 后缀")
    return int(home[0]) if len(home) else len(phase)


def _task_active_bounds(demo: h5py.Group, task_end: int, *, pose: bool = False) -> tuple[int, int]:
    """Keep the anchor before the first change and the first final target.

    Joint modes retain the existing full dual-arm command criterion. Pose
    modes also retain follower settling visible in state or flange feedback.
    """
    fields = ["actions"]
    if pose:
        fields += ["states", "obs/left_ee_states", "obs/right_ee_states"]
    changed = np.zeros(max(0, task_end - 1), dtype=bool)
    for key in fields:
        values = np.asarray(demo[key][:task_end])
        changed |= np.any(values[1:] != values[:-1], axis=1)
    changes = np.flatnonzero(changed)
    if not len(changes):
        return 0, 0
    return int(changes[0]), int(changes[-1] + 1)


def _task_sample_indices(demo: h5py.Group, stride: int, *, pose: bool = False) -> tuple[np.ndarray, int, int]:
    """Trim idle edges and home for every sequence; never resample fused input."""
    task_end = _task_end(demo)
    if "temporal_fusion" in demo.attrs:
        stride = 1
    task_indices = np.arange(0, task_end, stride)
    excluded_home_frames = len(range(0, len(demo["actions"]), stride)) - len(task_indices)
    first_active, last_active = _task_active_bounds(demo, task_end, pose=pose)
    trimmed_idle_frames = task_end - last_active - 1
    # Keep the first fixed-grid sample at or after the final target change. This
    # preserves the endpoint when that sample exists before the home boundary.
    endpoint = int(np.searchsorted(task_indices, last_active, side="left"))
    if endpoint < len(task_indices):
        task_indices = task_indices[: endpoint + 1]
    # Keep the grid sample at/before the anchor for ordinary resampling. For
    # fused input stride=1, this removes exactly the leading duplicate frames.
    task_indices = task_indices[task_indices >= (first_active // stride) * stride]
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
    source_revision = revision(path)
    timing_errors = []
    approved = (settings.timing_overrides or {}).get(str(path.resolve())) == source_revision
    with h5py.File(path, "r") as f:
        demo = f["data/demo_0"]
        recorded_task = _text_attr(demo.attrs.get("language", "")).strip()
        task = settings.task.strip() if settings.task_source == "override" else recorded_task
        task_source = settings.task_source
        if not task:
            raise ValueError(f"{path.name}: 缺少任务指令 language；可关闭继承并填写统一指令")
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
        task_end = _task_end(demo)
        if task_end < 2:
            raise ValueError(f"{path.name}: task 阶段至少需要两帧，不能使用 home 帧估计任务帧率")
        # Homing is excluded from training and may have a different cadence.
        # In short demonstrations it can dominate the full-episode median.
        dt = np.diff(t[:task_end])
        measured = float(1 / np.median(dt))
        fused = "temporal_fusion" in demo.attrs
        source_fps = settings.source_fps or int(round(measured))
        if not fused and source_fps <= 0:
            raise ValueError("无法估计有效源频率，请填写源频率后重新检查")
        if not fused and abs(measured - source_fps) / source_fps > .02:
            timing_errors.append(f"时间戳估计 {measured:.3f} Hz，与源帧率 {source_fps} 不符")
        if not fused and (settings.target_fps > source_fps or source_fps % settings.target_fps):
            raise ValueError(f"{path.name}: 目标 {settings.target_fps} Hz 必须整除源帧率 {source_fps} Hz，不支持升采样")
        arrays = {}
        for key in ("actions", "states"):
            arrays[key] = np.asarray(demo[key], dtype=np.float64)
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{path.name}: {key} 含 NaN/Inf")
            _validate_piper_range(arrays[key], f"{path.name}/{key}")
        problems = []
        def check_numeric(name, dataset):
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim >= 4:
                return
            values = np.asarray(dataset)
            if values.ndim and len(values) != n:
                problems.append(f"{name} 帧数与 actions 不一致")
            if values.dtype.kind in "fiu" and not np.isfinite(values).all():
                problems.append(f"{name} 含 NaN/Inf")
        demo.visititems(check_numeric)
        for combined, parts in (("actions", ("left_actions", "right_actions")),
                                ("states", ("obs/left_joint_states", "obs/left_gripper_states",
                                            "obs/right_joint_states", "obs/right_gripper_states"))):
            if all(key in demo for key in parts):
                if not np.array_equal(arrays[combined], np.concatenate([demo[key][:] for key in parts], axis=1)):
                    problems.append(f"{combined} 与左右臂拆分字段不一致")
        if problems:
            raise ValueError("；".join(problems))
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
        stride = 1 if fused else source_fps // settings.target_fps
        selected, excluded_home_frames, trimmed_idle_frames = _task_sample_indices(demo, stride, pose=settings.action_mode in POSE_MODES)
        sampled_frames = len(selected)
        output_frames = sampled_frames - int(settings.action_mode == "delta_joint" or settings.action_mode in POSE_MODES)
        if settings.action_mode in POSE_MODES:
            from easy_embodied_deploy.pose_conversion import read_raw_poses
            read_raw_poses(demo)
        if output_frames < 1:
            raise ValueError(f"{path.name}: 裁剪及降采样后不足两帧，无法构造未来动作；请延长任务或提高目标帧率")
        selected_dt = np.full(len(selected)-1, 1/settings.target_fps) if fused else np.diff(t[selected])
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
            timing_errors.append(
                f"task 段有 {gap_count} 个采样间隔异常，异常范围 "
                f"{selected_dt[gap_positions].min()*1000:.2f}–{max_gap*1000:.2f} ms"
            )
        warnings = (["人工融合序列：裁剪首尾静止帧并排除 HOME，不再降采样；目标 FPS 仅用于逻辑时间轴"] if fused else [])
        last_active = task_end - trimmed_idle_frames - 1
        if selected[-1] < last_active:
            warnings.append(
                f"末次任务变化在最后一个固定网格采样之后，末尾 {last_active - int(selected[-1])} "
                "个源帧未覆盖；可提高目标 FPS，或采集时在最终动作后稍等再结束"
            )
        if gap_count:
            warnings.append(
                f"task 段 {gap_count} 个采样间隔异常将被映射到固定 {settings.target_fps} Hz；"
                f"最大间隔 {max_gap:.3f}s，跨异常间隔动作 {max_gap_action:.4f} rad L2"
            )
        cleaning = None
        if settings.pose_singularity_filter:
            _, cleaning = clean_pose_indices(arrays["states"], selected, settings.pose_singularity_threshold)
            output_frames = cleaning["output_frames"]
            warnings.append(f"位姿奇异清洗：过滤 {cleaning['removed_singular_frames']} / {sampled_frames} 个采样帧，"
                            f"孤立帧 {cleaning['isolated_frames']}，生成 {cleaning['output_episodes']} 条 episode；不跨删除区间拼接")
        info = dict(name=path.name, path=str(path.resolve()), revision=source_revision,
                    task=task, task_source=task_source,
                    errors=[], timing_errors=timing_errors, timing_ignored=bool(approved and timing_errors),
                    frames=n, source_fps="mixed" if fused else source_fps, temporal_fusion=fused,
                    measured_fps=round(measured, 4), stride=stride,
                    sampled_frames=sampled_frames, output_frames=output_frames,
                    excluded_home_frames=excluded_home_frames,
                    trimmed_idle_frames=trimmed_idle_frames,
                    trimmed_leading_idle_frames=int(selected[0]),
                    source_task_frames=int(np.count_nonzero(np.asarray(demo["trajectory_phase"]) == TASK_PHASE)),
                    source_home_frames=int(np.count_nonzero(np.asarray(demo["trajectory_phase"]) == HOME_PHASE)),
                    camera_shapes=shapes, max_camera_skew_s=max_camera_skew,
                    gaps=gap_count, max_gap_s=round(max_gap, 6),
                    max_gap_action_joint_l2=round(max_gap_action, 6), warnings=warnings)
        if cleaning is not None:
            info["pose_cleaning"] = cleaning
        if revision(path) != source_revision:
            raise ValueError("检查期间文件已变化，请重新检查")
        if timing_errors and not approved:
            raise TimingError(info, timing_errors)
        if timing_errors:
            warnings.append("已人工允许物理时序异常；保持选帧与动作语义，未插值修补原始时间戳")
        return info


def scan(settings: Settings, *, collect_errors: bool = False) -> dict:
    settings = canonical_settings(settings)
    if (isinstance(settings.target_fps, bool) or not isinstance(settings.target_fps, int)
            or isinstance(settings.limit, bool) or not isinstance(settings.limit, int)
            or settings.target_fps <= 0 or settings.limit < 0):
        raise ValueError("目标帧率必须为正整数，文件上限不能为负数")
    if settings.source_fps is not None and (
        isinstance(settings.source_fps, bool) or not isinstance(settings.source_fps, int) or settings.source_fps <= 0
    ):
        raise ValueError("源帧率必须为正整数，自动检测请留空")
    if settings.action_mode not in ACTION_MODES:
        raise ValueError(f"动作模式必须为 {' 或 '.join(ACTION_MODES)}")
    if settings.gap_policy not in GAP_POLICIES:
        raise ValueError(f"时间空档策略必须为 {' 或 '.join(GAP_POLICIES)}")
    if settings.task_source == "override" and not settings.task.strip():
        raise ValueError("统一覆盖时必须填写任务指令")
    if settings.action_mode in POSE_MODES:
        from easy_embodied_deploy.pose_actions import load_calibration
        if settings.gap_policy != "reject":
            raise ValueError("位姿动作必须使用 reject，不能跨异常时间间隔组织未来目标")
        load_calibration(resolve_path(settings.calibration_bundle) if settings.calibration_bundle else None,
                         settings.action_mode, settings.calibration_mount_id)
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
        try:
            info = inspect_file(path, settings)
        except TimingError as exc:
            if not collect_errors:
                raise
            info = {**exc.info, "errors": exc.errors, "can_ignore_timing": True}
        except Exception as exc:
            if not collect_errors:
                raise
            info = dict(name=path.name, path=str(path.resolve()), revision=revision(path) if path.exists() else None,
                        errors=[str(exc)], timing_errors=[], can_ignore_timing=False, warnings=[],
                        frames=0, output_frames=0, source_task_frames=0, source_home_frames=0, gaps=0)
        info["source_dir"] = portable_path(source)
        files.append(info)
    shaped = [x for x in files if "camera_shapes" in x]
    if shaped and any(x["camera_shapes"] != shaped[0]["camera_shapes"] for x in shaped):
        if not collect_errors:
            raise ValueError("不同 episode 的相机尺寸不一致")
        for item in shaped:
            if item["camera_shapes"] != shaped[0]["camera_shapes"]:
                item["errors"].append("相机尺寸与其他 episode 不一致")
                item["can_ignore_timing"] = False
    source_label = sources[0].name if len(sources) == 1 else f"{sources[0].name}_merged"
    frequency_label = "mixed" if any(item.get("temporal_fusion") or item.get("timing_ignored") for item in files) else f"{settings.target_fps}hz"
    suffix = "_cleaned" if settings.pose_singularity_filter else ""
    default_out = Path("datasets/lerobot") / f"{source_label}_{frequency_label}_{settings.action_mode}{suffix}"
    out = resolve_path(settings.output_root or default_out)
    if any(out == source or out in source.parents or source in out.parents for source in sources):
        raise ValueError("输出目录必须与输入目录分开，不能互相包含")
    warnings = [f"{item['name']}: {warning}" for item in files for warning in item["warnings"]]
    return dict(input_dir=portable_path(sources[0]), input_dirs=[portable_path(path) for path in sources],
                target_fps=settings.target_fps, sampling_fps="mixed" if frequency_label == "mixed" else settings.target_fps, action_mode=settings.action_mode,
                output_root=portable_path(out), files=files,
                tasks=list(dict.fromkeys(x["task"] for x in files if x.get("task"))),
                errors=[f"{x['name']}: {error}" for x in files for error in x.get("errors", [])],
                invalid_files=sum(bool(x.get("errors")) for x in files),
                episodes=len(episode_sources(files)), input_frames=sum(x["frames"] for x in files),
                pose_cleaning=({"enabled": True, "threshold": settings.pose_singularity_threshold,
                    **{key: sum(x.get("pose_cleaning", {}).get(key, 0) for x in files)
                       for key in ("candidate_frames", "removed_singular_frames", "isolated_frames", "retained_source_frames")}}
                    if settings.pose_singularity_filter else None),
                output_frames=sum(x["output_frames"] for x in files),
                gaps=sum(x["gaps"] for x in files),
                warnings=warnings,
                repo_id=settings.repo_id or f"local/piper_{frequency_label}_{settings.action_mode}")


def make_features(shapes: list, action_mode="absolute_joint") -> dict:
    features = {key: {"dtype": "float32", "shape": (14,),
                      "names": {"axes": AXES if key == "observation.state" else action_axes(action_mode)}}
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
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
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
                files = report.get("plan", {}).get("files", [])
                if files:
                    expected["episode_tasks"] = [f.get("task", expected.get("task") or None) for f in files]
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
        fps = int(info["video_fps"] if info["fps"] == "mixed" else info["fps"])
        total_episodes = int(info["total_episodes"])
        total_frames = int(info["total_frames"])
        if fps <= 0 or total_episodes <= 0 or total_frames <= 0:
            raise ValueError("info 中 fps/episode/frame 总数必须为正数")
        if info.get("robot_type") != "piper_dual":
            errors.append("robot_type 必须为 piper_dual")
        features = info.get("features", {})
        pose_contract = read_pose_contract(root)
        action_mode = canonical_action_mode(str((expected or {}).get("action_mode", "")))
        if pose_contract:
            if action_mode and action_mode != pose_contract["action_mode"]:
                errors.append("位姿契约与所选 action_mode 不一致")
            action_mode = pose_contract["action_mode"]
        elif action_mode in POSE_MODES:
            errors.append("位姿数据缺少动作契约")
        for key in ("observation.state", "action"):
            feature = features.get(key, {})
            if feature.get("dtype") != "float32" or feature.get("shape") != [14]:
                errors.append(f"{key} feature 不是 float32[14]")
            axes = feature.get("names", {}).get("axes")
            expected_axes = action_axes(action_mode) if key == "action" and action_mode else AXES
            if axes != expected_axes:
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
        task_errors, task_mapping = validate_tasks(root, info, task_index, episode,
                                                   (expected or {}).get("episode_tasks"))
        errors.extend(task_errors)
        metrics["tasks"] = task_mapping
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
        action_values = np.asarray(columns["action"], dtype=np.float64)
        if pose_contract:
            from easy_embodied_deploy.pose_conversion import validate_pose_bundle
            checked_pose = validate_pose_bundle(root, columns)
            errors.extend(checked_pose["errors"])
            metrics["pose"] = checked_pose["metrics"]
        if action_mode == "absolute_joint" and action_values.shape == (rows, 14):
            try:
                _validate_piper_range(action_values, "action")
            except ValueError as exc:
                errors.append(str(exc))
        if action_values.shape == (rows, 14) and rows and np.isfinite(action_values).all():
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
                if not statistics_close(stored, expected_values, name, values):
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
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled("用户取消了处理")
    plan = prepared_plan if prepared_plan is not None else scan(settings)
    if plan.get("errors"):
        raise ValueError("仍有未处理的输入错误：" + "；".join(plan["errors"]))
    paths = [(resolve_path(info.get("source_dir") or plan["input_dir"]) / info["name"],
              info.get("revision")) for info in plan["files"]]
    with hold_trajectories(paths):
        return _run(settings, progress, cancel_event, plan)


def _run(settings: Settings, progress: Callable[[dict], None] | None = None,
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
    checkpoint()
    plan = prepared_plan if prepared_plan is not None else scan(settings)
    output = resolve_path(plan["output_root"])
    if plan["episodes"] == 0 or plan["output_frames"] == 0:
        raise ValueError("清洗后没有至少两帧的可用片段，请降低奇异阈值或关闭清洗")
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
            features=make_features(plan["files"][0]["camera_shapes"], settings.action_mode),
            robot_type="piper_dual", use_videos=True,
            image_writer_threads=cpu_count,
        )
        pose_contract = None
        if settings.action_mode in POSE_MODES:
            from easy_embodied_deploy.pose_actions import load_calibration
            from easy_embodied_deploy.pose_dataset import new_contract
            calibration = load_calibration(resolve_path(settings.calibration_bundle) if settings.calibration_bundle else None,
                                           settings.action_mode, settings.calibration_mount_id)
            sampling_fps = plan["sampling_fps"]
            pose_contract = new_contract(settings.action_mode, sampling_fps, calibration, settings.calibration_mount_id)
        for index, info in enumerate(episode_sources(plan["files"])):
            checkpoint()
            source_dir = resolve_path(info.get("source_dir") or plan["input_dir"])
            with h5py.File(source_dir / info["name"], "r") as f:
                demo = f["data/demo_0"]
                # Stream one HDF5 directly into the current LeRobot episode.
                # A bounded contiguous read block is held while frames are sent
                # to LeRobot's asynchronous image writer.
                selected, _, _ = _task_sample_indices(demo, info["stride"], pose=settings.action_mode in POSE_MODES)
                if "clean_segment" in info:
                    first, stop = info["clean_segment"]
                    selected = selected[first:stop]
                raw_actions = np.asarray(demo["actions"][selected], dtype=np.float32)
                if pose_contract is not None:
                    from easy_embodied_deploy.pose_conversion import convert_episode
                    actions = convert_episode(output / "lerobot", pose_contract, demo, selected,
                                              source=dict(info))
                    selected = selected[:-1]
                elif settings.action_mode == "delta_joint":
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
                            "task": info["task"],
                        })
                        if i % 20 == 0 or i == n - 1:
                            emit("writing", episode=index, episodes=count, frame=i + 1, frames=n,
                                 file=info["name"], message="写入图像与状态")
                    block_start = block_end
            emit("encoding", episode=index, episodes=count, file=info["name"],
                 message="编码三路视频并统计本 episode，完成后进入下一条")
            dataset.save_episode()
            checkpoint()
        emit("finalizing", message="保存 LeRobot 元数据与官方统计量")
        dataset.finalize()
        if plan["sampling_fps"] == "mixed":
            from lerobot.datasets.io_utils import write_info
            dataset.meta.info.video_fps = settings.target_fps
            dataset.meta.info.fps = "mixed"
            write_info(dataset.meta.info, output / "lerobot")
        if pose_contract is not None:
            from easy_embodied_deploy.pose_dataset import compute_chunk_statistics
            emit("pose_statistics", message="按固定锚点的完整 50 步 chunk 统计有效位姿目标，排除尾部填充")
            manifest["pose_action_statistics"] = compute_chunk_statistics(output / "lerobot", checkpoint)
            manifest["pose_contract"] = pose_contract
        stats = dataset.meta.stats
        for key in ("action", "observation.state"):
            for stat in ("q01", "q99", "mean", "std", "min", "max"):
                values = np.asarray(stats[key][stat])
                if values.shape != (14,) or not np.isfinite(values).all():
                    raise ValueError(f"无效统计量：{key}/{stat}")
        validation = validate_lerobot_bundle(
            output / "lerobot",
            expected={"action_mode": settings.action_mode,
                      "episode_tasks": [f["task"] for f in episode_sources(plan["files"])],
                      "total_frames": plan["output_frames"], "episodes": plan["episodes"]},
        )
        manifest["validation"] = validation
        checkpoint()
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
