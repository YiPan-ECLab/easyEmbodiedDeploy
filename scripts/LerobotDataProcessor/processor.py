"""Piper data processing: resample HDF5, optionally difference actions, then LeRobot."""
from __future__ import annotations

import json
import time
import threading
from dataclasses import asdict, dataclass
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


class ProcessingCancelled(RuntimeError):
    """Raised when the user cancels between safe processing checkpoints."""


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
    target_fps: int = 20
    source_fps: int | None = None
    action_mode: str = "absolute"
    repo_id: str | None = None
    limit: int = 0


def _task_sample_indices(demo: h5py.Group, stride: int) -> tuple[np.ndarray, int]:
    """Select resampled task frames and exclude the recorder's homing tail."""
    indices = np.arange(0, len(demo["actions"]), stride)
    if "trajectory_phase" not in demo:
        return indices, 0

    phase = np.asarray(demo["trajectory_phase"])
    if phase.shape != (len(demo["actions"]),):
        raise ValueError("trajectory_phase 必须为 (T,)")
    sampled_phase = phase[indices]
    invalid = np.setdiff1d(np.unique(sampled_phase), (0, 1))
    if len(invalid):
        raise ValueError(f"trajectory_phase 含未知值：{invalid.tolist()}")
    task_indices = indices[sampled_phase == TASK_PHASE]
    if not len(task_indices):
        raise ValueError("trajectory_phase 中没有 task(0) 帧")
    if len(task_indices) > 1 and np.any(np.diff(task_indices) != stride):
        raise ValueError("task phase 必须是连续区间，不能跨非任务片段计算 action")
    return task_indices, len(indices) - len(task_indices)


def inspect_file(path: Path, settings: Settings) -> dict:
    with h5py.File(path, "r") as f:
        demo = f["data/demo_0"]
        actions, states = demo["actions"], demo["states"]
        n = len(actions)
        if n < 2 or actions.shape != (n, 14) or states.shape != (n, 14):
            raise ValueError(f"{path.name}: 需要至少两帧，且 actions/states 均为 (T,14)")
        if int(demo.attrs.get("num_samples", n)) != n:
            raise ValueError(f"{path.name}: num_samples 与实际帧数不符")
        if bool(f.attrs.get("normalized", False)):
            raise ValueError(f"{path.name}: 需要未归一化的原始控制量")
        representation = str(demo["actions"].attrs.get("representation", ""))
        if "delta" in representation or "delta" in str(f.attrs.get("actions_representation", "")):
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
        for key in ("actions", "states"):
            if not np.isfinite(demo[key][:]).all():
                raise ValueError(f"{path.name}: {key} 含 NaN/Inf")
        shapes = []
        for camera in CAMERAS:
            image = demo[f"obs/{camera}"]
            if image.ndim != 4 or image.shape[0] != n or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{path.name}: {camera} 需要 (T,H,W,3) uint8")
            shapes.append(list(image.shape[1:]))
        stride = source_fps // settings.target_fps
        selected, excluded_home_frames = _task_sample_indices(demo, stride)
        sampled_frames = len(selected)
        output_frames = sampled_frames - (settings.action_mode == "delta_joint")
        if output_frames < 1:
            raise ValueError(f"{path.name}: 降采样后不足以计算增量")
        return dict(name=path.name, frames=n, source_fps=source_fps,
                    measured_fps=round(measured, 4), stride=stride,
                    sampled_frames=sampled_frames, output_frames=output_frames,
                    excluded_home_frames=excluded_home_frames,
                    camera_shapes=shapes, gaps=int(np.count_nonzero(dt > 1.5 / source_fps)))


def scan(settings: Settings) -> dict:
    if settings.target_fps <= 0 or settings.limit < 0:
        raise ValueError("目标帧率必须为正整数，文件上限不能为负数")
    if settings.action_mode not in ("absolute", "delta_joint"):
        raise ValueError("动作模式必须为 absolute 或 delta_joint")
    if not settings.task.strip():
        raise ValueError("请填写任务描述")
    source = resolve_path(settings.input_dir)
    if not source.is_dir():
        raise ValueError(f"输入目录不存在：{source}")
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in (".h5", ".hdf5"))
    if settings.limit:
        paths = paths[:settings.limit]
    if not paths:
        raise ValueError("输入目录中没有 .hdf5/.h5 文件（只扫描当前层）")
    files = [inspect_file(p, settings) for p in paths]
    if any(x["camera_shapes"] != files[0]["camera_shapes"] for x in files):
        raise ValueError("不同 episode 的相机尺寸不一致")
    default_out = Path("datasets/processed") / f"{source.name}_{settings.target_fps}hz_{settings.action_mode}"
    out = resolve_path(settings.output_root or default_out)
    if out == source or out in source.parents or source in out.parents:
        raise ValueError("输出目录必须与输入目录分开，不能互相包含")
    return dict(input_dir=portable_path(source), output_root=portable_path(out), files=files,
                episodes=len(files), input_frames=sum(x["frames"] for x in files),
                output_frames=sum(x["output_frames"] for x in files),
                gaps=sum(x["gaps"] for x in files),
                repo_id=settings.repo_id or f"local/piper_{settings.target_fps}hz_{settings.action_mode}")


def _copy_selected(src: h5py.Group, dst: h5py.Group, n: int, indices: np.ndarray,
                   seen: dict, frame_scope: bool = False) -> None:
    """Preserve groups, attributes and hard links; slice every frame-aligned array."""
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for name in src:
        link = src.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ValueError(f"不支持外部或软链接：{src.name}/{name}")
        obj = src[name]
        addr = h5py.h5o.get_info(obj.id).addr
        if addr in seen:
            dst[name] = seen[addr]
            continue
        if isinstance(obj, h5py.Group):
            group = dst.create_group(name)
            seen[addr] = group
            _copy_selected(obj, group, n, indices, seen, frame_scope or obj.name == "/data/demo_0")
        else:
            aligned = frame_scope and obj.ndim > 0 and obj.shape[0] == n
            if aligned:
                shape = (len(indices), *obj.shape[1:])
                ds = dst.create_dataset(name, shape=shape, dtype=obj.dtype,
                                        compression="lzf", shuffle=True)
                # Bound RAM, including for high-resolution camera arrays.
                block = max(1, min(64, 16_000_000 // max(1, int(np.prod(obj.shape[1:])) * obj.dtype.itemsize)))
                for start in range(0, len(indices), block):
                    ds[start:start + block] = obj[indices[start:start + block]]
                for key, value in obj.attrs.items():
                    ds.attrs[key] = value
            else:
                src.copy(obj, dst, name=name)
                ds = dst[name]
            seen[addr] = ds


def transform_hdf5(source: Path, output: Path, *, stride: int, fps: int,
                   task: str, delta: bool = False) -> int:
    """Write task-phase frames; delta input must already be resampled."""
    if output.exists():
        raise FileExistsError(output)
    if stride < 1 or (delta and stride != 1):
        raise ValueError("必须先独立降采样，再以 stride=1 转增量")
    with h5py.File(source, "r") as src:
        demo = src["data/demo_0"]
        indices, _ = _task_sample_indices(demo, stride)
        if delta:
            if len(indices) < 2:
                raise ValueError("至少需要两帧才能差分")
            differences = np.diff(demo["actions"][indices], axis=0)
            indices = indices[:-1]
        with h5py.File(output, "x") as dst:
            _copy_selected(src, dst, len(demo["actions"]), indices, {})
            out = dst["data/demo_0"]
            out.attrs["num_samples"] = len(indices)
            out.attrs["language"] = task
            dst.attrs["fps"] = fps
            dst.attrs["normalized"] = False
            dst.attrs["actions_representation"] = "delta_joint" if delta else "absolute_joint"
            dst.attrs["processing_source_file"] = source.name
            dst.attrs["processing_stride"] = stride
            if "dones" in out:
                out["dones"][:] = False
                out["dones"][-1] = True
            if delta:
                # Deliberately distinguish these from directly replayable absolute commands.
                dst.attrs["format"] = "piper_processed_v1"
                dst.attrs["processing_tail"] = "drop_last_frame"
                for key, values in (("actions", differences),
                                    ("left_actions", differences[:, :7]),
                                    ("right_actions", differences[:, 7:])):
                    if key not in out:
                        continue
                    out[key][:] = values
                    out[key].attrs["representation"] = "delta_joint"
                    out[key].attrs["definition"] = "action[t+1]-action[t] after resampling; all 14 dimensions including grippers"
                    if "replay" in out[key].attrs:
                        del out[key].attrs["replay"]
            return len(indices)


def make_features(shapes: list) -> dict:
    features = {key: {"dtype": "float32", "shape": (14,), "names": {"axes": AXES}}
                for key in ("observation.state", "action")}
    for camera, shape in zip(CAMERAS, shapes):
        features[f"observation.images.{camera}"] = dict(dtype="video", shape=tuple(shape),
                                                       names=["height", "width", "channels"])
    return features


def run(settings: Settings, progress: Callable[[dict], None] | None = None,
        cancel_event: threading.Event | None = None) -> dict:
    start_time = time.monotonic()
    def emit(stage, **fields):
        if progress:
            progress(dict(stage=stage, elapsed=round(time.monotonic() - start_time, 1), **fields))

    def checkpoint():
        if cancel_event is not None and cancel_event.is_set():
            raise ProcessingCancelled("用户取消了处理")

    emit("checking", message="检查全部输入文件与帧率")
    plan = scan(settings)
    source, output = resolve_path(plan["input_dir"]), resolve_path(plan["output_root"])
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请换一个新目录：{output}")
    # Never overwrite existing output, including an empty directory.
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(settings=asdict(settings), plan=plan, status="running",
                    processing="streaming: task phase -> resample -> optional delta -> LeRobot",
                    lerobot="lerobot")
    def save_report():
        (output / REPORT_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_report()
    dataset = None
    try:
        count = plan["episodes"]
        emit("loading", message="加载 LeRobot，准备写入数据集")
        from lerobot.datasets import LeRobotDataset
        dataset = LeRobotDataset.create(
            repo_id=plan["repo_id"], root=output / "lerobot", fps=settings.target_fps,
            features=make_features(plan["files"][0]["camera_shapes"]),
            robot_type="piper_dual", use_videos=True, image_writer_threads=4,
        )
        for index, info in enumerate(plan["files"]):
            checkpoint()
            with h5py.File(source / info["name"], "r") as f:
                demo = f["data/demo_0"]
                # Stream one HDF5 directly into the current LeRobot episode.
                # Only the selected frame and its three images are held at a time.
                selected, _ = _task_sample_indices(demo, info["stride"])
                raw_actions = np.asarray(demo["actions"][selected], dtype=np.float32)
                if settings.action_mode == "delta_joint":
                    actions = np.diff(raw_actions, axis=0)
                    selected = selected[:-1]
                else:
                    actions = raw_actions
                n = len(selected)
                emit("resampling", episode=index, episodes=count, file=info["name"],
                     frame=0, frames=n, message="降采样并直接写入 LeRobot（不保存中间 HDF5）")
                for i in range(n):
                    checkpoint()
                    dataset.add_frame({
                        "observation.state": np.asarray(demo["states"][selected[i]], dtype=np.float32),
                        "action": actions[i],
                        **{f"observation.images.{c}": demo[f"obs/{c}"][selected[i]] for c in CAMERAS},
                        "task": settings.task,
                    })
                    if i % 20 == 0 or i == n - 1:
                        emit("writing", episode=index, episodes=count, frame=i + 1, frames=n,
                             file=info["name"], message="写入图像与状态")
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
        manifest["status"] = "complete"
        manifest["elapsed_seconds"] = round(time.monotonic() - start_time, 1)
        save_report()
        emit("complete", episode=count, episodes=count, message="处理完成", result=manifest)
        return manifest
    except ProcessingCancelled as exc:
        manifest.update(status="cancelled", error=str(exc), elapsed_seconds=round(time.monotonic() - start_time, 1))
        save_report()
        emit("cancelled", message=str(exc), result=manifest)
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), elapsed_seconds=round(time.monotonic() - start_time, 1))
        save_report()
        raise
    finally:
        if dataset is not None:
            # Close handles even after failure, but do not label partial output complete.
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
