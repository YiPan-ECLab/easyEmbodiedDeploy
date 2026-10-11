"""Manual, index-based temporal decimation, independent of action representation.

The exported recording has mixed sampling frequency. Physical timestamps and
frame provenance remain intact; playback speed is separate metadata.
"""

import json
import os
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

MARKER = "temporal_fusion"
PROVENANCE = "temporal_provenance"
TIME_FIELDS = ("timestamps", "camera_timestamps", "action_source_timestamps", "scheduled_timestamps")


def metadata(demo):
    raw = demo.attrs.get(MARKER)
    return json.loads(raw) if raw else None


def task_length(demo):
    phase = np.asarray(demo["trajectory_phase"])
    if phase.shape != (len(demo["actions"]),) or not np.isin(phase, [0, 1]).all():
        raise ValueError("trajectory_phase 必须是 task/home 序列")
    home = np.flatnonzero(phase == 1)
    end = int(home[0]) if len(home) else len(phase)
    if not np.all(phase[end:] == 1) or end < 2:
        raise ValueError("需要至少两帧连续 task，home 只能出现在末尾")
    return end


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数")
    return value


def selection(demo, plan):
    """Inclusive zero-based ranges; keep boundaries, never average observations."""
    end = task_length(demo)
    base = _integer(plan.get("base_stride", 1), "基础步长")
    fps = _integer(plan.get("playback_fps", 30), "预览 FPS")
    segments = []
    for item in plan.get("segments", []):
        start = _integer(item["start"], "起始帧", 0)
        stop = _integer(item["end"], "结束帧", 0)
        factor = _integer(item["factor"], "片段倍率")
        if not start < stop < end:
            raise ValueError(f"片段必须满足 0 <= start < end < {end}（仅 task 帧）")
        segments.append(dict(start=start, end=stop, factor=factor))
    segments.sort(key=lambda item: item["start"])
    for left, right in zip(segments, segments[1:]):
        if right["start"] < left["end"]:
            raise ValueError("片段不能重叠；可以共用一个边界帧")
    keep = np.zeros(end, dtype=bool)
    keep[::base] = True
    for item in segments:
        start, stop = item["start"], item["end"]
        keep[start:stop+1] = False
        keep[start:stop+1:base*item["factor"]] = True
        keep[stop] = True
    keep[0] = keep[-1] = True
    indices = np.flatnonzero(keep)
    canonical = dict(version=1, base_stride=base, playback_fps=fps, segments=segments)
    return canonical, indices


def preview(path, demo_name, plan=None):
    with h5py.File(path, "r") as handle:
        demo = handle[demo_name]
        if plan is None:
            times = np.asarray(demo["timestamps"])[:task_length(demo)]
            fps = int(demo.attrs.get("playback_fps", max(1, int(round(1/np.median(np.diff(times)))))))
            plan = dict(playback_fps=fps)
        canonical, indices = selection(demo, plan)
        return dict(plan=canonical, indices=indices.tolist(), task_frames=task_length(demo),
                    source_frames=len(demo["actions"]), frames=len(indices),
                    output_directory=str(Path(path).parent.with_name(Path(path).parent.name + "_fusion")))


def _draft_path(path, demo_name):
    return Path(str(path) + "." + demo_name.strip("/").replace("/", "_") + ".fusion.json")


def _signature(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def save_plan(path, demo_name, plan):
    result = preview(path, demo_name, plan)
    destination = _draft_path(path, demo_name)
    temp = destination.with_name(destination.name + "." + uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(dict(source=_signature(path), plan=result["plan"]), ensure_ascii=False, indent=2))
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return dict(result, draft_path=str(destination))


def load_plan(path, demo_name):
    draft = _draft_path(path, demo_name)
    if not draft.exists():
        return preview(path, demo_name)
    data = json.loads(draft.read_text())
    if data["source"] != _signature(path):
        raise ValueError("源文件已改变，旧标注不能自动套用；请重新标注")
    return dict(preview(path, demo_name, data["plan"]), draft_path=str(draft))


def export_episode(path, demo_name, plan, output_directory=None):
    path = Path(path).resolve()
    signature = _signature(path)
    result = preview(path, demo_name, plan)
    directory = Path(output_directory or result["output_directory"]).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{path.stem}_fusion_{uuid4().hex[:12]}.hdf5"
    temporary = destination.with_suffix(".tmp")
    indices = np.asarray(result["indices"], dtype=np.int64)
    try:
        with h5py.File(path, "r") as source, h5py.File(temporary, "x") as target:
            if source.attrs.get("format") != "pyagxarm_direct_v1":
                raise ValueError("融合仅支持 pyagxarm_direct_v1 录制")
            demo = source[demo_name]
            count = len(demo["actions"])
            target.attrs.update(source.attrs)
            data = target.create_group("data")
            if "data" in source:
                data.attrs.update(source["data"].attrs)
            output = data.create_group("demo_0")
            seen = {}

            def copy_group(src, dst):
                dst.attrs.update(src.attrs)
                for name, obj in src.items():
                    if src == demo and name == PROVENANCE:
                        continue
                    address = h5py.h5o.get_info(obj.id).addr
                    if address in seen:
                        dst[name] = target[seen[address]]
                        continue
                    if isinstance(obj, h5py.Group):
                        child = dst.create_group(name)
                        seen[address] = child.name
                        copy_group(obj, child)
                    elif obj.ndim and obj.shape[0] == count:
                        child = dst.create_dataset(name, shape=(len(indices), *obj.shape[1:]),
                                                   dtype=obj.dtype, chunks=True, compression="lzf")
                        # Bound source spans as well as selected rows, especially for RGB.
                        offset = 0
                        while offset < len(indices):
                            stop = int(np.searchsorted(indices, indices[offset]+32))
                            block = indices[offset:stop]
                            child[offset:stop] = obj[int(block[0]):int(block[-1])+1][block-block[0]]
                            offset = stop
                        child.attrs.update(obj.attrs)
                        seen[address] = child.name
                    else:
                        src.copy(name, dst, name=name)
                        seen[address] = dst[name].name

            copy_group(demo, output)
            previous = metadata(demo)
            description = dict(version=1, source=str(path), source_demo=demo_name,
                               source_signature=signature, plan=result["plan"],
                               original_task_frames=result["task_frames"], frames=len(indices),
                               fps="mixed", timeline="physical", previous=previous)
            provenance = output.create_group(PROVENANCE)
            provenance.create_dataset("input_frame_indices", data=indices)
            prior = demo.get(PROVENANCE)
            ultimate = prior["source_frame_indices"][indices] if prior is not None else indices
            provenance.create_dataset("source_frame_indices", data=ultimate)
            for name in TIME_FIELDS:
                if name not in demo:
                    continue
                original = prior[name] if prior is not None and name in prior else demo[name]
                saved = provenance.create_dataset(name, data=original[indices])
                saved.attrs.update(original.attrs)
            output["trajectory_phase"][...] = 0
            output["dones"][...] = False
            output["dones"][-1] = True
            output.attrs["num_samples"] = len(indices)
            output.attrs[MARKER] = json.dumps(description, ensure_ascii=False)
            target.attrs[MARKER] = "v1"
            target.attrs["fps"] = output.attrs["fps"] = "mixed"
            if "record_hz" in output.attrs:
                output.attrs["source_record_hz"] = output.attrs.get("source_record_hz", output.attrs["record_hz"])
                output.attrs["record_hz"] = "mixed"
            output.attrs["playback_fps"] = result["plan"]["playback_fps"]
            if "total" in data.attrs:
                data.attrs["total"] = len(indices)
        if _signature(path) != signature:
            raise ValueError("源文件在导出期间发生变化，请等待采集结束后重试")
        # Atomic publication without replacing any existing user file.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(result, path=str(destination), output_directory=str(directory))
