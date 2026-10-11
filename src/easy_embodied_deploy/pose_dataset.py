"""Portable pose sidecars and frozen-anchor training chunks.

The ordinary LeRobot action column is the first future target. Full-chunk
statistics live separately, so official per-row statistics retain their meaning.
"""

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .action_contract import POSE_AXES, POSE_CONTRACT, STATE_AXES, read_pose_contract
from .pose_actions import PoseCodec, calibration_for_mode, fingerprint


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def new_contract(mode, fps, calibration=None, mount_id=""):
    PoseCodec(mode, calibration)
    return dict(schema="piper_pose_actions", version=1, action_mode=mode,
                state_axes=STATE_AXES, action_axes=POSE_AXES, state_units="rad x6,m per arm",
                rotation="rotvec_rad", position_units="m", pose_frame="own_base_flange",
                target_offset=1, chunk_anchor="observation_time", horizon=50, fps="mixed" if fps == "mixed" else int(fps),
                padding="repeat_last_target_mask_loss_and_statistics",
                calibration=calibration, calibration_sha256=fingerprint(calibration),
                mount_id=mount_id, episodes=[])


def write_episode(root, contract, poses, grippers, source_indices, timestamps, source):
    poses, grippers = np.asarray(poses, dtype=np.float64), np.asarray(grippers, dtype=np.float64)
    if poses.shape != (len(source_indices), 12) or grippers.shape != (len(poses), 2) or len(poses) < 2:
        raise ValueError("位姿 sidecar 的姿态 / 夹爪 / 源帧数量不一致")
    if not np.isfinite(poses).all() or not np.isfinite(grippers).all():
        raise ValueError("位姿 sidecar 包含非有限数值")
    if np.any(np.diff(source_indices) <= 0) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("位姿源帧与实际时间必须严格递增")
    index = len(contract["episodes"])
    folder = Path(root) / "meta/pose"
    folder.mkdir(parents=True, exist_ok=True)
    relative = f"pose/episode-{index:06d}.npz"
    np.savez_compressed(Path(root) / "meta" / relative, poses=poses, grippers=grippers,
                        source_indices=np.asarray(source_indices, dtype=np.int64),
                        timestamps=np.asarray(timestamps, dtype=np.float64))
    item = dict(episode_index=index, row_start=sum(e["rows"] for e in contract["episodes"]),
                rows=len(poses)-1, path=relative, source=source)
    contract["episodes"].append(item)
    write_json(Path(root) / "meta" / POSE_CONTRACT, contract)


class PoseDataset:
    def __init__(self, root, horizon=50):
        self.root = Path(root)
        self.contract = read_pose_contract(root, required=True)
        if horizon != self.contract["horizon"]:
            raise ValueError("训练 chunk 长度必须与位姿统计契约一致（50）")
        if fingerprint(self.contract.get("calibration")) != self.contract.get("calibration_sha256"):
            raise ValueError("位姿契约中的标定快照校验失败")
        self.codec = PoseCodec(self.contract["action_mode"], self.contract.get("calibration"))
        calibration_for_mode(self.contract.get("calibration"), self.contract["action_mode"], self.contract["mount_id"])
        self.episodes = self.contract["episodes"]
        start = 0
        for index, item in enumerate(self.episodes):
            if item["episode_index"] != index or item["row_start"] != start or item["rows"] < 1:
                raise ValueError("位姿 episode 索引不连续")
            if item["path"] != f"pose/episode-{index:06d}.npz":
                raise ValueError("位姿 sidecar 路径不符合契约")
            start += item["rows"]
        self.rows = start
        self.starts = np.asarray([e["row_start"] for e in self.episodes], dtype=np.int64)

    @lru_cache(maxsize=4)
    def episode(self, index):
        item = self.episodes[index]
        with np.load(self.root / "meta" / item["path"], allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ("poses", "grippers", "source_indices", "timestamps")}
        count = item["rows"]+1
        for key, width in (("poses", 12), ("grippers", 2)):
            if arrays[key].shape != (count, width) or not np.isfinite(arrays[key]).all():
                raise ValueError(f"{item['path']} 的 {key} 无效")
        for key in ("source_indices", "timestamps"):
            if arrays[key].shape != (count,) or not np.isfinite(arrays[key]).all() or np.any(np.diff(arrays[key]) <= 0):
                raise ValueError(f"{item['path']} 的 {key} 无效")
        if np.any((arrays["grippers"] < 0) | (arrays["grippers"] > .1)):
            raise ValueError("位姿夹爪宽度超出 [0,0.1] m")
        return arrays

    def chunk(self, index):
        if not 0 <= index < self.rows:
            raise IndexError(index)
        ep = int(np.searchsorted(self.starts, index, side="right")-1)
        arrays = self.episode(ep)
        return self.codec.chunk(arrays["poses"], arrays["grippers"], index-int(self.starts[ep]), 50)

    def anchor(self, index):
        if not 0 <= index < self.rows:
            raise IndexError(index)
        ep = int(np.searchsorted(self.starts, index, side="right")-1)
        return self.episode(ep)["poses"][index-int(self.starts[ep])].copy()


def compute_chunk_statistics(root, checkpoint=lambda: None, *, persist=True):
    dataset = PoseDataset(root)
    total = sum(sum(min(50, left) for left in range(1, ep["rows"]+1)) for ep in dataset.episodes)
    descriptor, name = tempfile.mkstemp(prefix="piper-pose-statistics-", suffix=".npy")
    os.close(descriptor)
    temporary = Path(name)
    values = None
    try:
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(total, 14))
        count, mean, m2 = 0, np.zeros(14), np.zeros(14)
        for i in range(dataset.rows):
            if i % 32 == 0:
                checkpoint()
            action, valid = dataset.chunk(i)
            block = action[valid].astype(np.float64)
            n = len(block)
            values[count:count+n] = block
            block_mean = block.mean(axis=0)
            difference = block_mean - mean
            m2 += ((block-block_mean)**2).sum(axis=0) + difference**2 * count*n/(count+n)
            mean += difference*n/(count+n)
            count += n
        values.flush()
        stats = {"mean": mean.tolist(), "std": np.sqrt(m2/count).tolist(), "count": [count]}
        quantiles = np.stack([np.quantile(values[:, axis], [0, .01, .1, .5, .9, .99, 1]) for axis in range(14)], axis=1)
        stats.update({key: value.tolist() for key, value in zip(("min", "q01", "q10", "q50", "q90", "q99", "max"), quantiles)})
        if persist:
            write_json(Path(root) / "meta/pose_action_stats.json", stats)
        return stats
    finally:
        if values is not None:
            del values
        temporary.unlink(missing_ok=True)


def training_stats(root, ordinary_stats):
    contract = read_pose_contract(root)
    if contract is None:
        return ordinary_stats
    stats = json.loads((Path(root) / "meta/pose_action_stats.json").read_text())
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        value = np.asarray(stats.get(key), dtype=np.float64)
        if value.shape != (14,) or not np.isfinite(value).all():
            raise ValueError(f"位姿 chunk 统计 {key} 无效")
    ordered = np.asarray([stats[k] for k in ("min", "q01", "q99", "max")])
    expected_count = sum(sum(min(50, i) for i in range(1, ep["rows"]+1)) for ep in contract["episodes"])
    if stats.get("count") != [expected_count] or expected_count < 1 or np.any(np.diff(ordered, axis=0) < 0) or np.any(np.asarray(stats["std"]) < 0):
        raise ValueError("位姿 chunk 统计为空或分位数无效")
    return {**ordinary_stats, "action": stats}


def copy_retained_episodes(source, destination, deleted):
    """Keep pose metadata aligned with the official LeRobot episode editor."""
    if read_pose_contract(source) is None:
        return
    data = PoseDataset(source)
    old = data.contract
    contract = new_contract(old["action_mode"], old["fps"], old["calibration"], old["mount_id"])
    for index, episode in enumerate(data.episodes):
        if index not in deleted:
            a = data.episode(index)
            write_episode(destination, contract, a["poses"], a["grippers"], a["source_indices"],
                          a["timestamps"], episode["source"])
    compute_chunk_statistics(destination)
