#!/usr/bin/env python3
"""Read current LeRobot v3 datasets inside OpenPI's pinned v2.1 environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from easy_embodied_deploy.action_contract import read_pose_contract

CAMERAS = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
IMAGE_KEYS = tuple(f"observation.images.{name}" for name in CAMERAS)
VIDEO_BACKEND = os.environ.get("OPENPI_VIDEO_BACKEND", "torchcodec").strip().lower()
if VIDEO_BACKEND not in {"torchcodec", "pyav"}:
    raise ValueError("OPENPI_VIDEO_BACKEND must be torchcodec or pyav")


def _read_parquet_files(paths: list[Path], columns: list[str] | None = None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not paths:
        raise FileNotFoundError("没有找到 parquet 文件")
    return pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])


def _fixed_vectors(table, key: str, width: int) -> np.ndarray:
    column = table.column(key).combine_chunks()
    import pyarrow as pa
    import pyarrow.compute as pc
    if column.null_count or not (pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type)):
        raise ValueError(f"{key} 必须是无空值的向量列")
    if not np.all(np.asarray(pc.list_value_length(column)) == width):
        raise ValueError(f"{key} 每一行必须恰好 {width} 维")
    values = np.asarray(column.values, dtype=np.float32)
    result = values.reshape(len(column), width)
    if not np.isfinite(result).all():
        raise ValueError(f"{key} 包含 NaN 或无穷值")
    return result


class LeRobotV3Dataset:
    """Small read-only adapter for the v3 consolidated parquet/video layout."""

    def __init__(self, root: str | Path, action_horizon: int):
        self.root = Path(root).expanduser().resolve()
        self.action_horizon = int(action_horizon)
        if self.action_horizon < 1:
            raise ValueError("action_horizon 必须为正数")
        info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"只支持 LeRobot v3.0，检测到 {info.get('codebase_version')!r}")
        self.fps = float(info["video_fps"] if info["fps"] == "mixed" else info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("数据集 fps 必须为有限正数")
        self._video_path_template = str(
            info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        )

        data = _read_parquet_files(
            sorted((self.root / "data").glob("chunk-*/file-*.parquet")),
            ["observation.state", "action", "timestamp", "episode_index", "task_index"],
        )
        self.pose_data = None
        if read_pose_contract(self.root):
            from easy_embodied_deploy.pose_dataset import PoseDataset
            self.pose_data = PoseDataset(self.root, action_horizon)
        self._states = _fixed_vectors(data, "observation.state", 14)
        self._actions = _fixed_vectors(data, "action", 14)
        self._timestamps = np.asarray(data.column("timestamp"), dtype=np.float64)
        self._episode_indices = np.asarray(data.column("episode_index"), dtype=np.int64)
        self._task_indices = np.asarray(data.column("task_index"), dtype=np.int64)

        tasks = _read_parquet_files(sorted((self.root / "meta").glob("tasks*.parquet")))
        self._tasks = {
            int(index): str(task)
            for index, task in zip(tasks.column("task_index").to_pylist(), tasks.column("task").to_pylist(), strict=True)
        }
        episodes = _read_parquet_files(sorted((self.root / "meta/episodes").glob("chunk-*/file-*.parquet")))
        self._episodes: dict[int, dict[str, Any]] = {
            int(row["episode_index"]): row for row in episodes.to_pylist()
        }
        if len(self._states) != int(info["total_frames"]):
            raise ValueError(f"帧数不一致：info={info['total_frames']} parquet={len(self._states)}")

        if len(self._episodes) != int(info["total_episodes"]) or len(self._episodes) != len(episodes):
            raise ValueError("episode 元数据数量不一致或存在重复索引")
        self._episode_end = np.zeros(len(self), dtype=np.int64)
        for episode_index, row in self._episodes.items():
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            if start < 0 or end <= start or end > len(self):
                raise ValueError(f"episode {episode_index} 的帧范围无效：[{start}, {end})")
            if np.any(self._episode_end[start:end]) or not np.all(self._episode_indices[start:end] == episode_index):
                raise ValueError(f"episode {episode_index} 范围重叠或与数据不一致")
            self._episode_end[start:end] = end
        if not len(self) or np.any(self._episode_end == 0):
            raise ValueError("episode 元数据未完整覆盖数据帧")
        if not np.isfinite(self._timestamps).all() or any(int(index) not in self._tasks for index in np.unique(self._task_indices)):
            raise ValueError("数据时间戳或任务索引无效")

    def __len__(self) -> int:
        return len(self._states)

    def _video_location(self, key: str, index: int) -> tuple[Path, float]:
        episode_index = int(self._episode_indices[index])
        episode = self._episodes[episode_index]
        prefix = f"videos/{key}"
        chunk_index = int(episode[f"{prefix}/chunk_index"])
        file_index = int(episode[f"{prefix}/file_index"])
        video_path = self.root / self._video_path_template.format(
            video_key=key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        timestamp = float(episode[f"{prefix}/from_timestamp"]) + float(self._timestamps[index])
        return video_path, timestamp

    def _decode_image(self, key: str, index: int):
        try:
            from lerobot.common.datasets.video_utils import decode_video_frames
        except ModuleNotFoundError:
            from lerobot.datasets.video_utils import decode_video_frames

        path, timestamp = self._video_location(key, index)
        frames = decode_video_frames(
            path,
            [timestamp],
            tolerance_s=max(1e-4, 0.51 / self.fps),
            # TorchCodec avoids the long-lived PyAV/libdav1d decoder growth
            # seen with AV1 videos when several DataLoader workers prefetch.
            # Set OPENPI_VIDEO_BACKEND=pyav only for an older environment that
            # does not provide TorchCodec.
            backend=VIDEO_BACKEND,
        )
        frame = frames[0]
        if tuple(frame.shape) != (3, 224, 224):
            raise ValueError(f"{key} 解码形状异常：{tuple(frame.shape)}")
        return frame

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        end = int(self._episode_end[index])
        action_indices = np.minimum(np.arange(index, index + self.action_horizon), end - 1)
        sample: dict[str, Any] = {
            "observation.state": self._states[index].copy(),
            "action": self._actions[action_indices].copy(),
            "task": self._tasks[int(self._task_indices[index])],
            "task_index": self._task_indices[index],
        }
        sample["action_valid"] = np.arange(index, index+self.action_horizon) < end
        if self.pose_data is not None:
            sample["action"], sample["action_valid"] = self.pose_data.chunk(index)
        sample.update({key: self._decode_image(key, index) for key in IMAGE_KEYS})
        return sample


def install_openpi_loader(dataset_root: str | Path) -> None:
    """Route OpenPI's torch dataset factory to the local v3 adapter."""
    import openpi.training.data_loader as data_loader

    root = Path(dataset_root).expanduser().resolve()

    def create_torch_dataset(_data_config, action_horizon, _model_config):
        return LeRobotV3Dataset(root, action_horizon)

    data_loader.create_torch_dataset = create_torch_dataset
    from easy_embodied_deploy.pose_training_jax import install_loader
    install_loader()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True)
    args = parser.parse_args()
    dataset = LeRobotV3Dataset(args.check, action_horizon=50)
    sample = dataset[0]
    if sample["observation.state"].shape != (14,) or sample["action"].shape != (50, 14):
        raise ValueError("首帧 state/action 形状异常")
    for key in IMAGE_KEYS:
        image = np.asarray(sample[key])
        if image.shape != (3, 224, 224) or not np.isfinite(image).all():
            raise ValueError(f"首帧 {key} 解码异常")
    print(json.dumps({"frames": len(dataset), "sample": "ok", "task": sample["task"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
