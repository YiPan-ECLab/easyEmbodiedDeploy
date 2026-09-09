"""Write direct pyAgxArm observations and actions to HDF5."""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from .camera import CAMERA_NAMES, CameraFrame
from .robot import SIDES, ArmSample

TASK_PHASE = 0
HOME_PHASE = 1
ACTION_LAYOUT = (
    "left_j1_rad,left_j2_rad,left_j3_rad,left_j4_rad,left_j5_rad,left_j6_rad,"
    "left_gripper_m,right_j1_rad,right_j2_rad,right_j3_rad,right_j4_rad,"
    "right_j5_rad,right_j6_rad,right_gripper_m"
)


@dataclass
class EpisodeData:
    timestamps: list[float] = field(default_factory=list)
    phases: list[int] = field(default_factory=list)
    camera_timestamps: list[list[float]] = field(default_factory=list)
    images: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {name: [] for name in CAMERA_NAMES}
    )
    samples: dict[str, list[ArmSample]] = field(
        default_factory=lambda: {side: [] for side in SIDES}
    )

    def append(
        self,
        frames: dict[str, CameraFrame],
        samples: dict[str, ArmSample],
        phase: int,
    ) -> None:
        self.timestamps.append(time.monotonic())
        self.phases.append(phase)
        self.camera_timestamps.append([frames[name].timestamp for name in CAMERA_NAMES])
        for name in CAMERA_NAMES:
            self.images[name].append(frames[name].rgb)
        for side in SIDES:
            self.samples[side].append(samples[side])

    def __len__(self) -> int:
        return len(self.timestamps)


def _side_arrays(samples: list[ArmSample]) -> dict[str, np.ndarray]:
    joints = np.asarray([sample.joints for sample in samples], dtype=np.float64)
    poses = np.asarray([sample.flange_pose for sample in samples], dtype=np.float64)
    grippers = np.asarray(
        [[sample.gripper_width] for sample in samples], dtype=np.float64
    )
    forces = np.asarray(
        [[sample.gripper_force] for sample in samples], dtype=np.float64
    )
    actions = np.asarray(
        [(*sample.action.joints, sample.action.gripper_width) for sample in samples],
        dtype=np.float64,
    )
    action_timestamps = np.asarray(
        [sample.action.timestamp for sample in samples], dtype=np.float64
    )
    return {
        "joints": joints,
        "poses": poses,
        "grippers": grippers,
        "forces": forces,
        "states": np.concatenate((joints, grippers), axis=1),
        "actions": actions,
        "action_timestamps": action_timestamps,
    }


def _create_dataset(group, name: str, data: np.ndarray):
    return group.create_dataset(name, data=data, compression="lzf", shuffle=True)


def _set_format(dataset, *, units: str, layout: str = "") -> None:
    dataset.attrs["normalized"] = False
    dataset.attrs["units"] = units
    if layout:
        dataset.attrs["layout"] = layout


def _create_formatted_dataset(
    group, name: str, data: np.ndarray, *, units: str, layout: str = ""
):
    dataset = _create_dataset(group, name, data)
    _set_format(dataset, units=units, layout=layout)
    return dataset


def write_episode(
    episode: EpisodeData,
    instruction: str,
    output_dir: Path,
    *,
    pyagxarm_version: str,
    firmware_profile: str,
    firmware_version: str,
    gripper_force: float,
) -> Path:
    if not episode:
        raise ValueError("空 episode 不能保存")

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"pyagxarm_dual_{stamp}.hdf5"
    temporary = path.with_suffix(".hdf5.tmp")
    t0 = episode.timestamps[0]
    sides = {side: _side_arrays(episode.samples[side]) for side in SIDES}

    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = "pyagxarm_direct_v1"
            handle.attrs["control_library"] = "pyAgxArm"
            handle.attrs["pyagxarm_version"] = pyagxarm_version
            handle.attrs["firmware_profile"] = firmware_profile
            handle.attrs["firmware_version"] = firmware_version
            handle.attrs["normalized"] = False

            data = handle.create_group("data")
            demo = data.create_group("demo_0")
            demo.attrs["num_samples"] = len(episode)
            demo.attrs["language"] = instruction
            demo.attrs["gripper_force_n"] = gripper_force

            obs = demo.create_group("obs")
            for name in CAMERA_NAMES:
                image_dataset = _create_dataset(
                    obs, name, np.stack(episode.images[name])
                )
                _set_format(image_dataset, units="uint8", layout="height,width,rgb")

            for side in SIDES:
                joint_dataset = _create_dataset(
                    obs, f"{side}_joint_states", sides[side]["joints"]
                )
                _set_format(joint_dataset, units="rad", layout="j1,j2,j3,j4,j5,j6")
                pose_dataset = _create_dataset(
                    obs, f"{side}_ee_states", sides[side]["poses"]
                )
                _set_format(
                    pose_dataset,
                    units="m,m,m,rad,rad,rad",
                    layout="x,y,z,roll,pitch,yaw",
                )
                gripper_dataset = _create_dataset(
                    obs, f"{side}_gripper_states", sides[side]["grippers"]
                )
                _set_format(gripper_dataset, units="m", layout="width")
                force_dataset = _create_dataset(
                    obs, f"{side}_gripper_force", sides[side]["forces"]
                )
                _set_format(force_dataset, units="N", layout="force")

            joint_states = np.concatenate(
                [sides[side]["joints"] for side in SIDES], axis=1
            )
            ee_states = np.concatenate([sides[side]["poses"] for side in SIDES], axis=1)
            gripper_states = np.concatenate(
                [sides[side]["grippers"] for side in SIDES], axis=1
            )
            states = np.concatenate([sides[side]["states"] for side in SIDES], axis=1)
            joint_dataset = _create_dataset(obs, "joint_states", joint_states)
            _set_format(joint_dataset, units="rad", layout="left_j1..j6,right_j1..j6")
            pose_dataset = _create_dataset(obs, "ee_states", ee_states)
            _set_format(
                pose_dataset,
                units="m,m,m,rad,rad,rad per arm",
                layout="left_xyzrpy,right_xyzrpy",
            )
            gripper_dataset = _create_dataset(obs, "gripper_states", gripper_states)
            _set_format(gripper_dataset, units="m", layout="left_width,right_width")
            states_dataset = _create_dataset(demo, "states", states)
            _set_format(
                states_dataset,
                units="rad x6,m per arm",
                layout="left_j1..j6,left_gripper,right_j1..j6,right_gripper",
            )
            demo["robot_states"] = states_dataset

            for side in SIDES:
                side_actions = _create_dataset(
                    demo, f"{side}_actions", sides[side]["actions"]
                )
                _set_format(
                    side_actions,
                    units="rad,rad,rad,rad,rad,rad,m",
                    layout="j1,j2,j3,j4,j5,j6,gripper_width",
                )
            actions = _create_dataset(
                demo,
                "actions",
                np.concatenate([sides[side]["actions"] for side in SIDES], axis=1),
            )
            _set_format(
                actions,
                units="rad x6,m per arm",
                layout=ACTION_LAYOUT,
            )
            actions.attrs["replay"] = (
                "move_j(action[:6]); move_gripper_m(action[6], force=gripper_force_n)"
            )

            _create_formatted_dataset(
                demo,
                "timestamps",
                np.asarray(episode.timestamps) - t0,
                units="s",
                layout="seconds_since_first_frame",
            )
            _create_formatted_dataset(
                demo,
                "camera_timestamps",
                np.asarray(episode.camera_timestamps, dtype=np.float64) - t0,
                units="s",
                layout="agentview,left_eye_in_hand,right_eye_in_hand",
            )
            _create_formatted_dataset(
                demo,
                "action_source_timestamps",
                np.column_stack(
                    [sides[side]["action_timestamps"] - t0 for side in SIDES]
                ),
                units="s",
                layout="left,right",
            )
            _create_formatted_dataset(
                demo,
                "trajectory_phase",
                np.asarray(episode.phases, dtype=np.int8),
                units="enum",
                layout="0=task,1=home",
            )
            _create_formatted_dataset(
                demo,
                "rewards",
                np.zeros(len(episode), dtype=np.float32),
                units="dimensionless",
                layout="per_frame",
            )
            dones = np.zeros(len(episode), dtype=np.bool_)
            dones[-1] = True
            _create_formatted_dataset(
                demo,
                "dones",
                dones,
                units="boolean",
                layout="per_frame",
            )
            handle.flush()
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path
