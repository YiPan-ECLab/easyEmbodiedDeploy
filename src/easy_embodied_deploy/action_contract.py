"""Action metadata shared by consoles without loading ML or hardware libraries."""

import json
from pathlib import Path

JOINT_MODES = ("absolute_joint", "delta_joint")
POSE_MODES = ("absolute_base_pose", "camera_delta_wrist", "camera_delta_global")
ACTION_MODES = (*JOINT_MODES, *POSE_MODES)
STATE_AXES = [f"{side}_{axis}" for side in ("left", "right")
              for axis in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")]
POSE_AXES = [f"{side}_{axis}" for side in ("left", "right")
             for axis in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
POSE_CONTRACT = "pose_action_contract.json"


def action_axes(mode):
    if mode not in ACTION_MODES:
        raise ValueError(f"未知动作模式：{mode}")
    return POSE_AXES if mode in POSE_MODES else STATE_AXES


def read_pose_contract(root, required=False):
    path = Path(root) / "meta" / POSE_CONTRACT
    if not path.is_file():
        if required:
            raise ValueError(f"缺少位姿动作契约：{path}")
        return None
    contract = json.loads(path.read_text(encoding="utf-8"))
    validate_pose_contract(contract)
    return contract


def validate_pose_contract(contract):
    if (not isinstance(contract, dict)
            or contract.get("schema") != "piper_pose_actions" or contract.get("version") != 1
            or contract.get("action_mode") not in POSE_MODES
            or contract.get("state_axes") != STATE_AXES
            or contract.get("action_axes") != POSE_AXES
            or contract.get("target_offset") != 1
            or contract.get("rotation") != "rotvec_rad"
            or contract.get("position_units") != "m"
            or contract.get("state_units") != "rad x6,m per arm"
            or contract.get("pose_frame") != "own_base_flange"
            or contract.get("padding") != "repeat_last_target_mask_loss_and_statistics"
            or contract.get("chunk_anchor") != "observation_time"
            or contract.get("horizon") != 50
            or not isinstance(contract.get("fps"), int) or contract["fps"] <= 0):
        raise ValueError("位姿动作契约版本、字段、锚点或 horizon 不兼容")
    return contract
