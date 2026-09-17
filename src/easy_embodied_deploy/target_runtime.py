"""Decode complete prediction chunks before RTC drops elapsed time steps."""

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from .action_contract import POSE_MODES, validate_pose_contract
from .pose_actions import PoseCodec, calibration_for_mode, finite, fingerprint

POSE_KEY = "observation.flange_pose"
COMMAND_KEY = "command_anchor"


def checkpoint_pose_contract(checkpoint, mode, *, mount_id="", policy_hz=None):
    if mode not in POSE_MODES:
        return None
    checkpoint = Path(checkpoint)
    for directory in (checkpoint, *list(checkpoint.parents)[:3]):
        path = directory / "training_manifest.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            contract = manifest.get("pose_contract")
            validate_pose_contract(contract)
            if contract["action_mode"] != mode or manifest["config"]["action_mode"] != mode:
                raise ValueError("checkpoint 位姿模式与所选模式不一致")
            if fingerprint(contract.get("calibration")) != contract.get("calibration_sha256"):
                raise ValueError("checkpoint 标定快照校验失败")
            if mode != "absolute_base_pose" and mount_id != contract["mount_id"]:
                raise ValueError("请确认当前相机安装版本与训练标定 mount_id 一致")
            if policy_hz is not None and not np.isclose(policy_hz, contract["fps"]):
                raise ValueError("位姿部署 policy_hz 必须等于训练数据 FPS")
            PoseCodec(mode, contract.get("calibration"))
            calibration_for_mode(contract.get("calibration"), mode, contract["mount_id"])
            return contract
    raise ValueError("位姿 checkpoint 必须附带 training_manifest.json")


def validate_live_cameras(contract, configured):
    if not contract or contract["action_mode"] == "absolute_base_pose" or configured.get("offline"):
        return
    roles = {"left_eye_in_hand", "right_eye_in_hand"} if contract["action_mode"] == "camera_delta_wrist" else {"agentview"}
    live = configured.get("cameras", {})
    jobs = {"left_wrist", "right_wrist"} if contract["action_mode"] == "camera_delta_wrist" else {"global_left", "global_right"}
    for item in contract["calibration"]["calibrations"]:
        if item.get("job") not in jobs:
            continue
        camera = item["camera"]
        if camera["role"] in roles:
            actual = live.get(camera["role"], {})
            if any(actual.get(key) != camera.get(key) for key in ("serial", "sensor", "width", "height")):
                raise ValueError(f"{camera['role']} 的实际相机 / 视频配置与训练标定不一致")


class TargetSampler:
    """Queue absolute base targets, re-encode RTC prefixes at the new anchor.

    Joint deltas are successive increments. Camera deltas are independent
    future targets relative to one observation. Neither may be decoded after
    a queue has removed the beginning of the chunk.
    """

    absolute_targets = True

    def __init__(self, sampler, mode, contract=None):
        self.sampler, self.mode = sampler, mode
        self.codec = PoseCodec(mode, contract.get("calibration")) if mode in POSE_MODES else None
        self._latencies = deque(maxlen=1000)

    @property
    def latencies(self):
        return self._latencies

    @property
    def last_guidance_horizon(self):
        return self.sampler.last_guidance_horizon

    def infer(self, observation, *, prev_action_chunk=None, **kwargs):
        started = time.perf_counter()
        previous = prev_action_chunk
        if self.codec is not None:
            anchor = finite(observation[POSE_KEY], (12,), "部署法兰位姿")
            if previous is not None:
                previous = np.asarray([self.codec.encode(anchor, np.r_[row[:6], row[7:13]], np.clip(row[[6, 13]], 0, .1))
                                       for row in previous], dtype=np.float32)
        elif self.mode == "delta_joint":
            anchor = finite(observation[COMMAND_KEY], (14,), "增量关节指令锚点")
            if previous is not None:
                previous = np.diff(np.vstack((anchor, previous)), axis=0).astype(np.float32)
        chunk = self.sampler.infer(observation, prev_action_chunk=previous, **kwargs)
        if self.codec is not None:
            targets = []
            for row in chunk:
                pose, grip = self.codec.decode(anchor, row)
                targets.append(np.r_[pose[:6], grip[0], pose[6:], grip[1]])
            result = np.asarray(targets)
        elif self.mode == "delta_joint":
            result = anchor + np.cumsum(chunk.astype(np.float64), axis=0)
        else:
            result = chunk
        self._latencies.append(time.perf_counter()-started)
        return result


class PoseTargetAdapter:
    """The sampler already decoded xyz/RPY into the SDK's base frame."""
    last_was_limited = False

    def reset(self, state):
        finite(state, (14,), "state")

    def to_absolute(self, target, state):
        finite(state, (14,), "state")
        value = finite(target, (14,), "绝对法兰目标").copy()
        requested = value.copy()
        value[[6, 13]] = np.clip(value[[6, 13]], 0, .1)
        self.last_was_limited = not np.allclose(value, requested, atol=1e-7, rtol=0)
        return value


class TorchPolicyAdapter:
    """Use native LeRobot processors and PI05 sampling with the shared queue."""

    def __init__(self, policy, preprocessor, postprocessor, device):
        self.policy, self.preprocessor = policy, preprocessor
        self.postprocessor, self.device = postprocessor, device

    def infer(self, request):
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.processor.normalize_processor import NormalizerProcessorStep
        from lerobot.lerobot_types import TransitionKey

        obs = {"observation.state": request["state"],
               **{f"observation.images.{k}": v for k, v in request["images"].items()}}
        # Guided RTC locally enables gradients with respect to noisy actions.
        # inference_mode would prevent that even inside native enable_grad().
        with torch.no_grad():
            prepared = prepare_observation_for_inference(obs, torch.device(self.device), request["prompt"], "piper_dual")
            batch = self.preprocessor(prepared)
            kwargs = {}
            if "prev_action_chunk" in request:
                horizon = int(request["prefix_attention_horizon"])
                previous = torch.as_tensor(request["prev_action_chunk"][:horizon], device=self.device).unsqueeze(0)
                normalizer = next(s for s in self.preprocessor.steps if isinstance(s, NormalizerProcessorStep))
                previous = normalizer({TransitionKey.ACTION: previous})[TransitionKey.ACTION]
                kwargs.update(inference_delay=int(request["prefix_attention_start"]), prev_chunk_left_over=previous.squeeze(0))
            chunk = self.policy.predict_action_chunk(batch, **kwargs)
            return {"actions": self.postprocessor(chunk).squeeze(0).float().cpu().numpy()}


class EngineFacade:
    """LeRobot console lifecycle around the absolute-target queue."""

    def __init__(self, engine):
        self.engine = engine

    def start(self):
        pass

    def resume(self):
        pass

    def stop(self):
        self.engine.close()

    @property
    def failed(self):
        return self.engine.failed

    @property
    def failure_traceback(self):
        return str(getattr(self.engine, "failure", ""))

    def get_action(self, observation):
        return self.engine.get_action(observation)

    def qsize(self):
        return self.engine.queue_size

    @property
    def action_queue(self):
        return self
