"""Piper-specific validation and action conversion for LeRobot deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


SIDES = ("left", "right")
CAMERAS = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
AXES = tuple(
    f"{side}_{axis}"
    for side in SIDES
    for axis in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
)
STATE_KEY = "observation.state"
IMAGE_KEYS = tuple(f"observation.images.{name}" for name in CAMERAS)


def _feature_names(feature: dict[str, Any]) -> tuple[str, ...]:
    names = feature.get("names")
    if isinstance(names, dict):
        names = names.get("axes")
    if not isinstance(names, (list, tuple)):
        raise ValueError("vector feature is missing an ordered names/axes list")
    return tuple(str(name) for name in names)


def piper_feature_contract() -> dict[str, dict[str, Any]]:
    """Build the dataset-shaped feature mapping required by LeRobot rollout APIs."""
    features: dict[str, dict[str, Any]] = {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (14,),
            "names": {"axes": list(AXES)},
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": {"axes": list(AXES)},
        },
    }
    for key in IMAGE_KEYS:
        features[key] = {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def validate_feature_contract(features: dict[str, dict[str, Any]]) -> None:
    """Reject feature mappings whose axes or cameras differ from Piper."""
    required = {STATE_KEY, "action", *IMAGE_KEYS}
    missing = required.difference(features)
    if missing:
        raise ValueError(f"checkpoint is missing deployment features: {sorted(missing)}")
    for key in (STATE_KEY, "action"):
        feature = features[key]
        if tuple(feature.get("shape", ())) != (14,) or feature.get("dtype") != "float32":
            raise ValueError(f"{key} must be float32[14], got {feature}")
        names = _feature_names(feature)
        if names != AXES:
            raise ValueError(f"{key} axis order differs from Piper: {names}")
    for key in IMAGE_KEYS:
        feature = features[key]
        if tuple(feature.get("shape", ())) != (224, 224, 3):
            raise ValueError(f"{key} must have HWC shape (224, 224, 3)")
        if feature.get("dtype") not in ("image", "video"):
            raise ValueError(f"{key} must be an image/video feature")


def validate_checkpoint_contract(config: dict[str, Any]) -> None:
    """Verify that checkpoint model features match the fixed Piper hardware contract."""
    input_features = config.get("input_features")
    output_features = config.get("output_features")
    if not isinstance(input_features, dict) or not isinstance(output_features, dict):
        raise ValueError("checkpoint config must contain input_features and output_features")

    expected_inputs = {STATE_KEY, *IMAGE_KEYS}
    missing = expected_inputs.difference(input_features)
    if missing:
        raise ValueError(f"checkpoint input_features are missing: {sorted(missing)}")
    unexpected = set(input_features).difference(expected_inputs)
    if unexpected:
        raise ValueError(f"checkpoint has unsupported input_features: {sorted(unexpected)}")

    expected = {
        STATE_KEY: ("STATE", (14,)),
        **{key: ("VISUAL", (3, 224, 224)) for key in IMAGE_KEYS},
    }
    for key, (feature_type, shape) in expected.items():
        feature = input_features[key]
        if str(feature.get("type", "")).upper() != feature_type or tuple(
            feature.get("shape", ())
        ) != shape:
            raise ValueError(
                f"checkpoint {key} must be {feature_type}{shape}, got {feature}"
            )

    if set(output_features) != {"action"}:
        raise ValueError("checkpoint output_features must contain only action")
    action = output_features["action"]
    if str(action.get("type", "")).upper() != "ACTION" or tuple(
        action.get("shape", ())
    ) != (14,):
        raise ValueError(f"checkpoint action must be ACTION(14,), got {action}")


def features_for_rollout(
    features: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return the policy subset with LeRobot's vector names flattened."""
    validate_feature_contract(features)
    result: dict[str, dict[str, Any]] = {}
    for key in (STATE_KEY, "action", *IMAGE_KEYS):
        feature = dict(features[key])
        feature["shape"] = tuple(feature["shape"])
        if key in (STATE_KEY, "action"):
            feature["names"] = list(_feature_names(feature))
        result[key] = feature
    return result


def validate_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Validate a bridge observation without mutating the received mapping."""
    if not isinstance(observation, dict):
        raise ValueError("bridge observation must be a mapping")
    missing = {STATE_KEY, *IMAGE_KEYS}.difference(observation)
    if missing:
        raise ValueError(f"bridge observation is missing: {sorted(missing)}")

    state = np.asarray(observation[STATE_KEY], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"Piper state must be finite float32[14], got {state.shape}")
    checked = {STATE_KEY: state}
    if "observation.flange_pose" in observation:
        poses = np.asarray(observation["observation.flange_pose"], dtype=np.float64)
        if poses.shape != (12,) or not np.isfinite(poses).all():
            raise ValueError("flange pose must be finite xyz/RPY[12]")
        checked["observation.flange_pose"] = poses
    if "command_anchor" in observation:
        checked["command_anchor"] = PiperActionAdapter._vector(observation["command_anchor"], "command anchor")
    for key in IMAGE_KEYS:
        image = np.asarray(observation[key])
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"{key} must be uint8 HWC (224, 224, 3), got {image.dtype} {image.shape}"
            )
        checked[key] = image
    return checked


def to_rtc_hardware_observation(
    observation: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Convert dataset-shaped arrays to the raw mapping expected by RTC."""
    state = observation[STATE_KEY]
    raw: dict[str, Any] = {name: float(value) for name, value in zip(AXES, state)}
    raw.update(
        {
            key.removeprefix("observation.images."): observation[key]
            for key in IMAGE_KEYS
        }
    )
    return raw


@dataclass
class PiperActionAdapter:
    """Convert policy values into bounded absolute Piper targets.

    Delta datasets store ``command[t+1] - command[t]``.  Their correct anchor
    is therefore the last commanded target, initialized from the first follower
    observation, rather than the follower feedback sampled on every cycle.
    """

    action_mode: str
    joint_limits: np.ndarray
    gripper_limits: tuple[float, float]
    _target: np.ndarray | None = field(default=None, init=False, repr=False)
    last_was_limited: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.action_mode not in ("delta_joint", "absolute_joint"):
            raise ValueError(f"unknown action_mode: {self.action_mode}")
        # Keep targets in float64 so exact documented endpoints such as 0.1 m
        # are not rounded outside the hardware interval by float32 conversion.
        self.joint_limits = np.asarray(self.joint_limits, dtype=np.float64)
        if self.joint_limits.shape != (6, 2):
            raise ValueError("joint_limits must have shape (6, 2)")
        if not np.isfinite(self.joint_limits).all() or np.any(
            self.joint_limits[:, 0] >= self.joint_limits[:, 1]
        ):
            raise ValueError("joint_limits must contain finite low/high pairs")
        gripper_limits = np.asarray(self.gripper_limits, dtype=np.float64)
        if gripper_limits.shape != (2,) or not np.array_equal(
            gripper_limits, (0.0, 0.1)
        ):
            raise ValueError("Piper gripper limits must be exactly [0, 0.1] m")
        self.gripper_limits = (float(gripper_limits[0]), float(gripper_limits[1]))

    def reset(self, current_state: np.ndarray) -> None:
        state = self._vector(current_state, "state")
        self._target = self._clip_hardware_limits(state)
        self.last_was_limited = False

    @staticmethod
    def _vector(value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape != (14,) or not np.isfinite(vector).all():
            raise ValueError(f"Piper {name} must contain 14 finite values")
        return vector

    def _clip_hardware_limits(self, target: np.ndarray) -> np.ndarray:
        clipped = target.copy()
        for offset in (0, 7):
            clipped[offset : offset + 6] = np.clip(
                clipped[offset : offset + 6],
                self.joint_limits[:, 0],
                self.joint_limits[:, 1],
            )
        clipped[[6, 13]] = np.clip(
            clipped[[6, 13]], self.gripper_limits[0], self.gripper_limits[1]
        )
        return clipped

    def to_absolute(
        self, model_action: np.ndarray, current_state: np.ndarray
    ) -> np.ndarray:
        action = self._vector(model_action, "action")
        state = self._vector(current_state, "state")
        if self._target is None:
            self.reset(state)
        assert self._target is not None
        anchor = self._target
        requested = anchor + action if self.action_mode == "delta_joint" else action

        target = self._clip_hardware_limits(requested)
        self.last_was_limited = not np.allclose(target, requested, atol=1e-7)
        self._target = target
        return target.copy()

    @property
    def command_target(self):
        if self._target is None:
            raise ValueError("command anchor has not been initialized")
        return self._target.copy()

    @staticmethod
    def split(absolute_action: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
        action = PiperActionAdapter._vector(absolute_action, "target")
        return {
            "left": (action[:6], float(action[6])),
            "right": (action[7:13], float(action[13])),
        }


class PiperFeatureRobot:
    """Metadata-only robot facade required by LeRobot's RTC engine."""

    robot_type = "piper_dual"
    name = "piper_dual"
    is_connected = True
    cameras: dict[str, Any] = {}
    observation_features = {
        **{name: float for name in AXES},
        **{name: (224, 224, 3) for name in CAMERAS},
    }
    action_features = {name: float for name in AXES}

    def get_observation(self) -> dict[str, Any]:
        raise RuntimeError("metadata-only robot does not perform hardware I/O")

    def send_action(self, _action: Any) -> None:
        raise RuntimeError("metadata-only robot does not perform hardware I/O")
