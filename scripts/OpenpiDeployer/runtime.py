"""JAX OpenPI policy loading and chunk scheduling for dual Piper deployment."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LEROBOT_DEPLOYER_DIR = ROOT / "scripts/LerobotDeployer"
OPENPI_TRAINER_DIR = ROOT / "scripts/OpenpiTrainer"

CAMERAS = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
STATE_KEY = "observation.state"
IMAGE_KEYS = tuple(f"observation.images.{name}" for name in CAMERAS)
ACTION_DIM = 14
ACTION_HORIZON = 50
GRIPPER_INDICES = (6, 13)


class OpenpiDeploymentError(RuntimeError):
    """A checkpoint, observation, or inference result violates the Piper contract."""


def resolve_checkpoint(value: str | Path) -> Path:
    """Resolve a run, numbered step, or params directory to an OpenPI step."""
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path.name == "params" and (path / "_METADATA").is_file():
        path = path.parent
    if (path / "params/_METADATA").is_file() and (path / "assets").is_dir():
        return path
    if path.is_dir():
        steps = sorted(
            (
                child
                for child in path.iterdir()
                if child.is_dir()
                and child.name.isdigit()
                and (child / "params/_METADATA").is_file()
                and (child / "assets").is_dir()
            ),
            key=lambda item: int(item.name),
        )
        if steps:
            return steps[-1]
    raise OpenpiDeploymentError(
        f"OpenPI checkpoint must contain params/_METADATA and assets: {path}"
    )


def load_training_manifest(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (
        checkpoint / "training_manifest.json",
        checkpoint.parent / "training_manifest.json",
        checkpoint.parent.parent / "training_manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise OpenpiDeploymentError(f"cannot read {candidate}: {exc}") from exc
            config = payload.get("config")
            if payload.get("trainer") != "OpenpiTrainer" or not isinstance(config, dict):
                raise OpenpiDeploymentError(f"not an OpenpiTrainer manifest: {candidate}")
            return candidate, payload
    raise OpenpiDeploymentError(
        f"training_manifest.json is required beside the checkpoint: {checkpoint}"
    )


def validate_manifest(
    payload: dict[str, Any],
    *,
    action_mode: str,
    inference_mode: str,
    rtc_mode: str,
    rtc_horizon: int,
    queue_threshold: int,
) -> dict[str, Any]:
    config = payload["config"]
    if config.get("action_mode") != action_mode:
        raise OpenpiDeploymentError(
            f"checkpoint action_mode={config.get('action_mode')!r}, requested {action_mode!r}"
        )
    if int(config.get("action_horizon", ACTION_HORIZON)) != ACTION_HORIZON:
        raise OpenpiDeploymentError("this deployer requires an OpenPI action horizon of 50")
    if inference_mode == "rtc" and queue_threshold >= ACTION_HORIZON:
        raise OpenpiDeploymentError("RTC queue threshold must be below action horizon 50")
    trained_delay = int(config.get("rtc_max_delay", 0) or 0)
    if rtc_mode == "trained":
        if inference_mode != "rtc":
            raise OpenpiDeploymentError("trained RTC requires inference_mode=rtc")
        if trained_delay <= 0:
            raise OpenpiDeploymentError(
                "trained RTC requires a checkpoint trained with rtc_max_delay > 0"
            )
        if rtc_horizon < trained_delay or queue_threshold < trained_delay:
            raise OpenpiDeploymentError(
                "trained RTC horizon and queue threshold must both be at least "
                f"rtc_max_delay={trained_delay}"
            )
        if rtc_horizon > ACTION_HORIZON - trained_delay:
            raise OpenpiDeploymentError(
                "trained RTC horizon must be <= action_horizon - rtc_max_delay "
                f"({ACTION_HORIZON - trained_delay})"
            )
    return config


def policy_observation(observation: dict[str, Any], task: str) -> dict[str, Any]:
    """Map the local bridge schema to the raw schema consumed by PiperInputs."""
    missing = {STATE_KEY, *IMAGE_KEYS}.difference(observation)
    if missing:
        raise OpenpiDeploymentError(f"observation is missing: {sorted(missing)}")
    state = np.asarray(observation[STATE_KEY], dtype=np.float32)
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise OpenpiDeploymentError(f"state must be finite float32[14], got {state.shape}")
    images: dict[str, np.ndarray] = {}
    for camera, key in zip(CAMERAS, IMAGE_KEYS, strict=True):
        image = np.asarray(observation[key])
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise OpenpiDeploymentError(
                f"{key} must be uint8 HWC (224, 224, 3), got {image.dtype} {image.shape}"
            )
        images[camera] = image
    return {"state": state, "images": images, "prompt": task}


def load_policy(
    checkpoint: Path,
    manifest_config: dict[str, Any],
    *,
    task: str,
    num_steps: int,
):
    """Load one OpenPI JAX policy with local transforms and the CE RTC patch."""
    import sys

    if str(OPENPI_TRAINER_DIR) not in sys.path:
        sys.path.insert(0, str(OPENPI_TRAINER_DIR))
    from openpi.models import pi0_realtime
    from openpi.policies import policy_config
    from openpi_config import build_inference_config

    pi0_realtime.install_realtime_patch()
    config = build_inference_config(manifest_config, checkpoint)
    return policy_config.create_trained_policy(
        config,
        checkpoint,
        default_prompt=task,
        sample_kwargs={"num_steps": int(num_steps)},
    )


@dataclass(frozen=True)
class RTCSettings:
    mode: str = "guided"
    horizon: int = 20
    queue_threshold: int = 40
    schedule: str = "exp"
    max_guidance_weight: float = 5.0
    mask_grippers: bool = True
    policy_hz: float = 10.0
    # Maximum overlap the checkpoint saw during train-time RTC.  Zero means
    # this is a guided-only checkpoint and no trained-delay limit applies.
    training_max_delay: int = 0


class PolicyChunkSampler:
    """Build physically aligned RTC requests around an OpenPI Policy."""

    def __init__(self, policy: Any, task: str, rtc: RTCSettings | None) -> None:
        self.policy = policy
        self.task = task
        self.rtc = rtc
        self.last_guidance_horizon: int | None = None
        self.latencies: deque[float] = deque(maxlen=1000)

    def warmup(self) -> None:
        """Compile plain and selected RTC sampling before hardware is connected."""
        observation: dict[str, Any] = {
            STATE_KEY: np.zeros((ACTION_DIM,), dtype=np.float32),
            **{
                key: np.zeros((224, 224, 3), dtype=np.uint8)
                for key in IMAGE_KEYS
            },
        }
        self.infer(observation, force_no_guidance=True)
        if self.rtc is not None:
            self.infer(
                observation,
                prev_action_chunk=np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32),
                prefix_start=1,
            )
        self.last_guidance_horizon = None
        self.latencies.clear()

    def infer(
        self,
        observation: dict[str, Any],
        *,
        prev_action_chunk: np.ndarray | None = None,
        prefix_start: int = 0,
        force_no_guidance: bool = False,
    ) -> np.ndarray:
        request = policy_observation(observation, self.task)
        self.last_guidance_horizon = None
        if self.rtc is not None and prev_action_chunk is not None and not force_no_guidance:
            previous = np.asarray(prev_action_chunk, dtype=np.float32)
            if previous.ndim != 2 or previous.shape[1] != ACTION_DIM:
                raise OpenpiDeploymentError(
                    f"RTC previous actions must be [T, {ACTION_DIM}], got {previous.shape}"
                )
            if not np.isfinite(previous).all():
                raise OpenpiDeploymentError("RTC previous actions contain NaN or infinite values")
            # OpenPI/JAX compiles a fixed 50-frame request.  The queue itself
            # owns the already-consumed boundary; only the unconsumed tail is
            # passed here, so it must never be shifted a second time.
            take = min(len(previous), ACTION_HORIZON)
            aligned = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
            aligned[:take] = previous[:take]
            if take:
                effective = min(self.rtc.horizon, take)
                locked = max(0, min(int(prefix_start), effective))
                mask = np.ones((ACTION_DIM,), dtype=np.float32)
                if self.rtc.mask_grippers:
                    mask[list(GRIPPER_INDICES)] = 0.0
                request.update(
                    prev_action_chunk=aligned,
                    prefix_attention_start=locked,
                    prefix_attention_horizon=effective,
                    prefix_attention_schedule=self.rtc.schedule,
                    max_guidance_weight=float(self.rtc.max_guidance_weight),
                    guidance_action_dim=ACTION_DIM,
                    guidance_action_mask=mask,
                    rtc_sample_mode=("train_time" if self.rtc.mode == "trained" else "guidance"),
                )
                self.last_guidance_horizon = effective
        started = time.perf_counter()
        result = self.policy.infer(request)
        elapsed = time.perf_counter() - started
        self.latencies.append(elapsed)
        actions = np.asarray(result.get("actions") if isinstance(result, dict) else result, dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise OpenpiDeploymentError(
                f"OpenPI actions must be ({ACTION_HORIZON}, {ACTION_DIM}), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise OpenpiDeploymentError("OpenPI returned NaN or infinite actions")
        return actions


class SyncActionEngine:
    def __init__(self, sampler: PolicyChunkSampler, n_action_steps: int) -> None:
        self.sampler = sampler
        self.n_action_steps = int(n_action_steps)
        self._queue: deque[np.ndarray] = deque()

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def failed(self) -> bool:
        return False

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        if not self._queue:
            chunk = self.sampler.infer(observation)
            self._queue.extend(chunk[: self.n_action_steps])
        return self._queue.popleft()

    def close(self) -> None:
        self._queue.clear()


class AsyncRTCActionEngine:
    """One-worker RTC queue with explicit leftover-prefix replacement.

    The queue contains only actions that have not been consumed by the control
    loop.  Each refill conditions on that exact leftover and replaces it with
    the new chunk suffix after measured inference latency; it never appends a
    second chunk behind stale actions.
    """

    def __init__(self, sampler: PolicyChunkSampler, settings: RTCSettings) -> None:
        self.sampler = sampler
        self.settings = settings
        self._queue: deque[np.ndarray] = deque()
        self._condition = threading.Condition()
        self._latest_observation: dict[str, Any] | None = None
        self._requested = False
        self._stopping = False
        self._in_flight = False
        self._failure: BaseException | None = None
        self._consecutive_discards = 0
        self.enqueue_events: deque[dict[str, Any]] = deque(maxlen=1000)
        self._worker = threading.Thread(target=self._run, daemon=True, name="openpi-rtc-worker")
        self._worker.start()

    @staticmethod
    def _copy_observation(observation: dict[str, Any]) -> dict[str, Any]:
        return {key: np.asarray(value).copy() for key, value in observation.items()}

    @property
    def queue_size(self) -> int:
        with self._condition:
            return len(self._queue)

    @property
    def failed(self) -> bool:
        with self._condition:
            return self._failure is not None

    @property
    def failure(self) -> BaseException | None:
        with self._condition:
            return self._failure

    def get_action(self, observation: dict[str, Any]) -> np.ndarray | None:
        with self._condition:
            self._latest_observation = self._copy_observation(observation)
            action = self._queue.popleft() if self._queue else None
            if (
                len(self._queue) <= self.settings.queue_threshold
                and not self._in_flight
                and self._failure is None
            ):
                self._requested = True
                self._condition.notify()
            return action

    def prefill(
        self,
        observation: dict[str, Any],
        timeout_s: float,
        heartbeat: Callable[[], Any] | None = None,
    ) -> bool:
        """Submit the first observation and wait for a usable action chunk.

        RTC is intentionally asynchronous after startup, but allowing the
        control loop to run before the first chunk exists turns normal JAX
        compilation/inference latency into false ``starved`` ticks.  Prefill
        happens before the first command tick and therefore those ticks are
        never sent to the bridge or counted as runtime starvation.
        """
        with self._condition:
            self._latest_observation = self._copy_observation(observation)
            if self._failure is None and not self._in_flight and not self._queue:
                self._requested = True
                self._condition.notify()
        return self.wait_for_action(timeout_s, heartbeat=heartbeat)

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._requested or self._stopping)
                if self._stopping:
                    return
                if self._latest_observation is None:
                    self._requested = False
                    continue
                if len(self._queue) > self.settings.queue_threshold:
                    self._requested = False
                    continue
                observation = self._copy_observation(self._latest_observation)
                qsize_at_snapshot = len(self._queue)
                previous_actions = (
                    np.stack(tuple(self._queue), axis=0).astype(np.float32, copy=False)
                    if self._queue
                    else None
                )
                self._requested = False
                self._in_flight = True
            has_previous_actions = previous_actions is not None and len(previous_actions) > 0
            if has_previous_actions:
                # The prefix is conditioned on the measured wall-clock cost
                # of the preceding inference.  A trained checkpoint can only
                # condition on prefixes seen during training, so its maximum
                # is a cap after estimating the runtime delay, never the
                # runtime delay estimate itself.
                observed_latency = max(self.sampler.latencies, default=0.0)
                inference_delay = max(
                    1,
                    int(math.ceil(observed_latency * self.settings.policy_hz)),
                )
                if self.settings.mode == "trained" and self.settings.training_max_delay > 0:
                    inference_delay = min(inference_delay, self.settings.training_max_delay)
                inference_delay = min(inference_delay, len(previous_actions), ACTION_HORIZON - 1)
            else:
                inference_delay = 0
            try:
                chunk = self.sampler.infer(
                    observation,
                    prev_action_chunk=previous_actions,
                    prefix_start=inference_delay,
                )
                actual_s = self.sampler.latencies[-1]
                actual_frames = max(0, int(math.ceil(actual_s * self.settings.policy_hz)))
                with self._condition:
                    if self._stopping:
                        return
                    qsize_at_return = len(self._queue)
                    measured_delay = actual_frames
                    if (
                        self.settings.mode == "trained"
                        and has_previous_actions
                        and self.settings.training_max_delay > 0
                        and measured_delay > min(inference_delay, self.settings.training_max_delay)
                    ):
                        self._consecutive_discards += 1
                        self.enqueue_events.append(
                            {
                                "qsize_at_snapshot": qsize_at_snapshot,
                                "qsize_at_return": qsize_at_return,
                                "prev_action_steps": len(previous_actions),
                                "conditioned_delay": inference_delay,
                                "actual_infer_frames": actual_frames,
                                "start": None,
                                "pushed": 0,
                                "queue_after": len(self._queue),
                                "guidance_horizon": self.sampler.last_guidance_horizon,
                                "latency_s": actual_s,
                                "discarded": True,
                                "consecutive_discards": self._consecutive_discards,
                            }
                        )
                        if self._consecutive_discards >= 5:
                            raise OpenpiDeploymentError(
                                "trained RTC inference delay exceeded rtc_max_delay for "
                                f"{self._consecutive_discards} consecutive chunks"
                            )
                        # Keep the old queue and retry after the consumer has
                        # advanced.  Appending an unusable chunk here would
                        # silently violate the train-time RTC contract.
                        if len(self._queue) <= self.settings.queue_threshold:
                            self._requested = True
                            self._condition.notify()
                        continue

                    self._consecutive_discards = 0
                    # RTC replaces the remaining queue.  The first
                    # ``actual_frames`` outputs elapsed while inference was
                    # running, so only the unconsumed suffix is executable.
                    start = min(actual_frames, len(chunk))
                    selected = chunk[start:]
                    self._queue.clear()
                    self._queue.extend(selected)
                    self.enqueue_events.append(
                        {
                            "qsize_at_snapshot": qsize_at_snapshot,
                            "qsize_at_return": qsize_at_return,
                            "prev_action_steps": 0 if previous_actions is None else len(previous_actions),
                            "conditioned_delay": inference_delay,
                            "actual_infer_frames": actual_frames,
                            "start": start,
                            "pushed": len(selected),
                            "queue_after": len(self._queue),
                            "guidance_horizon": self.sampler.last_guidance_horizon,
                            "latency_s": actual_s,
                            "discarded": False,
                        }
                    )
                    if len(self._queue) <= self.settings.queue_threshold:
                        self._requested = True
                        self._condition.notify()
            except BaseException as exc:  # worker failures must reach the control loop
                with self._condition:
                    self._failure = exc
                    self._requested = False
            finally:
                with self._condition:
                    self._in_flight = False

    def wait_for_action(
        self,
        timeout_s: float,
        heartbeat: Callable[[], Any] | None = None,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        next_heartbeat = time.monotonic()
        while time.monotonic() < deadline:
            with self._condition:
                if self._queue:
                    return True
                if self._failure is not None:
                    return False
            if heartbeat is not None and time.monotonic() >= next_heartbeat:
                heartbeat()
                next_heartbeat = time.monotonic() + min(5.0, max(0.5, timeout_s / 4.0))
            time.sleep(0.005)
        return False

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._worker.join(timeout=10.0)
        with self._condition:
            self._queue.clear()


def latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"inference_chunks": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "inference_chunks": len(values),
        "inference_latency_s_last": float(array[-1]),
        "inference_latency_s_median": float(np.median(array)),
        "inference_latency_s_max": float(np.max(array)),
    }
