#!/usr/bin/env python3
"""Deploy a trained LeRobot PI05 policy through the Piper hardware bridge."""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from easy_embodied_deploy.action_contract import ACTION_MODES, POSE_MODES
from easy_embodied_deploy.target_runtime import (COMMAND_KEY, TargetSampler, PoseTargetAdapter,
    TorchPolicyAdapter, EngineFacade, checkpoint_pose_contract, validate_live_cameras)
CHECKPOINT = ROOT / (
    "models/finetuned_models/pi05_run_lora_test/checkpoints/005000/pretrained_model"
)
DEFAULT_BRIDGE_ENDPOINT = "tcp://127.0.0.1:6007"
DEFAULT_TASK = "Remove the shipping label and place it on the package on the table"
DEFAULT_ACTION_MODE = "delta_joint"
DEFAULT_INFERENCE_MODE = "rtc"
DEFAULT_RTC_MODE = "guided"
DEFAULT_RTC_HORIZON = 20
DEFAULT_RTC_QUEUE_THRESHOLD = 40
DEFAULT_DEVICE = "cuda:0"
DEFAULT_POLICY_HZ = 10.0
DEFAULT_CONTROL_HZ = 30.0
DEFAULT_SMOOTHING_TIME_CONSTANT_S = 0.1
DEFAULT_MAX_STEPS = 30
DEFAULT_SEED = 20260909
DEFAULT_RPC_TIMEOUT_S = 5.0
DEFAULT_WATCHDOG_TIMEOUT_S = 30.0


class InferenceLatencyMonitor:
    """Measure actual policy chunk latency without changing LeRobot source."""

    def __init__(self) -> None:
        self._values: deque[float] = deque(maxlen=1000)
        self._lock = threading.Lock()

    def install(self, policy) -> None:
        get_base_model = getattr(policy, "get_base_model", None)
        target = get_base_model() if callable(get_base_model) else policy
        original = target.predict_action_chunk

        @functools.wraps(original)
        def measured(*args, **kwargs):
            started = time.perf_counter()
            output = original(*args, **kwargs)
            if torch.is_tensor(output) and output.is_cuda:
                torch.cuda.synchronize(output.device)
            elapsed = time.perf_counter() - started
            with self._lock:
                self._values.append(elapsed)
            return output

        target.predict_action_chunk = measured

    def snapshot(self) -> list[float]:
        with self._lock:
            return list(self._values)


class RuntimeDiagnostics:
    """Bounded runtime counters for queue starvation and action continuity."""

    def __init__(self, output: Path | None, *, expected_period_s: float) -> None:
        self.output = output
        self._stream = None
        self.ticks = 0
        self.actions = 0
        self.prefill_starved = 0
        self.runtime_starved = 0
        self.late_ticks = 0
        self.queue_min_after_first: int | None = None
        self.action_change_max = 0.0
        self.target_step_max = 0.0
        self._previous_action: np.ndarray | None = None
        self._previous_target: np.ndarray | None = None
        self._late_threshold_s = 1.5 * expected_period_s
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            self._stream = output.open("w", encoding="utf-8")

    def record(
        self,
        *,
        tick_s: float,
        observation_s: float,
        engine_s: float,
        queue_size: int | None,
        action: np.ndarray | None,
        target: np.ndarray | None,
        limited: bool,
    ) -> None:
        self.ticks += 1
        if tick_s > 0:
            self.late_ticks += int(tick_s > self._late_threshold_s)
        if action is None:
            if self.actions:
                self.runtime_starved += 1
            else:
                self.prefill_starved += 1
        else:
            self.actions += 1
            if self._previous_action is not None:
                self.action_change_max = max(
                    self.action_change_max,
                    float(np.linalg.norm(action - self._previous_action)),
                )
            if self._previous_target is not None and target is not None:
                self.target_step_max = max(
                    self.target_step_max,
                    float(np.linalg.norm(target - self._previous_target)),
                )
            self._previous_action = action.copy()
            self._previous_target = None if target is None else target.copy()
        if self.actions and queue_size is not None:
            self.queue_min_after_first = (
                queue_size
                if self.queue_min_after_first is None
                else min(self.queue_min_after_first, queue_size)
            )
        if self._stream is not None:
            self._stream.write(
                json.dumps(
                    {
                        "tick": self.ticks,
                        "tick_s": tick_s,
                        "observation_s": observation_s,
                        "engine_s": engine_s,
                        "queue_size": queue_size,
                        "action": None if action is None else action.tolist(),
                        "target": None if target is None else target.tolist(),
                        "hardware_limit_clipped": limited,
                    }
                )
                + "\n"
            )
            self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def summary(self, chunk_latencies: list[float]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ticks": self.ticks,
            "actions": self.actions,
            "prefill_starved_ticks": self.prefill_starved,
            "runtime_starved_ticks": self.runtime_starved,
            "queue_min_after_first": self.queue_min_after_first,
            "late_ticks_over_1_5x_period": self.late_ticks,
            "max_policy_action_change_l2": self.action_change_max,
            "max_absolute_target_step_l2": self.target_step_max,
            "inference_chunks": len(chunk_latencies),
        }
        if chunk_latencies:
            values = np.asarray(chunk_latencies)
            result.update(
                inference_latency_s_last=float(values[-1]),
                inference_latency_s_median=float(np.median(values)),
                inference_latency_s_max=float(values.max()),
            )
        return result


def resolve_path(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> float:
    if not 0 < args.control_hz <= 200:
        parser.error("--control-hz must be in (0, 200]")
    if not 0 <= args.smoothing_time_constant <= 2:
        parser.error("--smoothing-time-constant must be in [0, 2]")
    policy_hz = float(args.policy_hz)
    if not 0 < policy_hz <= 200:
        parser.error("--policy-hz must be in (0, 200]")
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    if args.seed < 0:
        parser.error("--seed cannot be negative")
    if args.rtc_horizon <= 0:
        parser.error("--rtc-horizon must be positive")
    if args.rtc_queue_threshold < 0:
        parser.error("--rtc-queue-threshold cannot be negative")
    if not math.isfinite(args.rpc_timeout) or args.rpc_timeout <= 0:
        parser.error("--rpc-timeout must be finite and positive")
    if not math.isfinite(args.watchdog_timeout) or args.watchdog_timeout < 5:
        parser.error("--watchdog-timeout must be at least 5 seconds")
    device_match = re.fullmatch(r"cuda(?::(\d+))?", args.device)
    if args.device != "cpu" and device_match is None:
        parser.error("--device must be cpu, cuda, or cuda:N (for example cuda:0)")
    if device_match is not None and torch.cuda.is_available():
        requested_index = int(device_match.group(1) or 0)
        if requested_index >= torch.cuda.device_count():
            parser.error(
                f"--device={args.device} is unavailable; detected "
                f"{torch.cuda.device_count()} CUDA device(s)"
            )
    if args.rtc_mode == "trained" and args.inference_mode != "rtc":
        parser.error("--rtc-mode only applies when --inference-mode=rtc")
    return policy_hz


def load_policy(
    checkpoint: Path,
    device: str,
    rtc_config: Any | None,
):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    from deployer import (
        features_for_rollout,
        piper_feature_contract,
        validate_checkpoint_contract,
    )

    print("[ 10%] validating checkpoint feature contract...", flush=True)
    raw_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    validate_checkpoint_contract(raw_config)
    feature_contract = piper_feature_contract()
    features = features_for_rollout(feature_contract)
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.pretrained_path = str(checkpoint)
    config.device = device
    if rtc_config is not None:
        config.rtc_config = rtc_config

    print("[ 20%] loading PI05 model weights (and adapter when configured)...", flush=True)
    done = threading.Event()

    def pulse() -> None:
        progress = 20
        while not done.wait(1):
            progress = min(85, progress + 1)
            print(f"\r[{progress:3d}%] loading model weights...", end="", flush=True)

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        # LeRobot's factory currently requires a ds_meta or env_cfg object even for
        # pretrained policies.  This in-memory shim supplies only the fixed Piper
        # feature contract; it never reads a dataset or provides normalization stats.
        runtime_features = SimpleNamespace(features=feature_contract, stats={})
        policy = make_policy(config, ds_meta=runtime_features)
        policy.eval()
    finally:
        done.set()
        thread.join()
        print()

    # Match LeRobot's rollout setup.  Setting this before construction already
    # initializes PI05 correctly; reinitializing here also covers wrappers such
    # as PEFT that expose the base policy after loading.
    if rtc_config is not None:
        policy.config.rtc_config = rtc_config
        initialize_rtc = getattr(policy, "init_rtc_processor", None)
        if callable(initialize_rtc):
            initialize_rtc()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    print("[100%] model, processors and checkpoint normalization ready", flush=True)
    return policy, preprocessor, postprocessor, features


def make_inference_engine(
    *,
    args: argparse.Namespace,
    policy,
    preprocessor,
    postprocessor,
    features,
    policy_hz: float,
    rtc_config,
):
    from deployer import AXES, PiperFeatureRobot
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    if args.action_mode != "absolute_joint":
        from scripts.OpenpiDeployer.runtime import (PolicyChunkSampler, RTCSettings,
                                                   SyncActionEngine, AsyncRTCActionEngine)
        rtc = None
        if args.inference_mode == "rtc":
            from scripts.OpenpiDeployer.runtime import validate_manifest
            config = {"action_mode": args.action_mode,
                      "rtc_max_delay": int(policy.config.rtc_training_max_delay)}
            validate_manifest({"config": config}, action_mode=args.action_mode,
                              inference_mode=args.inference_mode, rtc_mode=args.rtc_mode,
                              rtc_horizon=args.rtc_horizon, queue_threshold=args.rtc_queue_threshold)
            rtc = RTCSettings(mode=args.rtc_mode, horizon=args.rtc_horizon,
                              queue_threshold=args.rtc_queue_threshold, policy_hz=policy_hz,
                              training_max_delay=config["rtc_max_delay"])
        native = TorchPolicyAdapter(policy, preprocessor, postprocessor, args.device)
        sampler = TargetSampler(PolicyChunkSampler(native, args.task, rtc), args.action_mode, args.pose_contract)
        engine = (SyncActionEngine(sampler, policy.config.n_action_steps) if rtc is None
                  else AsyncRTCActionEngine(sampler, rtc))
        return EngineFacade(engine)

    if args.inference_mode == "sync":
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=features,
            ordered_action_keys=list(AXES),
            task=args.task,
            device=args.device,
            robot_type="piper_dual",
        )

    from lerobot.rollout.inference.rtc import RTCInferenceEngine, supports_rtc_inference
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from lerobot.utils.feature_utils import hw_to_dataset_features

    if not supports_rtc_inference(policy):
        raise ValueError("this policy/checkpoint does not support RTC inference")
    if args.rtc_mode == "trained" and int(policy.config.rtc_training_max_delay) <= 0:
        raise ValueError(
            "--rtc-mode=trained requires a checkpoint trained with "
            "rtc_training_max_delay > 0; use --rtc-mode=guided for this checkpoint"
        )
    chunk_size = int(getattr(policy.config, "chunk_size", 0))
    if chunk_size and args.rtc_queue_threshold >= chunk_size:
        raise ValueError(
            f"--rtc-queue-threshold must be below checkpoint chunk_size={chunk_size}"
        )
    if args.rtc_mode == "trained":
        max_delay = int(policy.config.rtc_training_max_delay)
        if args.rtc_horizon < max_delay or args.rtc_queue_threshold < max_delay:
            raise ValueError(
                "trained RTC requires both horizon and queue threshold to be at least "
                f"rtc_training_max_delay={max_delay}"
            )
        if chunk_size and args.rtc_horizon > chunk_size - max_delay:
            raise ValueError(
                "trained RTC requires horizon <= chunk_size - rtc_training_max_delay "
                f"({chunk_size - max_delay})"
            )
    feature_robot = PiperFeatureRobot()
    hardware_features = hw_to_dataset_features(
        feature_robot.observation_features, "observation"
    )
    return RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=ThreadSafeRobot(feature_robot),
        rtc_config=rtc_config,
        hw_features=hardware_features,
        task=args.task,
        fps=policy_hz,
        device=args.device,
        use_torch_compile=False,
        compile_warmup_inferences=0,
        rtc_queue_threshold=args.rtc_queue_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT,
        help="LeRobot pretrained_model directory (full model or LoRA checkpoint)",
    )
    parser.add_argument(
        "--action-mode",
        choices=ACTION_MODES,
        default=DEFAULT_ACTION_MODE,
        help="action representation used to train the checkpoint",
    )
    parser.add_argument(
        "--inference-mode",
        choices=("sync", "rtc"),
        default=DEFAULT_INFERENCE_MODE,
        help="synchronous chunk execution or asynchronous RTC inference",
    )
    parser.add_argument(
        "--rtc-mode",
        choices=("guided", "trained"),
        default=DEFAULT_RTC_MODE,
        help="RTC prefix mode; trained requires an RTC-trained checkpoint",
    )
    parser.add_argument(
        "--rtc-horizon",
        type=int,
        default=DEFAULT_RTC_HORIZON,
        help="number of queued actions used to connect adjacent RTC chunks",
    )
    parser.add_argument(
        "--rtc-queue-threshold",
        type=int,
        default=DEFAULT_RTC_QUEUE_THRESHOLD,
        help="start an asynchronous refill when queued actions reach this value",
    )
    parser.add_argument(
        "--bridge-endpoint",
        default=DEFAULT_BRIDGE_ENDPOINT,
        help="ZeroMQ endpoint of hardware_bridge.py",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="English task instruction",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="policy inference device: cpu, cuda, or one GPU such as cuda:0",
    )
    parser.add_argument(
        "--control-hz",
        dest="control_hz",
        type=float,
        default=DEFAULT_CONTROL_HZ,
        help="frequency at which each follower receives pyAgxArm.move_js targets",
    )
    parser.add_argument(
        "--hz",
        dest="control_hz",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--policy-hz",
        type=float,
        default=DEFAULT_POLICY_HZ,
        help="rate for consuming new policy actions",
    )
    parser.add_argument(
        "--smoothing-time-constant",
        type=float,
        default=DEFAULT_SMOOTHING_TIME_CONSTANT_S,
        help="absolute-target low-pass time constant in seconds; 0 disables filtering",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="number of emitted policy actions; 0 runs continuously",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="policy sampling seed",
    )
    parser.add_argument(
        "--rpc-timeout",
        type=float,
        default=DEFAULT_RPC_TIMEOUT_S,
        help="timeout for one bridge RPC in seconds",
    )
    parser.add_argument(
        "--watchdog-timeout",
        type=float,
        default=DEFAULT_WATCHDOG_TIMEOUT_S,
        help="bridge inactivity timeout before safe shutdown in seconds",
    )
    parser.add_argument(
        "--diagnostics-jsonl",
        type=Path,
        default=None,
        help="optional per-tick diagnostics output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run inference and validate targets without sending actions",
    )
    parser.add_argument("--calibration-mount-id", default="")
    parser.add_argument("--offline-dataset-root", type=Path)
    parser.add_argument("--pose-start-margin-deg", type=float, default=2.0, help="仅位姿部署：J2=+偏移，J3=-偏移，范围 0..5 度")
    args = parser.parse_args()
    if not 0 <= args.pose_start_margin_deg <= 5:
        parser.error("--pose-start-margin-deg must be in [0,5]")
    if args.offline_dataset_root is not None:
        args.dry_run = True

    checkpoint = resolve_path(args.checkpoint)
    if not (checkpoint / "config.json").is_file():
        parser.error(f"checkpoint is missing config.json: {checkpoint}")
    if not (checkpoint / "policy_preprocessor.json").is_file():
        parser.error(f"checkpoint is missing policy processors: {checkpoint}")
    if not (checkpoint / "policy_postprocessor.json").is_file():
        parser.error(f"checkpoint is missing policy postprocessor: {checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")

    sys.path.insert(0, str(ROOT / "src"))
    from easy_embodied_deploy.contracts import checkpoint_action_mode
    try:
        checkpoint_action_mode(checkpoint, args.action_mode)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    policy_hz = validate_args(parser, args)
    args.pose_contract = checkpoint_pose_contract(checkpoint, args.action_mode,
        mount_id=args.calibration_mount_id, policy_hz=policy_hz)

    rtc_config = None
    if args.inference_mode == "rtc":
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        rtc_config = RTCConfig(
            mode=args.rtc_mode,
            execution_horizon=args.rtc_horizon,
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    startup = time.perf_counter()
    loaded_at = time.perf_counter()
    policy, preprocessor, postprocessor, features = load_policy(
        checkpoint, args.device, rtc_config
    )
    inference_monitor = InferenceLatencyMonitor()
    inference_monitor.install(policy)
    print(
        f"Model/processor load: {time.perf_counter() - loaded_at:.2f}s | "
        f"startup: {time.perf_counter() - startup:.2f}s",
        flush=True,
    )

    engine = make_inference_engine(
        args=args,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        features=features,
        policy_hz=policy_hz,
        rtc_config=rtc_config,
    )

    from bridge_client import BridgeError, PiperBridge
    from deployer import (
        PiperActionAdapter,
        to_rtc_hardware_observation,
        validate_observation,
    )

    if args.offline_dataset_root is not None:
        from easy_embodied_deploy.offline_bridge import OfflineBridge
        root = resolve_path(args.offline_dataset_root)
        if (root / "lerobot").is_dir():
            root = root / "lerobot"
        bridge = OfflineBridge(root)
    else:
        bridge = PiperBridge(args.bridge_endpoint, timeout_s=args.rpc_timeout)
    bridge_stopped = False
    engine_started = False
    diagnostics_path = (
        None
        if args.diagnostics_jsonl is None
        else resolve_path(args.diagnostics_jsonl)
    )
    diagnostics = RuntimeDiagnostics(
        diagnostics_path, expected_period_s=1.0 / policy_hz
    )

    try:
        engine.start()
        engine_started = True
        if args.inference_mode == "rtc":
            engine.resume()
        bridge.ping()
        configured = bridge.configure(
            control_hz=args.control_hz,
            smoothing_time_constant_s=args.smoothing_time_constant,
            watchdog_timeout_s=args.watchdog_timeout,
            motion_space="pose" if args.pose_contract else "joint",
            pose_start_margin_deg=args.pose_start_margin_deg,
        )
        validate_live_cameras(args.pose_contract, configured)
        if configured.get("joint_units") != "rad":
            raise BridgeError("hardware bridge joint unit is not rad")
        if configured.get("gripper_units") != "m":
            raise BridgeError("hardware bridge gripper unit is not m")
        converter = PoseTargetAdapter() if args.pose_contract else PiperActionAdapter(
            "absolute_joint",
            joint_limits=np.asarray(configured.get("joint_limits"), dtype=np.float64),
            gripper_limits=tuple(configured.get("gripper_limits", ())),
        )
        print(
            f"Bridge connected | {'move_p' if args.action_mode in POSE_MODES else 'move_js'}={configured['control_hz']:g} Hz | "
            f"smoothing_tau={configured['smoothing_time_constant_s']:g}s | "
            f"policy={policy_hz:g} Hz | "
            f"mode={args.inference_mode}/{args.action_mode} | "
            f"seed={args.seed} | "
            f"max_steps={args.max_steps} | dry_run={args.dry_run}",
            flush=True,
        )
        applied_steps = 0
        observation_count = 0
        waiting_reported = False
        first_action_at: float | None = None
        deadline = time.perf_counter()
        previous_tick = deadline
        while args.max_steps == 0 or applied_steps < args.max_steps:
            tick_started = time.perf_counter()
            tick_s = tick_started - previous_tick
            previous_tick = tick_started
            observation_started = time.perf_counter()
            observation = validate_observation(bridge.observation())
            observation_s = time.perf_counter() - observation_started
            observation_count += 1
            state = observation["observation.state"]
            if observation_count == 1:
                converter.reset(state)
                print(
                    f"First bridge observation: {time.perf_counter() - startup:.2f}s "
                    f"after startup",
                    flush=True,
                )

            engine_started_at = time.perf_counter()
            if args.action_mode == "delta_joint":
                observation[COMMAND_KEY] = converter.command_target
            if isinstance(engine, EngineFacade):
                action = engine.get_action(observation)
            elif args.inference_mode == "rtc":
                engine.notify_observation(to_rtc_hardware_observation(observation))
                action = engine.get_action(None)
            else:
                action = engine.get_action(observation)
            engine_s = time.perf_counter() - engine_started_at
            action_queue = getattr(engine, "action_queue", None)
            queue_size = None if action_queue is None else action_queue.qsize()

            if getattr(engine, "failed", False):
                details = getattr(engine, "failure_traceback", None) or "unknown RTC failure"
                raise RuntimeError(f"RTC inference thread failed:\n{details}")

            if action is None:
                values = None
                target = None
                if not waiting_reported:
                    print("Waiting for the first RTC action; followers hold position...", flush=True)
                    waiting_reported = True
            else:
                values = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
                target = converter.to_absolute(values, state)
                if first_action_at is None:
                    first_action_at = time.perf_counter()
                    print(
                        f"First action ready: {first_action_at - startup:.2f}s after startup",
                        flush=True,
                    )
                if not args.dry_run:
                    (bridge.pose_action if args.pose_contract else bridge.action)(target)
                applied_steps += 1
                limited = " | hardware-limit-clipped" if converter.last_was_limited else ""
                print(
                    f"action {applied_steps}"
                    + (f"/{args.max_steps}" if args.max_steps else "")
                    + f": policy=[{values.min():.4f},{values.max():.4f}] "
                    + f"target=[{target.min():.4f},{target.max():.4f}]"
                    + limited,
                    flush=True,
                )

            diagnostics.record(
                tick_s=tick_s,
                observation_s=observation_s,
                engine_s=engine_s,
                queue_size=queue_size,
                action=values,
                target=target,
                limited=False if action is None else converter.last_was_limited,
            )
            if action is not None and applied_steps % 20 == 0:
                latencies = inference_monitor.snapshot()
                latest = float("nan") if not latencies else latencies[-1]
                print(
                    f"diagnostics: queue={queue_size} | "
                    f"starved={diagnostics.runtime_starved} | "
                    f"chunk_latency={latest:.3f}s",
                    flush=True,
                )

            deadline = max(deadline + 1.0 / policy_hz, time.perf_counter())
            time.sleep(max(0.0, deadline - time.perf_counter()))
    except KeyboardInterrupt:
        print("Stopping safely...", flush=True)
    finally:
        if engine_started:
            try:
                engine.stop()
            except Exception as exc:
                print(f"Inference-engine stop failed: {exc}", file=sys.stderr, flush=True)
        try:
            bridge_stats = bridge.status().get("command_loop", {})
            print(
                "Runtime diagnostics: "
                + json.dumps(
                    {
                        **diagnostics.summary(inference_monitor.snapshot()),
                        "move_js": bridge_stats,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except BridgeError as exc:
            print(f"Bridge status failed: {exc}", file=sys.stderr, flush=True)
        diagnostics.close()
        try:
            bridge.stop()
            bridge_stopped = True
        except BridgeError as exc:
            print(f"Bridge stop failed: {exc}", file=sys.stderr, flush=True)
        finally:
            if not bridge_stopped:
                bridge.close()


if __name__ == "__main__":
    main()
