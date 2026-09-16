#!/usr/bin/env python3
"""Deploy an OpenPI JAX checkpoint through the current dual-Piper bridge."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[1]
LEROBOT_DEPLOYER_DIR = ROOT / "scripts/LerobotDeployer"
OPENPI_TRAINER_DIR = ROOT / "scripts/OpenpiTrainer"
DEFAULT_CHECKPOINT = ROOT / "models/finetuned_models/openpi_pi05_trainer_run"
DEFAULT_DATASET = ROOT / (
    "datasets/lerobot/"
    "Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta/lerobot"
)
DEFAULT_TASK = "Remove the shipping label and place it on the package on the table"
OFFLINE_JOINT_LIMITS = np.asarray(
    [
        (-2.617994, 2.617994),
        (0.0, 3.141593),
        (-2.96706, 0.0),
        (-1.74533, 1.74533),
        (-1.22173, 1.22173),
        (-2.094396, 2.094396),
    ],
    dtype=np.float64,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--action-mode", choices=("delta_joint", "absolute_joint"), default="delta_joint")
    parser.add_argument("--inference-mode", choices=("sync", "rtc"), default="rtc")
    parser.add_argument("--rtc-mode", choices=("guided", "trained"), default="guided")
    parser.add_argument("--rtc-horizon", type=int, default=20)
    parser.add_argument("--rtc-queue-threshold", type=int, default=40)
    parser.add_argument("--rtc-schedule", choices=("exp", "linear", "ones", "zeros"), default="exp")
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=5.0)
    parser.add_argument(
        "--rtc-mask-grippers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exclude the two gripper dimensions from RTC continuity guidance",
    )
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=0, help="0 uses the training manifest")
    parser.add_argument("--bridge-endpoint", default="tcp://127.0.0.1:6007")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda:1", help="cpu, cuda, or a physical GPU such as cuda:1")
    parser.add_argument("--policy-hz", type=float, default=10.0)
    parser.add_argument("--control-hz", "--hz", dest="control_hz", type=float, default=30.0)
    parser.add_argument("--smoothing-time-constant", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=30, help="0 runs until interrupted")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--rpc-timeout", type=float, default=5.0)
    parser.add_argument("--watchdog-timeout", type=float, default=30.0)
    parser.add_argument("--diagnostics-jsonl", type=Path)
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile plain and selected RTC samplers before connecting hardware",
    )
    parser.add_argument("--dry-run", action="store_true", help="infer and validate targets without sending them")
    parser.add_argument(
        "--offline-dataset-root",
        type=Path,
        help="decode observations from a LeRobot v3 dataset and do not connect to hardware",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.task.strip():
        parser.error("--task cannot be empty")
    if not 0 < args.policy_hz <= 200:
        parser.error("--policy-hz must be in (0, 200]")
    if not 0 < args.control_hz <= 200:
        parser.error("--control-hz must be in (0, 200]")
    if not 0 <= args.smoothing_time_constant <= 2:
        parser.error("--smoothing-time-constant must be in [0, 2]")
    if args.max_steps < 0 or args.seed < 0:
        parser.error("--max-steps and --seed cannot be negative")
    if args.rtc_horizon < 1 or args.rtc_horizon > 50:
        parser.error("--rtc-horizon must be in [1, 50]")
    if args.rtc_queue_threshold < 0 or args.rtc_queue_threshold >= 50:
        parser.error("--rtc-queue-threshold must be in [0, 49]")
    if args.sample_steps < 1:
        parser.error("--sample-steps must be positive")
    if args.n_action_steps < 0 or args.n_action_steps > 50:
        parser.error("--n-action-steps must be in [0, 50]")
    if args.rtc_max_guidance_weight < 0 or not np.isfinite(args.rtc_max_guidance_weight):
        parser.error("--rtc-max-guidance-weight must be finite and nonnegative")
    if (not np.isfinite(args.rpc_timeout) or not np.isfinite(args.watchdog_timeout)
            or args.rpc_timeout <= 0 or args.watchdog_timeout < 5):
        parser.error("RPC timeout must be positive and watchdog timeout at least 5 seconds")
    match = re.fullmatch(r"cuda(?::(\d+))?", args.device)
    if args.device != "cpu" and match is None:
        parser.error("--device must be cpu, cuda, or cuda:N")
    if args.device.startswith("cuda"):
        index = int(match.group(1) or 0)
        try:
            detected = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            parser.error(f"cannot verify CUDA devices: {exc}")
        available = {int(line) for line in detected.stdout.splitlines() if line.strip().isdigit()}
        if index not in available:
            parser.error(f"physical GPU {index} is unavailable; detected {sorted(available)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"


def _offline_observations(root: Path) -> Iterator[dict[str, Any]]:
    if str(OPENPI_TRAINER_DIR) not in sys.path:
        sys.path.insert(0, str(OPENPI_TRAINER_DIR))
    from dataset_v3 import IMAGE_KEYS, LeRobotV3Dataset

    dataset = LeRobotV3Dataset(root, action_horizon=50)
    index = 0
    while True:
        sample = dataset[index % len(dataset)]
        observation: dict[str, Any] = {
            "observation.state": np.asarray(sample["observation.state"], dtype=np.float32)
        }
        for key in IMAGE_KEYS:
            image = np.asarray(sample[key])
            if image.shape == (3, 224, 224):
                image = np.moveaxis(image, 0, -1)
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            observation[key] = image
        yield observation
        index += 1


class Diagnostics:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.stream = None
        self.ticks = 0
        self.actions = 0
        self.prefill_starved = 0
        self.runtime_starved = 0
        self.limited = 0
        self.minimum_queue: int | None = None
        self.max_target_step = 0.0
        self.previous_target: np.ndarray | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = path.open("w", encoding="utf-8")

    def record(self, **record: Any) -> None:
        self.ticks += 1
        action = record.get("action")
        target = record.get("target")
        if action is None:
            if self.actions:
                self.runtime_starved += 1
            else:
                self.prefill_starved += 1
        else:
            self.actions += 1
        self.limited += int(bool(record.get("limited")))
        queue = record.get("queue_size")
        if queue is not None:
            self.minimum_queue = int(queue) if self.minimum_queue is None else min(self.minimum_queue, int(queue))
        if target is not None and self.previous_target is not None:
            self.max_target_step = max(
                self.max_target_step,
                float(np.linalg.norm(np.asarray(target) - self.previous_target)),
            )
        if target is not None:
            self.previous_target = np.asarray(target).copy()
        if self.stream is not None:
            serializable = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in record.items()
            }
            self.stream.write(json.dumps({"tick": self.ticks, "wall_time_s": time.time(),
                                          **serializable, "counters": self.summary()}) + "\n")
            self.stream.flush()

    def save_summary(self, summary: dict[str, Any]) -> None:
        if self.path is not None:
            path = self.path.with_suffix(".summary.json")
            temporary = path.with_suffix(".summary.json.tmp")
            temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            temporary.replace(path)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def summary(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "actions": self.actions,
            "prefill_starved_ticks": self.prefill_starved,
            "runtime_starved_ticks": self.runtime_starved,
            # Keep the aggregate key for existing consumers.
            "starved_ticks": self.prefill_starved + self.runtime_starved,
            "hardware_limit_clips": self.limited,
            "minimum_queue": self.minimum_queue,
            "max_absolute_target_step_l2": self.max_target_step,
        }


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    from runtime import (
        AsyncRTCActionEngine,
        OpenpiDeploymentError,
        PolicyChunkSampler,
        RTCSettings,
        SyncActionEngine,
        latency_summary,
        load_policy,
        load_training_manifest,
        resolve_checkpoint,
        validate_manifest,
    )

    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
        manifest_path, manifest = load_training_manifest(checkpoint)
        manifest_config = validate_manifest(
            manifest,
            action_mode=args.action_mode,
            inference_mode=args.inference_mode,
            rtc_mode=args.rtc_mode,
            rtc_horizon=args.rtc_horizon,
            queue_threshold=args.rtc_queue_threshold,
        )
    except OpenpiDeploymentError as exc:
        parser.error(str(exc))
    n_action_steps = args.n_action_steps or int(manifest_config.get("n_action_steps", 10))
    if not 1 <= n_action_steps <= 50:
        parser.error("checkpoint n_action_steps must be in [1, 50]")

    np.random.seed(args.seed)
    os.environ["OPENPI_RTC_SAMPLE_MODE"] = "train_time" if args.rtc_mode == "trained" else "guidance"
    os.environ["OPENPI_TT_RTC_MASK_GRIPPER"] = "1" if args.rtc_mask_grippers else "0"
    os.environ["OPENPI_GRIPPER_INDICES"] = "6,13"
    os.environ["OPENPI_DATA_HOME"] = str(ROOT / "models/base_models")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
    print(
        "OPENPI_DEPLOYER_PLAN="
        + json.dumps(
            {
                "checkpoint": str(checkpoint),
                "manifest": str(manifest_path),
                "device": args.device,
                "inference_mode": args.inference_mode,
                "rtc_mode": args.rtc_mode,
                "offline": args.offline_dataset_root is not None,
                "dry_run": args.dry_run,
                "max_steps": args.max_steps,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    loaded_at = time.perf_counter()
    policy = load_policy(checkpoint, manifest_config, task=args.task, num_steps=args.sample_steps)
    rtc = None
    if args.inference_mode == "rtc":
        rtc = RTCSettings(
            mode=args.rtc_mode,
            horizon=args.rtc_horizon,
            queue_threshold=args.rtc_queue_threshold,
            schedule=args.rtc_schedule,
            max_guidance_weight=args.rtc_max_guidance_weight,
            mask_grippers=args.rtc_mask_grippers,
            policy_hz=args.policy_hz,
            training_max_delay=int(manifest_config.get("rtc_max_delay", 0) or 0),
        )
    sampler = PolicyChunkSampler(policy, args.task, rtc)
    if args.warmup:
        warmup_at = time.perf_counter()
        print("Compiling OpenPI sampling paths before hardware connection...", flush=True)
        sampler.warmup()
        print(f"OpenPI sampling warmup finished in {time.perf_counter() - warmup_at:.2f}s", flush=True)
    engine = (
        SyncActionEngine(sampler, n_action_steps)
        if rtc is None
        else AsyncRTCActionEngine(sampler, rtc)
    )
    print(f"OpenPI JAX policy loaded in {time.perf_counter() - loaded_at:.2f}s", flush=True)

    if str(LEROBOT_DEPLOYER_DIR) not in sys.path:
        sys.path.insert(0, str(LEROBOT_DEPLOYER_DIR))
    from deployer import PiperActionAdapter, validate_observation

    bridge = None
    diagnostics_path = None if args.diagnostics_jsonl is None else args.diagnostics_jsonl.expanduser().resolve()
    diagnostics = Diagnostics(diagnostics_path)
    stopped = False
    try:
        if args.offline_dataset_root is None:
            from bridge_client import BridgeError, PiperBridge

            bridge = PiperBridge(args.bridge_endpoint, timeout_s=args.rpc_timeout)
            bridge.ping()
            configured = bridge.configure(
                control_hz=args.control_hz,
                smoothing_time_constant_s=args.smoothing_time_constant,
                watchdog_timeout_s=args.watchdog_timeout,
            )
            if configured.get("joint_units") != "rad" or configured.get("gripper_units") != "m":
                raise BridgeError("hardware bridge must expose joint radians and gripper meters")
            joint_limits = np.asarray(configured["joint_limits"], dtype=np.float64)
            gripper_limits = tuple(configured["gripper_limits"])
            observations = None
        else:
            root = args.offline_dataset_root.expanduser().resolve()
            if not (root / "meta/info.json").is_file() and (root / "lerobot").is_dir():
                root = root / "lerobot"
            observations = _offline_observations(root)
            joint_limits = OFFLINE_JOINT_LIMITS
            gripper_limits = (0.0, 0.1)
        converter = PiperActionAdapter(args.action_mode, joint_limits, gripper_limits)

        # Prime the asynchronous queue before the first control tick.  JAX
        # compilation and the first inference are expected to take longer
        # than one policy period; counting those ticks as runtime starvation
        # both obscures the real health of RTC and needlessly delays startup.
        pending_observation: dict[str, Any] | None = None
        if isinstance(engine, AsyncRTCActionEngine):
            raw_observation = bridge.observation() if bridge is not None else next(observations)
            pending_observation = validate_observation(raw_observation)
            converter.reset(pending_observation["observation.state"])
            print("Prefilling OpenPI RTC action queue...", flush=True)
            if not engine.prefill(
                pending_observation,
                timeout_s=max(120.0, args.watchdog_timeout),
                heartbeat=None if bridge is None else bridge.ping,
            ):
                if engine.failed:
                    raise RuntimeError(f"OpenPI RTC prefill failed: {engine.failure}")
                raise RuntimeError("OpenPI RTC prefill timed out before the first action chunk")
            print(f"OpenPI RTC queue ready ({engine.queue_size} actions)", flush=True)

        applied = 0
        tick = 0
        deadline = time.perf_counter()
        while args.max_steps == 0 or applied < args.max_steps:
            tick += 1
            tick_started = time.perf_counter()
            observation_started = time.perf_counter()
            if pending_observation is None:
                raw_observation = bridge.observation() if bridge is not None else next(observations)
            else:
                raw_observation = pending_observation
                pending_observation = None
            observation = validate_observation(raw_observation)
            observation_s = time.perf_counter() - observation_started
            state = observation["observation.state"]
            if tick == 1:
                converter.reset(state)
            engine_started = time.perf_counter()
            action = engine.get_action(observation)
            engine_s = time.perf_counter() - engine_started
            if engine.failed:
                raise RuntimeError(f"OpenPI inference worker failed: {engine.failure}")
            target = None
            limited = False
            if action is not None:
                target = converter.to_absolute(action, state)
                limited = converter.last_was_limited
                if bridge is not None and not args.dry_run:
                    bridge.action(target)
                applied += 1
                print(
                    f"action {applied}{('/' + str(args.max_steps)) if args.max_steps else ''} "
                    f"queue={engine.queue_size} limited={limited}",
                    flush=True,
                )
            diagnostics.record(
                tick_s=time.perf_counter() - tick_started,
                observation_s=observation_s,
                engine_s=engine_s,
                queue_size=engine.queue_size,
                action=action,
                target=target,
                limited=limited,
                inference=latency_summary(list(sampler.latencies)),
                move_js={} if bridge is None else bridge.command_loop_stats,
            )
            deadline = max(deadline + 1.0 / args.policy_hz, time.perf_counter())
            time.sleep(max(0.0, deadline - time.perf_counter()))
    except KeyboardInterrupt:
        print("Interrupted; stopping policy and returning Piper followers to zero...", flush=True)
    finally:
        engine.close()
        bridge_stats: dict[str, Any] = {}
        if bridge is not None:
            try:
                bridge_stats = bridge.status().get("command_loop", {})
            except Exception as exc:
                print(f"Bridge status failed: {exc}", file=sys.stderr, flush=True)
        if bridge is not None and not stopped:
            try:
                bridge.stop()
                stopped = True
            except Exception as exc:
                print(f"Bridge stop/home failed: {exc}", file=sys.stderr, flush=True)
        diagnostics.close()
        summary = {
            **diagnostics.summary(),
            **latency_summary(list(sampler.latencies)),
            "rtc_enqueue_events": list(getattr(engine, "enqueue_events", ())),
            "move_js": bridge_stats,
        }
        try:
            diagnostics.save_summary(summary)
        except OSError as exc:
            print(f"Saving deployment summary failed: {exc}", file=sys.stderr, flush=True)
        print("OPENPI_DEPLOYER_SUMMARY=" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
