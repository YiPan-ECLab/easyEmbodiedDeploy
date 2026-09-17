#!/usr/bin/env python3
"""Piper/Orbbec hardware RPC server (run in the piper-home environment)."""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import zmq
from pyAgxArm import ArmModel
from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/PiperRecorder"))
sys.path.insert(0, str(ROOT / "src"))
from easy_embodied_deploy.resources import hardware_lease

from piper_recorder.camera import open_cameras  # noqa: E402
from piper_recorder.robot import DualPiper, ZERO_JOINTS, verify_can_interface  # noqa: E402
from record import (  # noqa: E402
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_MAX_AGE_S,
    CAMERA_SERIALS,
    CAMERA_WIDTH,
    FIRMWARE,
    FOLLOW_HZ,
    GRIPPER_FORCE_N,
    GRIPPER_HOME_TIMEOUT_S,
    GRIPPER_HOME_TOLERANCE_M,
    GRIPPER_MAX_WIDTH_M,
    HOME_SPEED,
    HOME_TIMEOUT_S,
    HOME_TOLERANCE_DEG,
    IMAGE_SIZE,
    PIPERS,
    PREPOSITION_SPEED,
    PREPOSITION_TIMEOUT_S,
    STATUS_PERIOD_S,
    ZERO_TOLERANCE_DEG,
)


PROTOCOL_VERSION = 3
DEFAULT_ENDPOINT = "tcp://127.0.0.1:6007"
DEFAULT_CONTROL_HZ = 30.0
DEFAULT_SMOOTHING_TIME_CONSTANT_S = 0.1
DEFAULT_WATCHDOG_TIMEOUT_S = 30.0
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
JOINT_LIMITS = np.asarray(
    [ROBOT_JOINT_LIMIT_PRESET_RAD[ArmModel.PIPER][name] for name in JOINT_NAMES],
    dtype=np.float64,
)
GRIPPER_LIMITS = np.asarray((0.0, GRIPPER_MAX_WIDTH_M), dtype=np.float64)

if not np.allclose(GRIPPER_LIMITS, (0.0, 0.1), atol=1e-7):
    raise RuntimeError("this Piper deployment requires a 0.1 m gripper configuration")


def _parallel(pairs: list[Any], operation) -> None:
    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        futures = [pool.submit(operation, pair) for pair in pairs]
        for future in futures:
            future.result()


def connect_followers(robot: DualPiper) -> None:
    """Connect only the two follower CAN interfaces used during deployment."""
    pairs = list(robot.pairs.values())
    print("Follower CAN interface preflight...")
    for pair in pairs:
        verify_can_interface(pair.follower_name)

    def connect(pair) -> None:
        pair.follower, pair.follower_gripper = pair._create_arm(pair.follower_name)
        pair._wait_follower_feedback(timeout=3.0)
        pair.follower.set_follower_mode()
        print(f"{pair.side}: {pair.follower_name}(follower)")

    _parallel(pairs, connect)


def home_follower(pair: Any) -> None:
    """Home one deployment follower without connecting its leader arm."""
    pair._move_to(
        {pair.follower_name: pair.follower},
        ZERO_JOINTS,
        speed=pair.home_speed,
        timeout=pair.home_timeout_s,
        tolerance=pair.home_tolerance,
        hz=50,
    )
    print(f"{pair.follower_name}: ZERO 完成")
    pair._home_gripper(pair.follower_gripper, pair.follower_name)


def prepare_followers(robot: DualPiper) -> None:
    """Home and preposition only the two follower arms."""
    pairs = list(robot.pairs.values())
    _parallel(pairs, home_follower)

    def preposition(pair) -> None:
        pair.follower.set_follower_mode()
        pair._move_to(  # PiperRecorder's checked position-velocity move helper.
            {pair.follower_name: pair.follower},
            pair.target,
            speed=PREPOSITION_SPEED,
            timeout=PREPOSITION_TIMEOUT_S,
            tolerance=pair.zero_tolerance,
            hz=50,
        )
        pair._wait_enabled(pair.follower, pair.follower_name)

    _parallel(pairs, preposition)


def home_followers(robot: DualPiper) -> None:
    """Return followers to ZERO without moving the leader arms."""
    pairs = list(robot.pairs.values())
    _parallel(pairs, home_follower)


def prepare_pose_followers(robot, margin_deg):
    """Optional, explicit Cartesian start margin; reuse original move helper.

    At J2=J3=0 a rounded Cartesian target may place firmware IK outside a
    joint limit. This applies only to the new pose mode, never joint rollout.
    """
    if not math.isfinite(margin_deg) or not 0 <= margin_deg <= 5:
        raise ValueError("pose_start_margin_deg must be in [0,5]")
    if margin_deg == 0:
        return
    def prepare(pair):
        target = list(pair.target)
        target[1], target[2] = math.radians(margin_deg), -math.radians(margin_deg)
        pair._move_to({pair.follower_name: pair.follower}, tuple(target),
                      speed=PREPOSITION_SPEED, timeout=PREPOSITION_TIMEOUT_S,
                      tolerance=pair.zero_tolerance, hz=50)
    _parallel(list(robot.pairs.values()), prepare)


def read_state(robot: DualPiper, *, include_pose=False):
    samples = robot.sample()
    state: list[float] = []
    for side in ("left", "right"):
        state.extend((*samples[side].joints, samples[side].gripper_width))
    result = np.asarray(state, dtype=np.float32)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise RuntimeError("Piper returned an invalid 14-dimensional state")
    if include_pose:
        # sample() already obtains flange feedback for the recorder. Reuse
        # those same samples; no additional SDK read or hardware mode change.
        poses = np.asarray([v for side in ("left", "right") for v in samples[side].flange_pose], dtype=np.float64)
        if poses.shape != (12,) or not np.isfinite(poses).all():
            raise RuntimeError("Piper returned an invalid flange pose")
        return result, poses
    return result


def low_pass_absolute_target(
    previous: np.ndarray,
    desired: np.ndarray,
    *,
    control_hz: float,
    time_constant_s: float,
) -> np.ndarray:
    """Advance one filter tick between two absolute hardware targets.

    Both delta and absolute policies reach the bridge as absolute targets, so
    this filter is deliberately independent of the policy action convention.
    """
    if time_constant_s == 0:
        return desired.copy()
    alpha = -math.expm1(-1.0 / (control_hz * time_constant_s))
    return previous + alpha * (desired - previous)


class FollowerCommandLoop:
    """Low-pass absolute targets and send them through move_js at control_hz.

    The policy may emit joint deltas, but the inference process integrates them
    before this boundary.  Filtering absolute targets here preserves total
    displacement; filtering and then accumulating raw deltas would change it.
    """

    def __init__(self, robot: DualPiper, initial_target: np.ndarray) -> None:
        self.robot = robot
        self._target = np.asarray(initial_target, dtype=np.float64).copy()
        self._filtered_target = self._target.copy()
        self._control_hz = DEFAULT_CONTROL_HZ
        self._smoothing_time_constant_s = DEFAULT_SMOOTHING_TIME_CONSTANT_S
        self._armed = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._cycles_sent = 0
        self._send_started_at: float | None = None
        self._last_send_at: float | None = None
        self._max_command_step = np.zeros(14, dtype=np.float64)
        self._feedback_samples = 0
        self._last_actual = {
            "left": (None, None),
            "right": (None, None),
        }
        self._last_velocity = {
            "left": np.zeros(6, dtype=np.float64),
            "right": np.zeros(6, dtype=np.float64),
        }
        self._max_tracking_error = np.zeros(12, dtype=np.float64)
        self._max_actual_step = np.zeros(12, dtype=np.float64)
        self._max_actual_velocity = np.zeros(12, dtype=np.float64)
        self._velocity_sign_reversals = np.zeros(12, dtype=np.int64)
        self._thread = threading.Thread(
            target=self._run, name="piper-deploy-command", daemon=True
        )
        self._thread.start()

    def configure(
        self, *, control_hz: float, smoothing_time_constant_s: float
    ) -> None:
        if not 1 <= control_hz <= 200:
            raise ValueError("control_hz must be in [1, 200]")
        if not 0 <= smoothing_time_constant_s <= 2:
            raise ValueError("smoothing_time_constant_s must be in [0, 2]")
        with self._lock:
            self._control_hz = float(control_hz)
            self._smoothing_time_constant_s = float(smoothing_time_constant_s)

    def set_target(self, value: Any) -> None:
        target = np.asarray(value, dtype=np.float64).reshape(-1)
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError("target must contain 14 finite values")
        for offset in (0, 7):
            joints = target[offset : offset + 6]
            if np.any(joints < JOINT_LIMITS[:, 0]) or np.any(
                joints > JOINT_LIMITS[:, 1]
            ):
                raise ValueError(f"{('left', 'right')[offset == 7]} target exceeds joint limits")
        if np.any(target[[6, 13]] < GRIPPER_LIMITS[0]) or np.any(
            target[[6, 13]] > GRIPPER_LIMITS[1]
        ):
            raise ValueError("gripper target exceeds [0, 0.1] m")

        with self._lock:
            self._target = target.copy()
            self._armed = True

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Piper command loop stopped: {self._error}") from self._error

    def stats(self) -> dict[str, Any]:
        with self._lock:
            elapsed = (
                0.0
                if self._send_started_at is None or self._last_send_at is None
                else self._last_send_at - self._send_started_at
            )
            achieved_hz = (
                0.0
                if elapsed <= 0 or self._cycles_sent < 2
                else (self._cycles_sent - 1) / elapsed
            )
            return {
                "cycles_sent": self._cycles_sent,
                "achieved_control_hz": achieved_hz,
                "max_command_step": self._max_command_step.tolist(),
                "feedback_samples": self._feedback_samples,
                "max_tracking_error_rad": self._max_tracking_error.tolist(),
                "max_actual_step_rad": self._max_actual_step.tolist(),
                "max_actual_velocity_rad_s": self._max_actual_velocity.tolist(),
                "velocity_sign_reversals": self._velocity_sign_reversals.tolist(),
                "filtered_target": self._filtered_target.tolist(),
                "desired_target": self._target.tolist(),
            }

    def _record_feedback(self, target: np.ndarray) -> None:
        for side, command_offset, stats_offset in (
            ("left", 0, 0),
            ("right", 7, 6),
        ):
            message = self.robot.pairs[side].follower.get_joint_angles()
            if message is None:
                continue
            timestamp = float(message.timestamp)
            actual = np.asarray(message.msg, dtype=np.float64)
            if actual.shape != (6,) or not np.isfinite(actual).all():
                continue
            previous, previous_timestamp = self._last_actual[side]
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                continue
            section = slice(stats_offset, stats_offset + 6)
            with self._lock:
                self._feedback_samples += 1
                self._max_tracking_error[section] = np.maximum(
                    self._max_tracking_error[section],
                    np.abs(actual - target[command_offset : command_offset + 6]),
                )
                if previous is not None and previous_timestamp is not None:
                    step = actual - previous
                    dt = timestamp - previous_timestamp
                    velocity = step / dt
                    old_velocity = self._last_velocity[side]
                    moving = (np.abs(velocity) > 0.01) & (np.abs(old_velocity) > 0.01)
                    self._velocity_sign_reversals[section] += (
                        moving & (velocity * old_velocity < 0)
                    )
                    self._max_actual_step[section] = np.maximum(
                        self._max_actual_step[section], np.abs(step)
                    )
                    self._max_actual_velocity[section] = np.maximum(
                        self._max_actual_velocity[section], np.abs(velocity)
                    )
                    self._last_velocity[side] = velocity
                self._last_actual[side] = (actual, timestamp)

    def _run(self) -> None:
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                with self._lock:
                    desired = self._target.copy()
                    previous = self._filtered_target.copy()
                    control_hz = self._control_hz
                    tau = self._smoothing_time_constant_s
                    armed = self._armed
                if armed:
                    target = low_pass_absolute_target(
                        previous,
                        desired,
                        control_hz=control_hz,
                        time_constant_s=tau,
                    )
                    now = time.monotonic()
                    for side, offset in (("left", 0), ("right", 7)):
                        pair = self.robot.pairs[side]
                        pair.follower.move_js(target[offset : offset + 6].tolist())
                        pair.follower_gripper.move_gripper_m(
                            float(target[offset + 6]), force=pair.gripper_force
                        )
                    self._record_feedback(target)
                    with self._lock:
                        self._filtered_target = target
                        self._max_command_step = np.maximum(
                            self._max_command_step, np.abs(target - previous)
                        )
                        self._cycles_sent += 1
                        if self._send_started_at is None:
                            self._send_started_at = now
                        self._last_send_at = now
                period = 1.0 / control_hz
                deadline = max(deadline + period, time.monotonic())
                self._stop.wait(max(0.0, deadline - time.monotonic()))
        except BaseException as exc:  # relay thread failures to the RPC loop
            if not self._stop.is_set():
                self._error = exc

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("Piper command loop did not stop")


class PoseCommandLoop(FollowerCommandLoop):
    """New Cartesian path through SDK move_p; original joint loop is unchanged."""

    def set_target(self, value):
        from easy_embodied_deploy.target_runtime import PoseTargetAdapter
        raw = np.asarray(value, dtype=np.float64)
        target = PoseTargetAdapter().to_absolute(raw, np.zeros(14))
        if np.any(raw[[6, 13]] != target[[6, 13]]):
            raise ValueError("gripper target exceeds [0, 0.1] m")
        for offset in (0, 7):
            if abs(target[offset+4]) > np.pi/2 + 1e-9 or np.any(np.abs(target[[offset+3, offset+5]]) > np.pi + 1e-9):
                raise ValueError("move_p RPY must use principal Euler angles")
        with self._lock:
            self._target = target.copy()
            self._armed = True

    def stats(self):
        result = super().stats()
        result["motion_space"] = "pose"
        result["command_units"] = "m,m,m,rad,rad,rad,m per arm"
        for key in ("max_tracking_error_rad", "max_actual_step_rad", "max_actual_velocity_rad_s", "velocity_sign_reversals"):
            result.pop(key, None)
        return result

    def _run(self):
        from easy_embodied_deploy.pose_actions import matrix_rotvec, matrix_rpy, rotvec_matrix, rpy_matrix
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                with self._lock:
                    desired, previous = self._target.copy(), self._filtered_target.copy()
                    hz, tau, armed = self._control_hz, self._smoothing_time_constant_s, self._armed
                if armed:
                    alpha = 1.0 if tau == 0 else -math.expm1(-1.0/(hz*tau))
                    target = previous + alpha*(desired-previous)
                    for offset in (0, 7):
                        current, future = rpy_matrix(previous[offset+3:offset+6]), rpy_matrix(desired[offset+3:offset+6])
                        target[offset+3:offset+6] = matrix_rpy(rotvec_matrix(alpha*matrix_rotvec(future @ current.T)) @ current)
                    now = time.monotonic()
                    for side, offset in (("left", 0), ("right", 7)):
                        pair = self.robot.pairs[side]
                        pair.follower.move_p(target[offset:offset+6].tolist())
                        pair.follower_gripper.move_gripper_m(float(target[offset+6]), force=pair.gripper_force)
                    with self._lock:
                        self._filtered_target = target
                        self._cycles_sent += 1
                        self._max_command_step = np.maximum(self._max_command_step, np.abs(target-previous))
                        if self._send_started_at is None:
                            self._send_started_at = now
                        self._last_send_at = now
                deadline = max(deadline+1/hz, time.monotonic())
                self._stop.wait(max(0.0, deadline-time.monotonic()))
        except BaseException as exc:
            if not self._stop.is_set():
                self._error = exc


def make_reply(request: dict[str, Any], *, ok: bool, **payload: Any) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": request.get("request_id"),
        "ok": ok,
        **payload,
    }


def serve(endpoint: str) -> None:
    with hardware_lease():
        _serve(endpoint)


def _serve(endpoint: str) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 200)
    ready = False
    command_loop: FollowerCommandLoop | None = None
    watchdog_timeout_s = DEFAULT_WATCHDOG_TIMEOUT_S
    last_contact: float | None = None
    sequence = 0
    motion_space = "joint"
    motion_configured = False

    try:
        socket.bind(endpoint)
        with ExitStack() as stack:
            cameras = open_cameras(
                CAMERA_SERIALS,
                stack,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS,
                image_size=IMAGE_SIZE,
                max_age_s=CAMERA_MAX_AGE_S,
            )
            robot = DualPiper(
                PIPERS,
                firmware=FIRMWARE,
                home_speed=HOME_SPEED,
                home_timeout_s=HOME_TIMEOUT_S,
                home_tolerance_deg=HOME_TOLERANCE_DEG,
                preposition_speed=PREPOSITION_SPEED,
                preposition_timeout_s=PREPOSITION_TIMEOUT_S,
                zero_tolerance_deg=ZERO_TOLERANCE_DEG,
                follow_hz=FOLLOW_HZ,
                status_period_s=STATUS_PERIOD_S,
                gripper_force=GRIPPER_FORCE_N,
                gripper_max_width=GRIPPER_MAX_WIDTH_M,
                gripper_home_timeout_s=GRIPPER_HOME_TIMEOUT_S,
                gripper_home_tolerance_m=GRIPPER_HOME_TOLERANCE_M,
            )
            stack.callback(robot.close)
            connect_followers(robot)
            prepare_followers(robot)
            initial_state = read_state(robot)
            command_loop = FollowerCommandLoop(robot, initial_state)

            def shutdown_hardware() -> None:
                nonlocal command_loop
                if command_loop is not None:
                    command_loop.stop()
                    command_loop = None
                print("Returning followers and grippers to ZERO...", flush=True)
                if motion_space == "pose":
                    # Begin the same joint homing target before its existing
                    # status checker runs. A rejected move_p status otherwise
                    # survives until the next CAN feedback frame.
                    for pair in robot.pairs.values():
                        pair.follower.set_speed_percent(pair.home_speed)
                        pair.follower.move_j(list(ZERO_JOINTS))
                    time.sleep(.1)
                home_followers(robot)

            # Registered after robot.close(), so homing runs before CAN disconnect.
            stack.callback(shutdown_hardware)
            ready = True
            print(
                f"hardware bridge ready: {endpoint} | awaiting deployer motion config "
                f"| joints=rad | grippers=[0, 0.1] m | followers only",
                flush=True,
            )

            stopping = False
            while not stopping:
                command_loop.raise_if_failed()
                try:
                    request = socket.recv_pyobj()
                except zmq.Again:
                    if (
                        last_contact is not None
                        and time.monotonic() - last_contact > watchdog_timeout_s
                    ):
                        raise TimeoutError(
                            f"deployment heartbeat absent for {watchdog_timeout_s:g}s"
                        )
                    continue

                if not isinstance(request, dict):
                    socket.send_pyobj(
                        make_reply({}, ok=False, error="request must be a mapping")
                    )
                    continue
                if request.get("protocol") != PROTOCOL_VERSION:
                    socket.send_pyobj(
                        make_reply(
                            request,
                            ok=False,
                            error=(
                                f"protocol mismatch: expected {PROTOCOL_VERSION}, "
                                f"got {request.get('protocol')!r}"
                            ),
                        )
                    )
                    continue

                last_contact = time.monotonic()
                operation = request.get("operation")
                try:
                    if operation == "ping":
                        payload = {"ready": True}
                    elif operation == "configure":
                        control_hz = float(request["control_hz"])
                        smoothing_tau = float(request["smoothing_time_constant_s"])
                        watchdog = float(request["watchdog_timeout_s"])
                        if not 1 <= control_hz <= 200:
                            raise ValueError("control_hz must be in [1, 200]")
                        if not 0 <= smoothing_tau <= 2:
                            raise ValueError("smoothing_time_constant_s must be in [0, 2]")
                        if not math.isfinite(watchdog) or watchdog < 5:
                            raise ValueError("watchdog_timeout_s must be at least 5 seconds")
                        selected_space = request.get("motion_space", "joint")
                        if selected_space not in ("joint", "pose"):
                            raise ValueError("motion_space must be joint or pose")
                        pose_margin = float(request.get("pose_start_margin_deg", 2.0))
                        if selected_space == "pose" and not 0 <= pose_margin <= 5:
                            raise ValueError("pose_start_margin_deg must be in [0,5]")
                        if selected_space != motion_space:
                            if motion_configured:
                                raise ValueError("restart bridge before changing motion space")
                            command_loop.stop()
                            prepare_pose_followers(robot, pose_margin)
                            state, poses = read_state(robot, include_pose=True)
                            initial = np.r_[poses[:6], state[6], poses[6:], state[13]]
                            command_loop = PoseCommandLoop(robot, initial)
                            motion_space = selected_space
                        command_loop.configure(
                            control_hz=control_hz,
                            smoothing_time_constant_s=smoothing_tau,
                        )
                        watchdog_timeout_s = watchdog
                        motion_configured = True
                        payload = {
                            "motion_space": motion_space,
                            "pose_start_margin_deg": float(request.get("pose_start_margin_deg", 2.0)) if motion_space == "pose" else 0.0,
                            "pose_units": "m,m,m,rad,rad,rad",
                            "cameras": {name: {"serial": camera.serial, "sensor": camera._sensor_type.name,
                                               "width": camera.width, "height": camera.height}
                                        for name, camera in cameras.items()},
                            "control_hz": control_hz,
                            "smoothing_time_constant_s": smoothing_tau,
                            "watchdog_timeout_s": watchdog_timeout_s,
                            "joint_limits": JOINT_LIMITS.tolist(),
                            "joint_units": "rad",
                            "gripper_limits": GRIPPER_LIMITS.tolist(),
                            "gripper_units": "m",
                        }
                    elif operation == "observe":
                        if motion_space == "pose":
                            state, poses = read_state(robot, include_pose=True)
                        else:
                            state = read_state(robot)
                        observation: dict[str, Any] = {
                            "observation.state": state
                        }
                        if motion_space == "pose":
                            observation["observation.flange_pose"] = poses
                        for name, camera in cameras.items():
                            observation[f"observation.images.{name}"] = camera.read().rgb
                        sequence += 1
                        # Piggyback already accumulated counters on the existing
                        # reply. No extra SDK reads or telemetry heartbeat RPCs.
                        payload = {"observation": observation, "sequence": sequence,
                                   "command_loop": command_loop.stats()}
                    elif operation in ("act", "act_pose"):
                        if (operation == "act_pose") != (motion_space == "pose"):
                            raise ValueError("action RPC does not match configured motion_space")
                        command_loop.set_target(request.get("target"))
                        payload = {"accepted": True}
                    elif operation == "status":
                        payload = {"command_loop": command_loop.stats()}
                    elif operation == "stop":
                        payload = {"stopping": True}
                        stopping = True
                    else:
                        raise ValueError(f"unknown operation: {operation!r}")
                    socket.send_pyobj(make_reply(request, ok=True, **payload))
                except Exception as exc:
                    socket.send_pyobj(make_reply(request, ok=False, error=str(exc)))

    finally:
        if command_loop is not None:
            try:
                command_loop.stop()
            except Exception as exc:
                print(f"command-loop shutdown failed: {exc}", file=sys.stderr, flush=True)
        if ready:
            print("hardware bridge stopped", flush=True)
        socket.close(linger=0)
        context.term()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()
    serve(args.endpoint)


if __name__ == "__main__":
    main()
