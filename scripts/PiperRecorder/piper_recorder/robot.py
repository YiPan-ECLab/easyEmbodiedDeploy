"""Dual leader-follower Piper control through pyAgxArm."""

import math
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

SIDES = ("left", "right")
ZERO_JOINTS = (0.0,) * 6
JOINT_LIMITS_RAD = (
    (-2.617994, 2.617994),
    (0.0, 3.141593),
    (-2.967060, 0.0),
    (-1.745330, 1.745330),
    (-1.221730, 1.221730),
    (-2.094396, 2.094396),
)


@dataclass(frozen=True)
class ArmCommand:
    joints: tuple[float, ...]
    gripper_width: float
    timestamp: float


@dataclass(frozen=True)
class ArmSample:
    joints: tuple[float, ...]
    flange_pose: tuple[float, ...]
    gripper_width: float
    gripper_force: float
    action: ArmCommand


def ticks(hz: float, stop: threading.Event | None = None) -> Iterator[float]:
    period = 1.0 / hz
    deadline = time.monotonic()
    while stop is None or not stop.is_set():
        yield time.monotonic()
        deadline = max(deadline + period, time.monotonic())
        delay = max(0.0, deadline - time.monotonic())
        if stop is None:
            time.sleep(delay)
        else:
            stop.wait(delay)


def verify_can_interface(name: str) -> None:
    result = subprocess.run(
        ["ip", "-details", "link", "show", "dev", name],
        capture_output=True,
        text=True,
        check=False,
    )
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    flags = first_line.split("<", 1)[-1].split(">", 1)[0].split(",")
    if result.returncode or "UP" not in flags or "bitrate 1000000" not in result.stdout:
        raise RuntimeError(f"{name} 不是 UP / 1 Mbps")
    print(f"  {name}: UP / 1 Mbps")


def _message_values(message, length: int, name: str) -> tuple[float, ...]:
    if message is None or len(message.msg) != length:
        raise RuntimeError(f"缺少有效的{name}反馈")
    return tuple(float(value) for value in message.msg)


class PiperPair:
    def __init__(
        self,
        side: str,
        leader_name: str,
        follower_name: str,
        joint6_deg: float,
        *,
        firmware: str,
        home_speed: int,
        home_timeout_s: float,
        home_tolerance_deg: float,
        home_leader_wait_s: float,
        preposition_speed: int,
        preposition_timeout_s: float,
        zero_tolerance_deg: float,
        follow_hz: float,
        status_period_s: float,
        gripper_force: float,
        gripper_max_width: float,
        gripper_home_timeout_s: float,
        gripper_home_tolerance_m: float,
    ):
        self.side = side
        self.leader_name = leader_name
        self.follower_name = follower_name
        self.firmware = firmware
        self.home_speed = home_speed
        self.home_timeout_s = home_timeout_s
        self.home_tolerance = math.radians(home_tolerance_deg)
        self.home_leader_wait_s = home_leader_wait_s
        self.preposition_speed = preposition_speed
        self.preposition_timeout_s = preposition_timeout_s
        self.zero_tolerance = math.radians(zero_tolerance_deg)
        self.follow_hz = follow_hz
        self.status_period_s = status_period_s
        self.gripper_force = gripper_force
        self.gripper_max_width = gripper_max_width
        self.gripper_home_timeout_s = gripper_home_timeout_s
        self.gripper_home_tolerance_m = gripper_home_tolerance_m
        self.target = (*ZERO_JOINTS[:5], math.radians(joint6_deg))
        self.leader: Any = None
        self.follower: Any = None
        self.leader_gripper: Any = None
        self.follower_gripper: Any = None
        self._ports = ExitStack()
        self._leader_is_follower = False
        self._command = ArmCommand(self.target, 0.0, time.monotonic())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def _create_arm(self, channel: str):
        from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config

        config = create_agx_arm_config(
            robot=ArmModel.PIPER,
            firmeware_version=self.firmware,
            channel=channel,
        )
        arm = AgxArmFactory.create_arm(config)
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        arm.set_joint_limits_enabled(True)
        arm.connect()
        self._ports.callback(arm.disconnect)
        return arm, gripper

    def connect(self) -> None:
        self.follower, self.follower_gripper = self._create_arm(self.follower_name)
        self.leader, self.leader_gripper = self._create_arm(self.leader_name)
        self._wait_follower_feedback(timeout=3.0)
        self.follower.set_follower_mode()
        self.leader.set_leader_mode()
        self.leader.restore_leader_drag_mode()
        print(
            f"{self.side}: {self.leader_name}(leader) -> {self.follower_name}(follower)"
        )

    def _wait_follower_feedback(
        self, timeout: float, after: float | None = None
    ) -> None:
        deadline = time.monotonic() + timeout
        for now in ticks(50):
            joints = self.follower.get_joint_angles()
            fresh = joints is not None and (after is None or joints.timestamp > after)
            if (
                fresh
                and self.follower.get_flange_pose() is not None
                and self.follower_gripper.get_gripper_status() is not None
            ):
                return
            if now >= deadline:
                raise RuntimeError(f"{self.follower_name} 没有完整反馈")

    @staticmethod
    def _wait_enabled(arm, name: str, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if arm.enable():
                return
            time.sleep(0.02)
        raise RuntimeError(f"{name} 使能失败")

    @staticmethod
    def _check_status(arm, name: str) -> None:
        status = arm.get_arm_status()
        if status is None:
            raise RuntimeError(f"{name} 没有状态反馈")
        if int(status.msg.arm_status) != 0:
            raise RuntimeError(f"{name} 状态异常: {status.msg}")

    def _move_to(
        self,
        arms: dict[str, Any],
        target: tuple[float, ...],
        *,
        speed: int,
        timeout: float,
        tolerance: float,
        hz: float,
    ) -> None:
        for name, arm in arms.items():
            arm.set_speed_percent(speed)
            self._wait_enabled(arm, name)

        deadline = time.monotonic() + timeout
        settled = 0
        for now in ticks(hz):
            ready = True
            for name, arm in arms.items():
                arm.move_j(list(target))
                self._check_status(arm, name)
                actual = _message_values(arm.get_joint_angles(), 6, f"{name} 关节")
                ready &= max(abs(a - b) for a, b in zip(actual, target)) <= tolerance
            settled = settled + 1 if ready else 0
            if settled >= 3:
                return
            if now >= deadline:
                raise RuntimeError(f"{', '.join(arms)} 运动到目标超时")

    def _follower_width(self) -> float:
        status = self.follower_gripper.get_gripper_status()
        if status is None:
            raise RuntimeError(f"{self.follower_name} 没有夹爪反馈")
        if status.msg.mode != "width":
            raise RuntimeError(f"{self.follower_name} 夹爪不是 width 模式")
        return min(self.gripper_max_width, max(0.0, float(status.msg.value)))

    def preposition(self) -> None:
        follower_joints = _message_values(
            self.follower.get_joint_angles(), 6, f"{self.follower_name} 关节"
        )
        if max(map(abs, follower_joints)) > self.zero_tolerance:
            raise RuntimeError(f"{self.follower_name} 不在 ZERO")

        previous = self.leader.get_joint_angles()
        previous_timestamp = previous.timestamp if previous is not None else 0.0
        self.leader.set_follower_mode()
        self._leader_is_follower = True
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            joints = self.leader.get_joint_angles()
            if joints is not None and joints.timestamp > previous_timestamp:
                break
            time.sleep(0.02)
        else:
            raise RuntimeError(f"{self.leader_name} 切换 follower 后没有关节反馈")

        if (
            max(map(abs, _message_values(joints, 6, f"{self.leader_name} 关节")))
            > self.zero_tolerance
        ):
            raise RuntimeError(f"{self.leader_name} 不在 ZERO")

        width = self._follower_width()
        with self._lock:
            self._command = ArmCommand(self.target, width, time.monotonic())
        self._move_to(
            {self.leader_name: self.leader, self.follower_name: self.follower},
            self.target,
            speed=self.preposition_speed,
            timeout=self.preposition_timeout_s,
            tolerance=self.zero_tolerance,
            hz=50,
        )
        print(f"{self.side}: J6 到达 {math.degrees(self.target[-1]):+.1f} deg")

    def _restore_teaching(self) -> None:
        self.leader.set_leader_mode()
        time.sleep(0.05)
        self.leader.restore_leader_drag_mode()
        self._leader_is_follower = False

    def start_following(self) -> None:
        with self._lock:
            command = self._command
        self.follower.set_speed_percent(100)
        self._wait_enabled(self.follower, self.follower_name)
        for _ in range(3):
            self.follower.move_j(list(command.joints))
            self.follower_gripper.move_gripper_m(
                command.gripper_width, force=self.gripper_force
            )
            time.sleep(0.01)

        self._restore_teaching()
        leader_joints = self.leader.get_leader_joint_angles()
        leader_gripper = self.leader_gripper.get_gripper_ctrl_states()
        self._error = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._follow_loop,
            args=(
                command,
                0.0 if leader_joints is None else leader_joints.timestamp,
                0.0 if leader_gripper is None else leader_gripper.timestamp,
            ),
            name=f"pyagxarm-follow-{self.side}",
            daemon=True,
        )
        self._thread.start()
        print(f"{self.side}: 跟随已启动 ({self.follow_hz:g} Hz)")

    @staticmethod
    def _clamp_joints(values) -> tuple[float, ...]:
        return tuple(
            min(high, max(low, float(value)))
            for value, (low, high) in zip(values, JOINT_LIMITS_RAD)
        )

    def _follow_loop(
        self, command: ArmCommand, joint_timestamp: float, gripper_timestamp: float
    ) -> None:
        next_check = time.monotonic() + self.status_period_s
        try:
            for now in ticks(self.follow_hz, self._stop):
                joints = command.joints
                width = command.gripper_width
                changed = False

                joint_message = self.leader.get_leader_joint_angles()
                if (
                    joint_message is not None
                    and joint_message.timestamp != joint_timestamp
                ):
                    joints = self._clamp_joints(joint_message.msg)
                    joint_timestamp = joint_message.timestamp
                    changed = True

                gripper_message = self.leader_gripper.get_gripper_ctrl_states()
                if (
                    gripper_message is not None
                    and gripper_message.timestamp != gripper_timestamp
                ):
                    if gripper_message.msg.status_code not in (0, 1, 2, 3):
                        raise RuntimeError(
                            f"{self.leader_name} 夹爪控制回显不是 width 模式"
                        )
                    width = min(
                        self.gripper_max_width,
                        max(0.0, abs(float(gripper_message.msg.value))),
                    )
                    gripper_timestamp = gripper_message.timestamp
                    changed = True

                command = ArmCommand(
                    joints,
                    width,
                    now if changed else command.timestamp,
                )
                with self._lock:
                    self._command = command
                # Real-time leader/follower control uses JS passthrough mode.
                # Unlike move_j(), move_js() avoids position-trajectory
                # smoothing, reducing teleoperation lag while preserving the
                # existing 200 Hz Python control loop.
                self.follower.move_js(list(command.joints))
                self.follower_gripper.move_gripper_m(
                    command.gripper_width, force=self.gripper_force
                )

                if now >= next_check:
                    self._check_status(self.follower, self.follower_name)
                    next_check = now + self.status_period_s
        except Exception as exc:  # noqa: BLE001 - relay background control failure
            self._error = exc

    def sample(self) -> ArmSample:
        if self._error is not None:
            raise RuntimeError(f"{self.side} 跟随停止: {self._error}") from self._error
        joints = _message_values(
            self.follower.get_joint_angles(), 6, f"{self.follower_name} 关节"
        )
        pose = _message_values(
            self.follower.get_flange_pose(), 6, f"{self.follower_name} 法兰位姿"
        )
        gripper = self.follower_gripper.get_gripper_status()
        if gripper is None or gripper.msg.mode != "width":
            raise RuntimeError(f"{self.follower_name} 缺少 width 模式夹爪反馈")
        with self._lock:
            command = self._command
        return ArmSample(
            joints,
            pose,
            float(gripper.msg.value),
            float(gripper.msg.force),
            command,
        )

    def prepare_home(self) -> None:
        self.stop_following()
        with self._lock:
            self._command = ArmCommand(ZERO_JOINTS, 0.0, time.monotonic())

    def _home_gripper(self, gripper: Any, name: str) -> None:
        """Close a gripper to 0 m and require fresh width feedback at zero."""
        previous = gripper.get_gripper_status()
        previous_timestamp = previous.timestamp if previous is not None else 0.0
        deadline = time.monotonic() + self.gripper_home_timeout_s
        settled = 0
        for now in ticks(20):
            gripper.move_gripper_m(0.0, force=self.gripper_force)
            status = gripper.get_gripper_status()
            if status is not None and status.msg.mode != "width":
                raise RuntimeError(f"{name} 夹爪不是 width 模式")
            fresh = status is not None and status.timestamp > previous_timestamp
            at_zero = fresh and abs(float(status.msg.value)) <= self.gripper_home_tolerance_m
            settled = settled + 1 if at_zero else 0
            if settled >= 3:
                print(f"{name}: 夹爪 ZERO 完成 (0 m)")
                return
            if now >= deadline:
                value = None if status is None else float(status.msg.value)
                raise RuntimeError(f"{name} 夹爪回零超时，当前宽度={value!r} m")

    def home_follower(self) -> None:
        self._move_to(
            {self.follower_name: self.follower},
            ZERO_JOINTS,
            speed=self.home_speed,
            timeout=self.home_timeout_s,
            tolerance=self.home_tolerance,
            hz=50,
        )
        print(f"{self.follower_name}: ZERO 完成")
        self._home_gripper(self.follower_gripper, self.follower_name)

    def home_leader(self) -> None:
        self.leader.move_leader_to_home()
        time.sleep(self.home_leader_wait_s)
        try:
            self._home_gripper(self.leader_gripper, self.leader_name)
        finally:
            self.leader.restore_leader_drag_mode()
        print(f"{self.leader_name}: 主臂和夹爪回零完成")

    def stop_following(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError(f"{self.side} 跟随线程未退出")
            self._thread = None

    def close(self) -> None:
        with self._ports:
            self.stop_following()
            if self._leader_is_follower:
                self._restore_teaching()


class DualPiper:
    def __init__(self, pipers: dict[str, dict], **motion):
        self.pairs = {side: PiperPair(side, **pipers[side], **motion) for side in SIDES}

    def _run_parallel(self, operation: Callable[[PiperPair], None]) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(operation, pair) for pair in self.pairs.values()]
            for future in futures:
                future.result()

    def connect(self) -> None:
        print("四臂 CAN 接口预检...")
        for pair in self.pairs.values():
            verify_can_interface(pair.leader_name)
            verify_can_interface(pair.follower_name)
        self._run_parallel(PiperPair.connect)

    def preposition_and_start(self) -> None:
        print("四臂执行 J6 预定位...")
        self._run_parallel(PiperPair.preposition)
        self._run_parallel(PiperPair.start_following)

    def sample(self) -> dict[str, ArmSample]:
        return {side: pair.sample() for side, pair in self.pairs.items()}

    def home(
        self, on_sample: Callable[[], None] | None = None, sample_hz: float = 20
    ) -> None:
        for pair in self.pairs.values():
            pair.prepare_home()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(method)
                for pair in self.pairs.values()
                for method in (pair.home_follower, pair.home_leader)
            ]
            if on_sample is not None:
                for _ in ticks(sample_hz):
                    on_sample()
                    if all(future.done() for future in futures):
                        break
            for future in futures:
                future.result()

    def close(self) -> None:
        for pair in self.pairs.values():
            pair.close()
