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
GRIPPER_LIMITS_M = (0.0, 0.1)
HARDWARE_TOLERANCE_M = 1e-3
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
    values = tuple(float(value) for value in message.msg)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{name}反馈含 NaN/Inf")
    return values


def _normalize_gripper_width(value: float, maximum: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} 夹爪宽度含 NaN/Inf")
    return min(maximum, max(GRIPPER_LIMITS_M[0], value))


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
        preposition_speed: int,
        preposition_timeout_s: float,
        zero_tolerance_deg: float,
        follow_hz: float,
        status_period_s: float,
        gripper_force: float,
        gripper_max_width: float,
        gripper_home_timeout_s: float,
        gripper_home_tolerance_m: float,
        start_target=None,
    ):
        self.side = side
        self.leader_name = leader_name
        self.follower_name = follower_name
        self.firmware = firmware
        self.home_speed = home_speed
        self.home_timeout_s = home_timeout_s
        self.home_tolerance = math.radians(home_tolerance_deg)
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
        self.custom_start = start_target is not None
        self._home_track_leader = not self.custom_start
        self.home_joints = tuple(start_target[:6]) if self.custom_start else ZERO_JOINTS
        self.home_width = float(start_target[6]) if self.custom_start else 0.0
        if self.custom_start:
            for i, (value, (lo, hi)) in enumerate(zip(self.home_joints, JOINT_LIMITS_RAD)):
                if not math.isfinite(value) or not lo <= value <= hi:
                    raise ValueError(f"{side} 起点 J{i+1}={math.degrees(value):.3f}° 超出可下发范围，请核对起点")
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
        self._last_sample_durations: dict[str, float] = {}

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
        self._ports.callback(arm.disconnect)
        arm.connect()
        return arm, gripper

    def connect(self) -> None:
        self.follower, self.follower_gripper = self._create_arm(self.follower_name)
        self.leader, self.leader_gripper = self._create_arm(self.leader_name)
        # Another application may have left this physical follower in leader
        # mode. Restore its role before waiting for follower-only feedback.
        # Do not enable motors or send motion targets during this handshake.
        switched_at = time.time()
        self.follower.set_follower_mode()
        self._wait_follower_feedback(timeout=3.0, after=switched_at)
        self.leader.set_leader_mode()
        self.leader.restore_leader_drag_mode()
        pendant = self.leader_gripper.get_gripper_teaching_pendant_param(
            timeout=2.0, min_interval=0.0
        )
        if pendant is None:
            raise RuntimeError(f"{self.leader_name} 夹爪示教器没有响应")
        print(
            f"{self.side}: {self.leader_name}(leader) -> "
            f"{self.follower_name}(follower)，夹爪示教器在线"
        )

    def _wait_follower_feedback(
        self, timeout: float, after: float | None = None
    ) -> None:
        deadline = time.monotonic() + timeout
        for now in ticks(50):
            messages = (
                self.follower.get_joint_angles(),
                self.follower.get_flange_pose(),
                self.follower_gripper.get_gripper_status(),
            )
            if all(message is not None and (after is None or message.timestamp > after)
                   for message in messages):
                return
            if now >= deadline:
                missing = [name for name, message in zip(("关节", "法兰", "夹爪"), messages)
                           if message is None or (after is not None and message.timestamp <= after)]
                raise RuntimeError(f"{self.follower_name} 没有完整反馈（已请求从臂模式，缺少新鲜的{'、'.join(missing)}反馈）")

    @staticmethod
    def _enable_failure_details(arm) -> str:
        details = []
        fault_names = (
            ("voltage_too_low", "欠压"),
            ("motor_overheating", "电机过温"),
            ("driver_overcurrent", "过流"),
            ("driver_overheating", "驱动器过温"),
            ("collision_status", "碰撞"),
            ("driver_error_status", "驱动器错误"),
            ("stall_status", "堵转"),
        )
        try:
            joints_message = arm.get_joint_angles()
            joints = None if joints_message is None else tuple(joints_message.msg)
            for index in range(1, 7):
                driver_message = arm.get_driver_states(index)
                if driver_message is None:
                    details.append(f"J{index}:无驱动反馈")
                    continue
                driver = driver_message.msg
                foc = driver.foc_status
                if foc.driver_enable_status:
                    continue
                faults = [label for attr, label in fault_names if bool(getattr(foc, attr, False))]
                motor_message = arm.get_motor_states(index)
                motor = None if motor_message is None else motor_message.msg
                position = (
                    "?"
                    if joints is None or len(joints) < index
                    else f"{math.degrees(float(joints[index - 1])):+.3f}°"
                )
                current = "?" if motor is None else f"{float(motor.current):+.3f}A"
                torque = "?" if motor is None else f"{float(motor.torque):+.3f}N·m"
                fault_text = "/".join(faults) if faults else "无故障位"
                details.append(
                    f"J{index}:位置{position},状态码0x{int(driver.foc_status_code):02X},"
                    f"{fault_text},电流{current},力矩{torque}"
                )
        except Exception as exc:  # noqa: BLE001 - preserve the original enable error
            return f"；读取关节诊断失败: {exc}"
        return "" if not details else "；" + "；".join(details)

    @staticmethod
    def _wait_enabled(arm, name: str, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if arm.enable():
                return
            time.sleep(0.02)
        raise RuntimeError(
            f"{name} 使能失败{PiperPair._enable_failure_details(arm)}"
        )

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
        return _normalize_gripper_width(
            status.msg.value, self.gripper_max_width, self.follower_name
        )

    def _enter_leader_control_mode(self, timeout: float = 5.0):
        previous = self.leader.get_joint_angles()
        previous_timestamp = 0.0 if previous is None else previous.timestamp
        self._leader_is_follower = True
        deadline = time.monotonic() + timeout
        for now in ticks(20):
            # Mode configuration is a single CAN frame. Repeat it until the
            # controller starts publishing follower-mode joint feedback.
            self.leader.set_follower_mode()
            joints = self.leader.get_joint_angles()
            if joints is not None and joints.timestamp > previous_timestamp:
                return joints
            if now >= deadline:
                raise RuntimeError(
                    f"{self.leader_name} 切换 follower 后没有关节反馈"
                )

    def preposition(self) -> None:
        follower_joints = _message_values(
            self.follower.get_joint_angles(), 6, f"{self.follower_name} 关节"
        )
        if max(map(abs, follower_joints)) > self.zero_tolerance:
            raise RuntimeError(f"{self.follower_name} 不在 ZERO")

        joints = self._enter_leader_control_mode()
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

        joint_timestamps = self._leader_joint_group_timestamps()
        leader_gripper = self.leader_gripper.get_gripper_ctrl_states()
        self._restore_teaching()
        self._error = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._follow_loop,
            args=(
                command,
                joint_timestamps,
                0.0
                if leader_gripper is None
                else leader_gripper.timestamp,
            ),
            name=f"pyagxarm-follow-{self.side}",
            daemon=True,
        )
        self._thread.start()
        print(f"{self.side}: 跟随已启动 ({self.follow_hz:g} Hz)")

    def _leader_joint_group_timestamps(self) -> tuple[float, float, float]:
        parser = self.leader._parser
        return tuple(
            0.0 if message is None else float(message.timestamp)
            for message in (
                getattr(parser, "leader_joint_12", None),
                getattr(parser, "leader_joint_34", None),
                getattr(parser, "leader_joint_56", None),
            )
        )

    def _complete_leader_joints(
        self, previous_timestamps: tuple[float, float, float]
    ) -> tuple[tuple[float, ...], tuple[float, float, float]] | None:
        parser = self.leader._parser
        messages = (
            getattr(parser, "leader_joint_12", None),
            getattr(parser, "leader_joint_34", None),
            getattr(parser, "leader_joint_56", None),
        )
        if any(message is None for message in messages):
            return None
        timestamps = tuple(float(message.timestamp) for message in messages)
        if any(
            timestamp <= previous
            for timestamp, previous in zip(timestamps, previous_timestamps)
        ):
            return None
        values = (
            float(messages[0].msg.joint_1),
            float(messages[0].msg.joint_2),
            float(messages[1].msg.joint_3),
            float(messages[1].msg.joint_4),
            float(messages[2].msg.joint_5),
            float(messages[2].msg.joint_6),
        )
        # If the receive thread updated a group while values were copied, wait
        # for the next loop instead of combining two CAN cycles.
        if timestamps != tuple(float(message.timestamp) for message in messages):
            return None
        return self._clamp_joints(values), timestamps

    @staticmethod
    def _clamp_joints(values) -> tuple[float, ...]:
        values = tuple(float(value) for value in values)
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise RuntimeError("关节控制量必须为有限的 6 维数组")
        return tuple(
            min(high, max(low, float(value)))
            for value, (low, high) in zip(values, JOINT_LIMITS_RAD)
        )

    def _follow_loop(
        self,
        command: ArmCommand,
        joint_timestamps: tuple[float, float, float],
        gripper_timestamp: float,
    ) -> None:
        next_check = time.monotonic() + self.status_period_s
        try:
            for now in ticks(self.follow_hz, self._stop):
                joints = command.joints
                width = command.gripper_width
                changed = False

                joint_frame = self._complete_leader_joints(joint_timestamps)
                if joint_frame is not None:
                    candidate_joints, joint_timestamps = joint_frame
                    if candidate_joints != joints:
                        joints = candidate_joints
                        changed = True

                gripper_message = self.leader_gripper.get_gripper_ctrl_states()
                if (
                    gripper_message is not None
                    and gripper_message.timestamp > gripper_timestamp
                ):
                    candidate_width = _normalize_gripper_width(
                        gripper_message.msg.value,
                        self.gripper_max_width,
                        self.leader_name,
                    )
                    message_timestamp = gripper_message.timestamp
                    gripper_timestamp = message_timestamp
                    if candidate_width != width:
                        width = candidate_width
                        changed = True

                if changed:
                    # SocketCAN message timestamps use the Unix clock, while the
                    # recorder timeline uses time.monotonic(). Stamp accepted
                    # commands on the recorder clock so HDF5 diagnostics remain
                    # comparable with episode timestamps.
                    command = ArmCommand(joints, width, now)
                with self._lock:
                    self._command = command
                # Poll leader feedback at follow_hz, but only send a control
                # frame when the target value actually changes. The current
                # Piper firmware retains the latest JS target, so repeating
                # identical frames only adds CAN traffic.
                if changed:
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
        started_at = time.monotonic()
        joint_started_at = started_at
        joints = _message_values(
            self.follower.get_joint_angles(), 6, f"{self.follower_name} 关节"
        )
        joint_duration = time.monotonic() - joint_started_at
        pose_started_at = time.monotonic()
        pose = _message_values(
            self.follower.get_flange_pose(), 6, f"{self.follower_name} 法兰位姿"
        )
        pose_duration = time.monotonic() - pose_started_at
        gripper_started_at = time.monotonic()
        gripper = self.follower_gripper.get_gripper_status()
        if gripper is None or gripper.msg.mode != "width":
            raise RuntimeError(f"{self.follower_name} 缺少 width 模式夹爪反馈")
        width = _normalize_gripper_width(
            gripper.msg.value, self.gripper_max_width, self.follower_name
        )
        # Firmware versions expose different force feedback scales. Preserve
        # the SDK value for logging; force feedback is not a control gate.
        force = float(gripper.msg.force)
        self._last_sample_durations = {
            "joints": joint_duration,
            "pose": pose_duration,
            "gripper": time.monotonic() - gripper_started_at,
            "total": time.monotonic() - started_at,
        }
        with self._lock:
            command = self._command
        return ArmSample(
            joints,
            pose,
            width,
            force,
            command,
        )

    def prepare_home(self) -> None:
        # Keep the original leader-driven return after a normal recording.
        # A newly hand-positioned follower may be far from its leader instead.
        self._home_track_leader = not self.custom_start or bool(self._thread and self._thread.is_alive())
        self.stop_following()
        with self._lock:
            self._command = ArmCommand(self.home_joints, self.home_width, time.monotonic())

    def _home_gripper(self, gripper: Any, name: str) -> None:
        """Move a gripper to the saved start width with fresh settled feedback."""
        previous = gripper.get_gripper_status()
        previous_timestamp = (
            0.0 if previous is None else previous.timestamp
        )
        deadline = time.monotonic() + self.gripper_home_timeout_s
        settled = 0
        for now in ticks(20):
            gripper.move_gripper_m(self.home_width, force=self.gripper_force)
            status = gripper.get_gripper_status()
            if status is not None and status.msg.mode != "width":
                raise RuntimeError(f"{name} 夹爪不是 width 模式")
            fresh = status is not None and status.timestamp > previous_timestamp
            value = None if status is None else _normalize_gripper_width(
                status.msg.value, self.gripper_max_width, name
            )
            at_zero = fresh and value is not None and abs(value-self.home_width) <= self.gripper_home_tolerance_m
            settled = settled + 1 if at_zero else 0
            if settled >= 3:
                print(f"{name}: 夹爪归位完成 ({self.home_width:g} m)")
                return
            if now >= deadline:
                raise RuntimeError(f"{name} 夹爪回零超时，当前宽度={value!r} m")

    def home_synchronized(self, start_barrier: threading.Barrier) -> None:
        """Reuse the synchronized return, aligning separately after manual setup."""
        try:
            # This mode transition runs while HOME frames are already being
            # sampled. The barrier below still prevents either pair from
            # moving before both leaders are ready.
            self._enter_leader_control_mode()
            for name, arm in (
                (self.leader_name, self.leader),
                (self.follower_name, self.follower),
            ):
                arm.set_speed_percent(self.home_speed)
                self._wait_enabled(arm, name)
            try:
                # A peer can spend up to 5 s switching modes and 5 s enabling
                # each arm before it reaches this synchronization point.
                start_barrier.wait(timeout=16.0)
            except threading.BrokenBarrierError as exc:
                raise RuntimeError("左右两组机械臂回零启动同步失败") from exc

            deadline = time.monotonic() + self.home_timeout_s
            next_check = time.monotonic()
            settled = 0
            for now in ticks(self.follow_hz):
                # Let move_j generate one smooth physical trajectory on the
                # leader. Tracking its measured position keeps the follower
                # from starting or finishing ahead of the leader.
                self.leader.move_j(list(self.home_joints))
                leader_joints = self._clamp_joints(
                    _message_values(
                        self.leader.get_joint_angles(),
                        6,
                        f"{self.leader_name} 关节",
                    )
                )
                if not self._home_track_leader:
                    # At setup the follower may have been hand-positioned while
                    # the leader is elsewhere. Both use firmware-planned paths;
                    # never jump the follower to the leader's initial position.
                    self.follower.move_j(list(self.home_joints))
                else:
                    self.follower.move_js(list(leader_joints))
                follower_joints = _message_values(
                    self.follower.get_joint_angles(),
                    6,
                    f"{self.follower_name} 关节",
                )

                if now >= next_check:
                    self._check_status(self.leader, self.leader_name)
                    self._check_status(self.follower, self.follower_name)
                    next_check = now + self.status_period_s
                ready = all(
                    max(abs(a-b) for a, b in zip(joints, self.home_joints)) <= self.home_tolerance
                    for joints in (leader_joints, follower_joints)
                )
                settled = settled + 1 if ready else 0
                if settled >= 3:
                    break
                if now >= deadline:
                    raise RuntimeError(
                        f"{self.leader_name}, {self.follower_name} 同步回零超时"
                    )

            # Leave the follower in the regular position-speed mode expected
            # by the next J6 preposition operation.
            self.follower.move_j(list(self.home_joints))
            print(f"{self.leader_name}, {self.follower_name}: 同步归位完成")
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = (
                    pool.submit(
                        self._home_gripper,
                        self.leader_gripper,
                        self.leader_name,
                    ),
                    pool.submit(
                        self._home_gripper,
                        self.follower_gripper,
                        self.follower_name,
                    ),
                )
                for future in futures:
                    future.result()
        except BaseException:
            # Wake the peer immediately so its real failure can be collected
            # instead of leaving one side hidden behind a barrier timeout.
            start_barrier.abort()
            raise
        finally:
            # A custom start must remain the idle pose, including its gripper
            # opening. Drag mode can let the teaching gripper spring closed.
            # start_following restores drag immediately before recording.
            if not self.custom_start:
                self._restore_teaching()
        print(f"{self.side}: 主从臂和两个夹爪归位完成")

    def stop_following(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError(f"{self.side} 跟随线程未退出")
            self._thread = None

    def close(self) -> None:
        error = None
        try:
            self.stop_following()
        except Exception as exc:  # noqa: BLE001 - preserve shutdown errors
            error = exc
        finally:
            try:
                if self._leader_is_follower and not self.custom_start:
                    self._restore_teaching()
            except Exception as exc:  # noqa: BLE001 - preserve shutdown errors
                if error is None:
                    error = exc
            finally:
                try:
                    self._ports.close()
                except Exception as exc:  # noqa: BLE001 - preserve shutdown errors
                    if error is None:
                        error = exc
        if error is not None:
            print(f"{self.side}: 关闭资源时出现警告: {error}")


class DualPiper:
    def __init__(self, pipers: dict[str, dict], start_point=None, **motion):
        from easy_embodied_deploy.recording_start import normalize
        self.start_point = normalize(start_point)
        self.pairs = {side: PiperPair(side, **pipers[side], **motion,
                      start_target=self.start_point["arms"][side] if self.start_point["mode"] == "custom" else None)
                      for side in SIDES}
        self._sample_pool = ThreadPoolExecutor(max_workers=len(SIDES), thread_name_prefix="piper-sample")

    def _run_parallel(self, operation: Callable[[PiperPair], None]) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(operation, pair) for pair in self.pairs.values()]
            self._finish_parallel(futures)

    @staticmethod
    def _finish_parallel(futures) -> None:
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - report every physical side
                message = str(exc)
                if message not in errors:
                    errors.append(message)
        if errors:
            raise RuntimeError("；".join(errors))

    def connect(self) -> None:
        print("四臂 CAN 接口预检...")
        for pair in self.pairs.values():
            verify_can_interface(pair.leader_name)
            verify_can_interface(pair.follower_name)
        self._run_parallel(PiperPair.connect)

    def preposition_and_start(self) -> None:
        if self.start_point["mode"] == "custom":
            print("四臂回到自定义起点，跳过 J6 预摆...")
            self.home()
        else:
            print("四臂执行 J6 预定位...")
            self._run_parallel(PiperPair.preposition)
        self._run_parallel(PiperPair.start_following)

    def sample(self) -> dict[str, ArmSample]:
        futures = {side: self._sample_pool.submit(pair.sample) for side, pair in self.pairs.items()}
        return {side: future.result() for side, future in futures.items()}

    def sample_diagnostics(self) -> dict[str, dict[str, float]]:
        return {side: dict(pair._last_sample_durations) for side, pair in self.pairs.items()}

    def home(
        self, on_sample: Callable[[], None] | None = None, sample_hz: float = 20
    ) -> None:
        self._run_parallel(PiperPair.prepare_home)
        start_barrier = threading.Barrier(len(self.pairs))
        with ThreadPoolExecutor(max_workers=len(self.pairs)) as pool:
            futures = [
                pool.submit(pair.home_synchronized, start_barrier)
                for pair in self.pairs.values()
            ]
            if on_sample is not None:
                for _ in ticks(sample_hz):
                    on_sample()
                    if all(future.done() for future in futures):
                        break
            self._finish_parallel(futures)

    def close(self) -> None:
        error = None
        try:
            for pair in self.pairs.values():
                try:
                    pair.close()
                except Exception as exc:  # noqa: BLE001 - close all physical arms
                    if error is None:
                        error = exc
        finally:
            self._sample_pool.shutdown(wait=True, cancel_futures=True)
        if error is not None:
            print(f"关闭机械臂资源时出现警告: {error}")
