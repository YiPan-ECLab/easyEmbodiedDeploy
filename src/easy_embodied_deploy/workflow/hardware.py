"""Workflow-owned hardware session. Importing this module opens no devices."""

import sys
import threading
import time
from contextlib import ExitStack

import numpy as np

from ..calibration import DEFAULT_SERIALS, sdk_intrinsics
from ..module_http import ROOT
from ..pose_actions import finite
from ..resources import hardware_lease

HOLD_FEEDBACK_TOLERANCE = np.deg2rad(3)


def bounded_joints(values, limits, label, tolerance=1e-7):
    """Accept only a bounded feedback margin; never widen commanded limits."""
    q = finite(values, (6,), label)
    bounded = np.clip(q, limits[:, 0], limits[:, 1])
    outside = np.abs(q-bounded) > tolerance
    if np.any(outside):
        i = int(np.flatnonzero(outside)[0])
        raise ValueError(f"{label} J{i+1}={np.degrees(q[i]):.3f}°，可执行范围 "
                         f"[{np.degrees(limits[i, 0]):.3f}°, {np.degrees(limits[i, 1]):.3f}°]")
    return bounded


def selected_sides(side):
    if side not in ("left", "right", "both"):
        raise ValueError("请选择左从臂、右从臂或双从臂")
    return ("left", "right") if side == "both" else (side,)


class MockHardware:
    simulated = True

    def __init__(self, config=None):
        self.joints = np.array([0, .2, -.2, 0, 0, -.785, .04, 0, .2, -.2, 0, 0, .785, .04])
        self.poses = dict(left=[.25, .1, .3, 0, 0, 0], right=[.25, -.1, .3, 0, 0, 0])
        self.teaching = False
        self.teaching_sides = set()
        self.commands = []
        self.emergency = False

    def snapshot(self):
        return dict(joints=self.joints.tolist(), poses={k: list(v) for k, v in self.poses.items()},
                    teaching=self.teaching, simulated=True)

    def point_values(self, space, sides):
        snap = self.snapshot()
        return {side: ([*snap["poses"][side], snap["joints"][offset+6]] if space == "pose"
                       else snap["joints"][offset:offset+7])
                for side, offset in (("left", 0), ("right", 7)) if side in sides}

    def capture_point(self, space, sides):
        arms = self.point_values(space, sides)
        return arms, dict(captured_at_s=time.time(), feedback={
            side: dict(joint_angles=self.point_values("joint", (side,))[side][:6],
                       simulated=True) for side in sides})

    def prepare(self, active_sides=("left", "right"), check=lambda: None):
        if self.teaching:
            raise ValueError("请先结束拖动示教")
        if self.emergency:
            raise ValueError("急停已触发，请重新连接设备")

    def teach(self, enabled, side="both", check=lambda: None):
        if self.emergency:
            raise ValueError("急停后请重新连接")
        self.teaching_sides = set(selected_sides(side)) if enabled else set()
        self.teaching = bool(self.teaching_sides)

    def recover(self):
        self.teach(False)

    def manual_mode_sides(self):
        return []

    def home(self, check, progress, close_grippers=False):
        from .plan import interpolate
        if self.teaching:
            self.teach(False)
        self.prepare()
        start = self.joints.copy()
        end = start.copy()
        end[:6] = end[7:13] = 0
        if close_grippers:
            end[[6, 13]] = 0
        for i in range(1, 26):
            check()
            self.send("joint", interpolate(start, end, i/25))
            progress(i/25)
            time.sleep(.02)
        check()
        self.send("joint", end)

    def send(self, space, target, active_sides=("left", "right"), *, joint_command="move_js"):
        if joint_command not in ("move_js", "move_j"):
            raise ValueError("关节下发接口须为 move_js 或 move_j")
        self.prepare()
        target = finite(target, (14,), "目标")
        for side, offset in (("left", 0), ("right", 7)):
            if side not in active_sides:
                continue
            if space == "joint":
                self.joints[offset:offset+7] = target[offset:offset+7]
            else:
                self.poses[side] = target[offset:offset+6].tolist()
                self.joints[offset+6] = target[offset+6]
        self.commands.append((space, target.copy()))
        self.commands = self.commands[-500:]

    def stop(self, emergency=False):
        self.emergency = self.emergency or emergency

    def close(self):
        pass

    def camera(self, role):
        yy, xx = np.indices((480, 640))
        rgb = np.stack((xx % 256, yy % 256, np.full_like(xx, 100)), axis=-1).astype(np.uint8)
        camera = dict(role=role, serial="SIMULATED", sensor="COLOR_SENSOR", width=640, height=480,
                      intrinsics=dict(K=[[600, 0, 320], [0, 600, 240], [0, 0, 1]],
                                      D=[0]*5, model="brown"))
        return rgb, np.full((480, 640), .4), camera

    def observation(self):
        result = {"observation.state": self.joints.astype(np.float32).copy(),
                  "observation.flange_pose": np.r_[self.poses["left"], self.poses["right"]]}
        for role in DEFAULT_SERIALS:
            result[f"observation.images.{role}"] = np.zeros((224, 224, 3), np.uint8)
        return result


class DepthCamera:
    """Orbbec SDK software D2C, with RGB and depth from the same frameset."""

    def __init__(self, role, serial, context):
        from pyorbbecsdk import (Pipeline, Config, OBSensorType, OBFormat,
                                OBFrameAggregateOutputMode, AlignFilter, OBStreamType)
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.latest, self.error = None, None
        device = context.query_devices().get_device_by_serial_number(serial)
        if device is None:
            raise ValueError(f"未找到相机 {role} / {serial}")
        self.pipeline = Pipeline(device)
        profile = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(640, 480, OBFormat.RGB, 30)
        depth = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
        config = Config()
        config.enable_stream(profile)
        config.enable_stream(depth)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.metadata = dict(role=role, serial=serial, sensor="COLOR_SENSOR", width=640, height=480,
                             intrinsics=sdk_intrinsics(profile))
        self.pipeline.start(config)
        self.thread = threading.Thread(target=self._run, name=f"workflow-depth-{role}", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                aligned = self.align.process(frames)
                if aligned is None:
                    continue
                frames = aligned.as_frame_set()
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if color is None or depth is None:
                    continue
                rgb = np.asarray(color.get_data()).reshape(color.get_height(), color.get_width(), 3).copy()
                z = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(depth.get_height(), depth.get_width()).astype(float)
                z *= depth.get_depth_scale() / 1000.0
                if z.shape != rgb.shape[:2]:
                    raise ValueError("深度没有成功对齐到 RGB")
                with self.lock:
                    self.latest = (rgb, z, time.monotonic())
        except Exception as exc:
            self.error = exc

    def read(self):
        with self.lock:
            if self.error:
                raise RuntimeError(f"深度相机异常：{self.error}")
            if self.latest is None or time.monotonic() - self.latest[2] > .5:
                raise RuntimeError("深度相机尚未提供新鲜帧")
            return self.latest[0].copy(), self.latest[1].copy(), dict(self.metadata)

    def close(self):
        self.stop_event.set()
        self.thread.join(2)
        self.pipeline.stop()


class PiperHardware:
    simulated = False

    def __init__(self, config):
        self.lock = threading.RLock()
        self.resources = ExitStack()
        self.arms, self.grippers, self.cameras = {}, {}, {}
        self.teaching = self.emergency = False
        self.teaching_sides, self.armed_sides = set(), set()
        try:
            # Acquire before imports, SDK contexts, CAN connections or mode writes.
            self.resources.enter_context(hardware_lease())
            sys.path.insert(0, str(ROOT / "scripts/PiperRecorder"))
            from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
            from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
            from record import FIRMWARE, PIPERS
            self.limits = np.asarray([ROBOT_JOINT_LIMIT_PRESET_RAD[ArmModel.PIPER][f"joint{i}"] for i in range(1, 7)])
            channels = {side: config.get("channels", {}).get(side, PIPERS[side]["follower_name"])
                        for side in ("left", "right")}
            self.channels = channels
            if len(set(channels.values())) != 2 or any(
                    channel in {pair["leader_name"] for pair in PIPERS.values()} for channel in channels.values()):
                raise ValueError("工作流只连接两个不同的从臂 CAN；不能选择主臂接口")
            for side, channel in channels.items():
                arm = AgxArmFactory.create_arm(create_agx_arm_config(robot=ArmModel.PIPER,
                    firmeware_version=FIRMWARE, channel=channel))
                self.resources.callback(arm.disconnect)
                arm.set_joint_limits_enabled(True)
                self.grippers[side] = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
                arm.connect()
                self.arms[side] = arm
            self.serials = {**DEFAULT_SERIALS, **config.get("serials", {})}
            self.depth_roles = set(config.get("depth_roles", ["agentview"]))
            self.context = None
            # Opening is read-only: no home, enable or motion command.
        except BaseException:
            self.resources.close()
            raise

    @staticmethod
    def _fresh(message, label):
        if message is None or not -.02 <= time.time() - float(message.timestamp) <= .5:
            raise RuntimeError(f"{label} 反馈未就绪或已过期")
        return message

    def _check_emergency(self):
        if self.emergency:
            raise RuntimeError("已急停；请在设备侧检查并复位后重新连接")

    @staticmethod
    def _manual_mode(status):
        ctrl = int(getattr(status, "ctrl_mode", 0))
        teach = int(getattr(status, "teach_status", 0))
        arm = int(status.arm_status)
        # Piper retains ctrl_mode=2 after the physical button ends recording:
        # measured 0x2A1 payload: 02 00 01 02 01 00 00 00.
        # STOP_RECORDING with NORMAL permits the next CAN motion command;
        # waiting for ctrl_mode to become CAN_CTRL first deadlocks that handoff.
        # Do not override an active recording, a fault or linkage teaching.
        if teach == 1 or arm == 0x0B:
            return True
        if ctrl == 2 and teach == 2 and arm == 0:
            return False
        return ctrl in (2, 6)

    def _status(self, side):
        status = self._fresh(self.arms[side].get_arm_status(), f"{side} 从臂状态")
        if self._manual_mode(status.msg):
            raise RuntimeError(f"{side} 从臂仍处于机身按键示教，请先用机身按钮退出；电机不会被释放")
        if int(status.msg.arm_status) != 0:
            raise RuntimeError(f"{side} 从臂报告故障：{status.msg}")

    def manual_mode_sides(self):
        sides = []
        for side, arm in self.arms.items():
            status = self._fresh(arm.get_arm_status(), f"{side} 从臂状态").msg
            if self._manual_mode(status):
                sides.append(side)
        return sides

    def point_values(self, space, sides):
        """Read measured follower feedback without changing motor power or mode.

        A point for one side must not depend on the other side's feedback or
        on flange feedback when recording joint angles.
        """
        with self.lock:
            values = {}
            for side in sides:
                arm = self.arms[side]
                if space == "joint":
                    q, _ = self._joint_sample(side)
                else:
                    q = finite(self._fresh(arm.get_flange_pose(), f"{side} 从臂法兰").msg, (6,), "反馈")
                grip = self._fresh(self.grippers[side].get_gripper_status(), f"{side} 夹爪").msg
                if grip.mode != "width":
                    raise ValueError("夹爪须为 width 模式")
                width = float(grip.value)
                if not np.isfinite(width) or not 0 <= width <= .1:
                    raise ValueError("夹爪反馈宽度超出范围")
                values[side] = [*q.tolist(), width]
            return values

    def _joint_sample(self, side):
        """Copy all measured CAN groups, not the SDK's last-group timestamp.

        get_joint_angles() fills absent axes with cached/initial values and dates
        the aggregate with the last received group. That alone cannot establish
        a complete six-axis sample. Use the same raw groups as the recorder.
        """
        parser = self.arms[side]._parser
        groups = ("joint_12", "joint_34", "joint_56")
        deadline = time.monotonic()+.25
        while True:
            messages = [getattr(parser, name, None) for name in groups]
            if all(m is not None for m in messages):
                stamps = [float(m.timestamp) for m in messages]
                now = time.time()
                if all(-.02 <= now-t <= .15 for t in stamps) and max(stamps)-min(stamps) <= .05:
                    values = [float(getattr(messages[i//2].msg, f"joint_{i+1}")) for i in range(6)]
                    if stamps == [float(getattr(parser, name).timestamp) for name in groups]:
                        return finite(values, (6,), "完整关节反馈"), dict(
                            channel=self.channels[side], received_at_s=stamps,
                            read_at_s=now, max_age_ms=(now-min(stamps))*1000,
                            group_span_ms=(max(stamps)-min(stamps))*1000)
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{side} 从臂六轴反馈不完整、过期或不同步；未录制缓存角度，请检查从臂模式和 CAN")
            time.sleep(.005)

    def capture_point(self, space, sides):
        with self.lock:
            arms = self.point_values(space, sides)
            feedback = {}
            for side in sides:
                joints, meta = self._joint_sample(side)
                if space == "joint":
                    arms[side][:6] = joints.tolist()
                feedback[side] = dict(meta, joint_angles=joints.tolist(), joint_limits=self.limits.tolist())
            return arms, dict(captured_at_s=time.time(), feedback=feedback,
                              feedback_version=2)

    def snapshot(self):
        with self.lock:
            values = self.point_values("joint", ("left", "right"))
            poses = {side: self.point_values("pose", (side,))[side][:6] for side in self.arms}
            return dict(joints=values["left"]+values["right"], poses=poses,
                        teaching=self.teaching, simulated=False)

    def _power(self, side, enabled, timeout=5, check=lambda: None):
        """Do not trust SDK bools backed by missing/cached driver feedback."""
        arm = self.arms[side]
        started, deadline = time.time(), time.monotonic()+timeout
        while time.monotonic() < deadline:
            check()
            self._check_emergency()
            (arm.enable if enabled else arm.disable)()
            drivers = [arm.get_driver_states(i) for i in range(1, 7)]
            if all(m is not None and m.timestamp >= started and time.time()-m.timestamp < .5
                   and bool(m.msg.foc_status.driver_enable_status) == enabled for m in drivers):
                return
            time.sleep(.02)
        raise RuntimeError(f"{side} 从臂{'使能' if enabled else '释放'}未收到六轴确认；请检查电机反馈")

    def prepare(self, active_sides=("left", "right"), check=lambda: None):
        with self.lock:
            self._check_emergency()
            if self.teaching:
                raise ValueError("请先结束手动拖动并保持当前位置")
            values = self.point_values("joint", active_sides)
            for side in active_sides:
                self._status(side)
                # A physical teaching-button cycle can change motor enable
                # state without a web teaching session. Reconfirm on every
                # motion handoff; armed_sides only tracks ownership for stop().
                self._check_emergency()
                # Prime the current measured target before enabling; no role
                # switch and no old leader target is involved.
                q = bounded_joints(values[side][:6], self.limits, f"{side} 当前反馈", HOLD_FEEDBACK_TOLERANCE)
                self.arms[side].move_js(q.tolist())
                self.armed_sides.add(side)
                self._power(side, True, check=check)

    def teach(self, enabled, side="both", check=lambda: None):
        """Recording session only; the physical button owns compensated dragging.

        Never disable motors or switch follower roles to imitate teaching.
        Acknowledging the follower's record command alone does not establish
        gravity-compensated dragging; this rig still needs the physical button.
        """
        with self.lock:
            self._check_emergency()
            check()
            if enabled:
                if self.teaching:
                    raise ValueError("请先结束当前录点，再更换侧别")
                sides = selected_sides(side)
                self.point_values("joint", sides)
                self.teaching_sides = set(sides)
                self.teaching = True
                self.armed_sides.difference_update(sides)
            else:
                self.teaching_sides.clear()
                self.teaching = False

    def recover(self):
        """Explicit operator action for arms left in leader mode by old code.

        Preserve motor power. Never replay cached angles or release the motors.
        """
        with self.lock:
            self._check_emergency()
            if self.teaching:
                self.teach(False)
            for side, arm in self.arms.items():
                self.armed_sides.discard(side)
                self._check_emergency()
                arm.set_follower_mode()
            deadline = time.monotonic()+3
            while True:
                self._check_emergency()
                try:
                    self.point_values("joint", ("left", "right"))
                    return
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("恢复从臂模式后仍无反馈，请检查 CAN/设备状态；未释放电机或发送运动目标")
                    time.sleep(.02)

    def send(self, space, target, active_sides=("left", "right"), *, joint_command="move_js"):
        if joint_command not in ("move_js", "move_j"):
            raise ValueError("关节下发接口须为 move_js 或 move_j")
        with self.lock:
            self._check_emergency()
            if self.teaching:
                raise ValueError("手动拖动中禁止执行运动")
            q = finite(target, (14,), "运动目标")
            if np.any((q[[6, 13]] < 0) | (q[[6, 13]] > .1)):
                raise ValueError("夹爪宽度超界")
            # Validate every active side before writing to either side.
            for side in active_sides:
                offset = 0 if side == "left" else 7
                if space == "joint" and (np.any(q[offset:offset+6] < self.limits[:, 0]) or
                                          np.any(q[offset:offset+6] > self.limits[:, 1])):
                    raise ValueError(f"{side} 目标超出 Piper 关节范围")
                self._status(side)
            for side in active_sides:
                self._check_emergency()
                offset = 0 if side == "left" else 7
                arm = self.arms[side]
                (getattr(arm, joint_command) if space == "joint" else arm.move_p)(q[offset:offset+6].tolist())
                self._check_emergency()
                self.grippers[side].move_gripper_m(float(q[offset+6]), force=1.0)

    def home(self, check, progress, close_grippers=False):
        # Same firmware-planned move_j, speed and three settled samples as the
        # recorder. No PiperPair: that class owns leader hardware as well.
        from record import HOME_SPEED, HOME_TIMEOUT_S, HOME_TOLERANCE_DEG, GRIPPER_HOME_TOLERANCE_M
        if self.teaching:
            check()
            self.teach(False, check=check)
        check()
        self.prepare(check=check)
        initial = self.point_values("joint", ("left", "right"))
        distance = max(abs(v) for q in initial.values() for v in q[:6])
        deadline, settled = time.monotonic()+HOME_TIMEOUT_S, 0
        previous = {side: 0.0 for side in self.arms}
        speeds = {}
        try:
            with self.lock:
                for side, arm in self.arms.items():
                    self._status(side)
                    # SDK has a setter only; preserve its cached mode parameter.
                    speeds[side] = int(arm._msg_mode.move_spd_rate_ctrl)
                for arm in self.arms.values():
                    check()
                    self._check_emergency()
                    arm.set_speed_percent(HOME_SPEED)
                    arm.move_j([0.0]*6)
            while True:
                check()
                self._check_emergency()
                with self.lock:
                    messages = {side: self._fresh(arm.get_joint_angles(), f"{side} 回零关节") for side, arm in self.arms.items()}
                    for side in self.arms:
                        self._status(side)
                    fresh = all(m.timestamp > previous[side] for side, m in messages.items())
                    error = max(float(np.max(np.abs(finite(m.msg, (6,), "回零反馈")))) for m in messages.values())
                    settled = settled+1 if fresh and error <= np.deg2rad(HOME_TOLERANCE_DEG) else 0
                    previous = {side: m.timestamp for side, m in messages.items()}
                progress(min(.95, max(0, 1-error/max(distance, 1e-9))))
                if settled >= 3:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("双从臂平滑回零超时，未确认到位")
                time.sleep(.02)
            if close_grippers:
                deadline, settled = time.monotonic()+5, 0
                while settled < 3:
                    check()
                    with self.lock:
                        for side in self.arms:
                            self._check_emergency()
                            self.grippers[side].move_gripper_m(0.0, force=1.0)
                        q = self.point_values("joint", ("left", "right"))
                        settled = settled+1 if all(v[6] <= GRIPPER_HOME_TOLERANCE_M for v in q.values()) else 0
                    if time.monotonic() >= deadline:
                        raise RuntimeError("关节已回零，夹爪闭合未到位")
                    time.sleep(.05)
            progress(1)
        finally:
            # Stopping is owned by the controller, including interrupted homing.
            if not self.emergency:
                self.stop()
                with self.lock:
                    for side, speed in speeds.items():
                        self._check_emergency()
                        self.arms[side].set_speed_percent(speed)

    def stop(self, emergency=False):
        if emergency:
            self.emergency = True
            errors = []
            for side, arm in tuple(self.arms.items()):
                try:
                    arm.electronic_emergency_stop()
                except Exception as exc:
                    errors.append(f"{side}: {exc}")
            if errors:
                raise RuntimeError("急停发送失败：" + "; ".join(errors))
            return
        with self.lock:
            if self.emergency:
                return
            for side in self.armed_sides - self.teaching_sides:
                status = self._fresh(self.arms[side].get_arm_status(), f"{side} 保持状态").msg
                if self._manual_mode(status):
                    continue
                q, _ = self._joint_sample(side)
                q = bounded_joints(q, self.limits, f"{side} 保持关节", HOLD_FEEDBACK_TOLERANCE)
                self._check_emergency()
                self.arms[side].move_js(q.tolist())

    def close(self):
        with self.lock:
            try:
                if self.teaching and not self.emergency:
                    self.teach(False)
                self.stop()
            finally:
                self.resources.close()

    def camera(self, role):
        with self.lock:
            if role not in self.serials:
                raise ValueError("相机不存在")
            if role not in self.cameras:
                from pyorbbecsdk import Context, OBFormat
                from piper_recorder.camera import OrbbecCamera
                self.context = self.context or Context()
                if role in self.depth_roles:
                    camera = DepthCamera(role, self.serials[role], self.context)
                    metadata = camera.metadata
                else:
                    camera = OrbbecCamera(role, self.serials[role], self.context, width=640, height=480,
                                          fps=30, image_size=(640, 480), max_age_s=.5)
                    self.resources.callback(camera.close)
                    camera.start(self.context.query_devices().get_device_by_serial_number(self.serials[role]))
                    profile = camera._pipeline.get_stream_profile_list(camera._sensor_type).get_video_stream_profile(640, 480, OBFormat.RGB, 30)
                    metadata = dict(role=role, serial=self.serials[role], sensor=camera._sensor_type.name,
                                    width=640, height=480, intrinsics=sdk_intrinsics(profile))
                if isinstance(camera, DepthCamera):
                    self.resources.callback(camera.close)
                self.cameras[role] = (camera, metadata)
            camera, metadata = self.cameras[role]
            if isinstance(camera, DepthCamera):
                return camera.read()
            return camera.read().rgb, None, metadata

    def observation(self):
        from piper_recorder.camera import resize_with_padding
        snapshot = self.snapshot()
        obs = {"observation.state": np.array(snapshot["joints"], np.float32),
               "observation.flange_pose": np.r_[snapshot["poses"]["left"], snapshot["poses"]["right"]]}
        for role in self.serials:
            rgb, _, _ = self.camera(role)
            obs[f"observation.images.{role}"] = resize_with_padding(rgb, (224, 224))
        return obs
