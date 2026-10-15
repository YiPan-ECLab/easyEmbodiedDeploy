"""Single motion owner, cancellable workflow runner and asynchronous VLA queues."""

import copy
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..action_contract import POSE_MODES
from ..pose_actions import rpy_matrix, matrix_rotvec, matrix_rpy, rotvec_matrix
from ..target_runtime import validate_live_cameras
from .hardware import (MockHardware, PiperHardware, selected_sides,
                       bounded_joints, HOLD_FEEDBACK_TOLERANCE)
from .models import ModelPool
from .plan import new_plan, validate, interpolate, tangent, require_models


def vector(snapshot, space):
    q = np.asarray(snapshot["joints"], float)
    if space == "joint":
        return q.copy()
    return np.r_[snapshot["poses"]["left"], q[6], snapshot["poses"]["right"], q[13]]


def difference(target, current, space):
    delta = np.asarray(target) - current
    if space == "pose":
        for offset in (0, 7):
            delta[offset+3:offset+6] = matrix_rotvec(rpy_matrix(target[offset+3:offset+6]) @ rpy_matrix(current[offset+3:offset+6]).T)
    return delta


def speed_limits(settings, space):
    one = ([settings["joint_speed"]]*6 if space == "joint" else
           [settings["linear_speed"]]*3 + [settings["angular_speed"]]*3)
    return np.array((one + [settings["gripper_speed"]])*2)


def smooth_target(previous, desired, period, tau, space):
    """Deployment's exponential time-constant filter, with SO(3) pose rotation."""
    if tau == 0:
        return np.asarray(desired).copy()
    alpha = -np.expm1(-period/tau)
    delta = difference(desired, previous, space)
    result = np.asarray(previous) + alpha*delta
    if space == "pose":
        for offset in (0, 7):
            result[offset+3:offset+6] = matrix_rpy(rotvec_matrix(alpha*delta[offset+3:offset+6]) @ rpy_matrix(previous[offset+3:offset+6]))
    return result


def safe_duration(start, end, requested, settings, space):
    delta = np.abs(difference(end, start, space))
    if space == "pose":
        for offset in (0, 3, 7, 10):
            delta[offset:offset+3] = np.linalg.norm(delta[offset:offset+3])
    speed = speed_limits(settings, space)
    # Quintic rest-to-rest peak velocity and acceleration coefficients.
    acceleration = np.minimum(settings["joint_acceleration"], speed * 3)
    return max(float(requested), float(np.max(2*delta/speed)), float(np.sqrt(np.max(6*delta/acceleration))))


class WorkflowController:
    def __init__(self, model_pool=None):
        self.lock = threading.RLock()
        self.hardware = None
        self.models = model_pool or ModelPool()
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="workflow-infer")
        self.thread = None
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.next_event = threading.Event()
        self.status = dict(state="idle", step=-1, progress=0, detail="连接模拟器即可离线编排和试运行", error=None)
        self.events = deque(maxlen=100)
        self.plan = None
        self.last_verification = None
        self.prefills = {}
        self.inferences = set()
        self.inference_cancel = threading.Event()
        self.command = self.space = None
        self.active_model = self.policy_anchor = None
        self.active_sides = ("left", "right")
        self.tick_period = .02
        self.operation = None
        self.pending_unloads = set()
        self.unload_thread = None

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def update(self, **values):
        with self.lock:
            self.status.update(values)

    def event(self, message):
        with self.lock:
            self.events.append(dict(time=time.time(), message=message))

    def snapshot(self):
        with self.lock:
            result = dict(self.status, connected=self.hardware is not None,
                          simulated=self.hardware.simulated if self.hardware else None,
                          teaching=self.hardware.teaching if self.hardware else False,
                          teaching_sides=sorted(self.hardware.teaching_sides) if self.hardware else [],
                          emergency=bool(self.hardware and self.hardware.emergency), operation=self.operation,
                          joint_limits=(self.hardware.limits.tolist() if self.hardware and not self.hardware.simulated else None),
                          busy=self.busy(), verification=self.last_verification, events=list(self.events),
                          pending_unloads=sorted(self.pending_unloads))
            if self.plan and 0 <= self.status["step"] < len(self.plan["steps"]):
                step = self.plan["steps"][self.status["step"]]
                result.update(step_id=step["id"], step_type=step["type"])
        result["models"] = self.models.snapshot()
        return result

    def connect(self, simulated, config=None):
        with self.lock:
            if self.busy() or self.hardware:
                raise ValueError("请先停止工作流并断开当前连接")
            self.hardware = MockHardware(config) if simulated else PiperHardware(config or {})
            self.operation = None
            self.last_verification = None
            self.update(state="idle", step=-1, progress=0, error=None,
                        detail="模拟器已连接" if simulated else "已连接从臂（未回零、未发运动指令）")

    def disconnect(self):
        self.stop()
        thread = self.thread
        if thread:
            thread.join(5)
            if thread.is_alive():
                raise RuntimeError("工作流仍在停止，请稍后断开")
        with self.lock:
            if self.hardware:
                try:
                    self.hardware.close()
                finally:
                    self.hardware = None
            self.operation = None
            self.last_verification = None
            self.update(state="idle", step=-1, progress=0, error=None, detail="已断开设备")

    def teach(self, enabled, side="both"):
        with self.lock:
            if self.busy() or not self.hardware:
                raise ValueError("请先连接设备并停止执行")
            self.hardware.teach(bool(enabled), side)
            self.operation = None
            self.last_verification = None
            self.update(state="idle", step=-1, progress=0,
                        detail="录点中：使用机身示教按钮拖动，网页只读取位置、不释放电机" if enabled else "已结束录点；如仍在机身示教模式，请用机身按钮退出", error=None)

    def capture(self, name, side="both", space="joint"):
        if self.busy() or not self.hardware:
            raise ValueError("请在停止执行后录制关键点")
        if space not in ("joint", "pose"):
            raise ValueError("坐标类型须为关节或末端位姿")
        sides = selected_sides(side)
        arms, feedback = self.hardware.capture_point(space, sides)
        import uuid
        return dict(id=uuid.uuid4().hex, name=name or "未命名点", space=space,
                    arms=arms,
                    source=dict(kind="feedback", simulated=self.hardware.simulated, **feedback), notes="")

    def recover(self):
        with self.lock:
            if self.busy() or not self.hardware:
                raise ValueError("请连接从臂并停止执行后恢复反馈")
            self.hardware.recover()
            self.operation = None
            self.last_verification = None
            self.update(state="idle", step=-1, progress=0,
                        detail="从臂反馈已恢复，未释放电机或发送运动目标", error=None)

    def home(self, close_grippers=False):
        with self.lock:
            if self.busy() or not self.hardware:
                raise ValueError("请先连接从臂并停止执行")
            if self.hardware.emergency:
                raise ValueError("急停后请先在设备侧检查并复位，再重新连接")
            self.operation = "home"
            self.stop_event.clear()
            self.paused.clear()
            self.last_verification = None
            self.update(state="running", step=-1, progress=0, error=None,
                        detail="双从臂平滑回零中；可随时停止或急停")
            self.thread = threading.Thread(target=self._home, args=(close_grippers,), daemon=True)
            self.thread.start()

    def _home(self, close_grippers):
        try:
            self._wait_manual_exit()
            self.hardware.home(self._check, lambda p: self.update(progress=p), close_grippers)
            self._check()
            self.update(state="completed", progress=1, detail="双从臂已平滑回零" +
                        ("，夹爪已闭合" if close_grippers else "，夹爪保持原宽度"))
            self.event("双从臂平滑回零完成")
        except InterruptedError:
            if self.status["state"] != "emergency":
                self.update(state="stopped", detail="回零已停止，保持当前位置")
        except Exception as exc:
            if self.status["state"] != "emergency":
                self.update(state="failed", error=str(exc), detail=f"回零中止：{exc}")
        finally:
            self._hold_after_run()

    def _hold_after_run(self):
        try:
            self.hardware.stop()
        except Exception as exc:
            try:
                self.hardware.stop(emergency=True)
            except Exception as stop_error:
                exc = RuntimeError(f"{exc}；{stop_error}")
            self.update(state="emergency", error=str(exc), detail="保持位置失败，已尝试双从臂阻尼急停；请检查设备")

    def preload(self, plan, identifiers=None):
        selected = new_plan()
        selected["settings"] = copy.deepcopy(plan.get("settings", {}))
        selected["models"] = [m for m in plan["models"] if identifiers is None or m["id"] in identifiers]
        selected = validate(selected)
        require_models(selected["models"], identifiers if identifiers is not None else [m['id'] for m in selected['models']])
        with self.lock:
            if self.busy():
                raise ValueError("执行中模型由工作流管理，请停止后更改配置")
            self.pending_unloads.difference_update(m['id'] for m in selected['models'])
        self.models.preload(selected["models"])

    def release_model(self, identifier):
        """A deleted last card releases weights after the current run, if any."""
        with self.lock:
            if not self.busy():
                self.models.close(identifier)
                return
            self.pending_unloads.add(identifier)
            if self.unload_thread is None or not self.unload_thread.is_alive():
                self.unload_thread = threading.Thread(target=self._release_when_idle, daemon=True)
                self.unload_thread.start()

    def _release_when_idle(self):
        while True:
            with self.lock:
                if not self.pending_unloads:
                    return
                if not self.busy():
                    # Serialize with start(): never unload a model a new run is using.
                    for identifier in self.pending_unloads:
                        self.models.close(identifier)
                    self.pending_unloads.clear()
                    return
                active = self.thread
            active.join()

    def start(self, plan, start_index=0, repetitions=1, home_before=True):
        plan = validate(plan)
        with self.lock:
            if self.busy() or self.hardware is None:
                raise ValueError("请先连接设备，且不能同时执行两个工作流")
            if self.hardware.emergency:
                raise ValueError("急停后请在设备侧检查并复位，再重新连接")
            if not plan["steps"] or not 0 <= int(start_index) < len(plan["steps"]):
                raise ValueError("请添加步骤并选择有效的起始步骤")
            if not 1 <= int(repetitions) <= 10000:
                raise ValueError("重复次数须为 1–10000")
            run_steps = plan["steps"][int(start_index):] if int(repetitions) == 1 else plan["steps"]
            used_models = {s["model"] for s in run_steps if s["type"] == "vla"}
            require_models(plan["models"], used_models)
            if not self.hardware.simulated:
                used_points = {s["point"] for s in run_steps if s["type"] == "point"}
                if any(m["backend"] == "mock" and m["id"] in used_models for m in plan["models"]):
                    raise ValueError("模拟模型不能连接真实机械臂")
                # Reject unreachable joint limits before enabling either arm.
                for point in plan["points"]:
                    if point["space"] == "joint" and point["id"] in used_points:
                        for side, q in point["arms"].items():
                            bounded_joints(q[:6], self.hardware.limits,
                                           f"关键点 {point['name']} / {side} 原始录制值（请核对或重录）")
            self.plan = plan
            self.pending_unloads.difference_update(used_models)
            self.operation = "preparing"
            self.stop_event.clear()
            self.paused.clear()
            self.next_event.clear()
            self.prefills = {}
            self.command = self.space = None
            self.active_model = self.policy_anchor = None
            self.last_verification = None
            self.update(state="running", error=None, progress=0, step=-1, actions=0, queue=0,
                        detail="准备执行；未回零时先平滑回零" if home_before else "从当前位置准备试运行")
            self.thread = threading.Thread(target=self._run, args=(int(start_index), int(repetitions), home_before), daemon=True)
            self.thread.start()

    def pause(self):
        if self.busy() and self.operation in ("home", "preparing"):
            raise ValueError("回零期间请使用停止；再次回零会从当前位置开始")
        if self.busy():
            self.paused.set()
            self.update(state="paused", detail="已请求暂停，保持当前位置")

    def resume(self):
        self.paused.clear()

    def next(self):
        with self.lock:
            index = self.status["step"]
            if not self.busy() or self.operation != "workflow" or self.plan["steps"][index]["type"] not in ("vla", "wait"):
                raise ValueError("只有 VLA 和等待步骤支持手动下一步")
            self.next_event.set()

    def stop(self, emergency=False):
        self.stop_event.set()
        self.paused.clear()
        if emergency and self.hardware:
            # teach/connect may hold the controller lock while awaiting feedback.
            # Send the emergency command before waiting to update UI state.
            try:
                self.hardware.stop(emergency=True)
            except Exception as exc:
                self.update(state="emergency", error=str(exc), detail=f"急停发送异常：{exc}")
                raise
            self.update(state="emergency", detail="已发送双从臂阻尼急停；请在设备侧检查并复位")

    def _check(self):
        if self.stop_event.is_set():
            raise InterruptedError()

    def _pause(self):
        if not self.paused.is_set():
            return False
        self.hardware.stop()
        self._cancel_inferences()
        while self.paused.is_set():
            self._check()
            self.stop_event.wait(.02)
        self._check()
        self.command = None
        self.prefills.clear()
        self.update(state="running", detail="继续执行")
        return True

    def _seed(self, space):
        if self.space != space or self.command is None:
            self.command = vector(self.hardware.snapshot(), space)
            if space == "joint" and not self.hardware.simulated:
                for side, offset in (("left", 0), ("right", 7)):
                    self.command[offset:offset+6] = bounded_joints(
                        self.command[offset:offset+6], self.hardware.limits,
                        f"{side} 当前反馈", HOLD_FEEDBACK_TOLERANCE)
            self.space = space
        return self.command.copy()

    def _send(self, space, value):
        self._check()
        self.hardware.send(space, value, self.active_sides,
                           joint_command=self.plan["settings"].get("joint_command", "move_js"))
        self.space, self.command = space, np.asarray(value).copy()

    def _sleep(self, started):
        self.stop_event.wait(max(0, self.tick_period - (time.monotonic() - started)))

    def _wait_manual_exit(self):
        while True:
            self._check()
            sides = self.hardware.manual_mode_sides()
            if not sides:
                return
            self.update(detail=f"等待 {'、'.join(sides)} 从臂退出机身示教：请操作机身按钮；退出后自动继续，可点击停止取消")
            self.stop_event.wait(.1)

    def _run(self, start_index, repetitions, home_before):
        try:
            self.tick_period = 1 / self.plan["settings"]["control_hz"]
            # Loading continues independently while fixed motions execute.
            selected_steps = self.plan["steps"][start_index:] if repetitions == 1 else self.plan["steps"]
            used = {s["model"] for s in selected_steps if s["type"] == "vla"}
            self.models.preload([m for m in self.plan["models"] if m["id"] in used])
            self._wait_manual_exit()
            if self.hardware.teaching:
                self.hardware.teach(False, check=self._check)
            self._check()
            if home_before:
                q = self.hardware.point_values("joint", ("left", "right"))
                if any(np.max(np.abs(value[:6])) > np.deg2rad(1) for value in q.values()):
                    self.update(detail="执行前双从臂平滑回零；夹爪保持原宽度，模型在后台加载", progress=0)
                    self.event("执行前自动平滑回零")
                    self.hardware.home(self._check, lambda p: self.update(progress=p), False)
                    self._check()
                    self.event("自动回零完成，开始工作流")
                else:
                    self.event("双从臂已在零位，直接开始工作流")
            self._check()
            with self.lock:
                self.operation = "workflow"
            self.update(progress=0, step=start_index)
            if used:
                self.update(detail="准备相机；模型在后台加载")
                self._prepare_cameras()
            self._check()
            points = {p["id"]: p for p in self.plan["points"]}
            active = set()
            for step in selected_steps:
                if step["type"] == "point":
                    active.update(points[step["point"]]["arms"])
                elif step["type"] in ("vla", "gripper"):
                    active.update(selected_sides(step.get("side", "both")))
            if active:
                self.hardware.prepare(tuple(sorted(active)), check=self._check)
            for repeat in range(repetitions):
                self.update(repetition=repeat+1, repetitions=repetitions)
                index = start_index if repeat == 0 else 0
                while index < len(self.plan["steps"]):
                    self._check()
                    self.next_event.clear()
                    step = self.plan["steps"][index]
                    self.update(step=index, progress=0, detail=step.get("name", step["type"]))
                    self.event(f"开始步骤 {index+1}：{step.get('name', step['type'])}")
                    if step["type"] == "point":
                        index = self._points(index)
                    elif step["type"] == "vla":
                        self._vla(index)
                        index += 1
                    elif step["type"] == "wait":
                        elapsed = 0
                        while elapsed < step.get("duration", 1) and not self.next_event.is_set():
                            self._check()
                            self._pause()
                            t = time.monotonic()
                            self._prefetch_next(index, step.get("duration", 1)-elapsed)
                            self.stop_event.wait(self.tick_period)
                            elapsed += time.monotonic()-t
                        index += 1
                    else:
                        self.active_sides = ("left", "right") if step.get("side", "both") == "both" else (step["side"],)
                        start = self._seed("joint")
                        end = start.copy()
                        for side, offset in (("left", 6), ("right", 13)):
                            if step.get("side", "both") in (side, "both"):
                                end[offset] = step["width"]
                        self._motion(start, end, step.get("duration", 1), self.space, index)
                        self._verify(end, self.space)
                        index += 1
            self.update(state="completed", progress=1, detail="工作流执行完成，保持当前位置")
            self.event("工作流完成")
        except InterruptedError:
            if self.status["state"] != "emergency":
                self.update(state="stopped", detail="已停止，保持当前位置")
        except Exception as exc:
            if self.status["state"] != "emergency":
                self.update(state="failed", error=str(exc), detail=f"执行中止：{exc}")
            self.event(f"执行失败：{exc}")
        finally:
            self._cancel_inferences()
            self.active_model = self.policy_anchor = None
            self._hold_after_run()

    def _prepare_cameras(self):
        deadline = time.monotonic()+15
        while True:
            self._check()
            try:
                self.hardware.observation()
                return
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                self.stop_event.wait(.1)

    def _point_target(self, point, previous):
        end = previous.copy()
        for side, q in point["arms"].items():
            offset = 0 if side == "left" else 7
            end[offset:offset+7] = q
        return end

    def _points(self, index):
        table = {p["id"]: p for p in self.plan["points"]}
        steps = self.plan["steps"]
        space = table[steps[index]["point"]]["space"]
        knots = [self._seed(space)]
        durations, end_index = [], index
        while end_index < len(steps) and steps[end_index]["type"] == "point":
            step = steps[end_index]
            point = table[step["point"]]
            if point["space"] != space:
                break
            end = self._point_target(point, knots[-1])
            duration = safe_duration(knots[-1], end, step.get("duration", 1), self.plan["settings"], space)
            knots.append(end)
            durations.append(duration)
            end_index += 1
            if not step.get("blend", True) or space != "joint":
                break
        velocities = [np.zeros(14) for _ in knots]
        if space == "joint":
            for i in range(1, len(knots)-1):
                velocities[i] = tangent(knots[i-1], knots[i], knots[i+1], durations[i-1], durations[i])
        for i, duration in enumerate(durations):
            current = index+i
            self.active_sides = tuple(table[steps[current]["point"]]["arms"])
            if i:
                self.event(f"开始步骤 {current+1}：{steps[current].get('name', '关键点运动')}")
            self.update(step=current, detail=steps[current].get("name", "关键点运动"))
            if self._motion(knots[i], knots[i+1], duration, space, current,
                            velocities[i], velocities[i+1]):
                velocities[i+1] = np.zeros(14)
        self.active_sides = tuple({side for step in steps[index:end_index] for side in table[step["point"]]["arms"]})
        self._verify(knots[-1], space)
        return end_index

    def _motion(self, start, end, duration, space, index, v0=None, v1=None):
        duration = safe_duration(start, end, duration, self.plan["settings"], space)
        elapsed = 0
        resumed = False
        while elapsed < duration:
            self._check()
            if self._pause():
                resumed = True
                start = self._seed(space)
                duration = safe_duration(start, end, duration-elapsed, self.plan["settings"], space)
                elapsed, v0, v1 = 0, None, None
            t = time.monotonic()
            self._send(space, interpolate(start, end, elapsed/duration, space, v0, v1, duration))
            self.update(progress=elapsed/duration, actual_duration=duration)
            self._prefetch_next(index, duration-elapsed)
            self._sleep(t)
            elapsed += min(time.monotonic()-t, self.tick_period*2)
        self._send(space, end)
        return resumed

    def _verify(self, target, space):
        deadline = time.monotonic()+8
        while True:
            self._check()
            actual = vector(self.hardware.snapshot(), space)
            delta = np.abs(difference(target, actual, space))
            tolerance = np.array(([.03]*6+[.006])*2 if space == "joint" else ([.005]*3+[.08]*3+[.006])*2)
            for side, offset in (("left", 0), ("right", 7)):
                if side not in self.active_sides:
                    tolerance[offset:offset+7] = np.inf
            report = dict(space=space, target=np.asarray(target).tolist(), actual=actual.tolist(),
                          error=delta.tolist(), reached=bool(np.all(delta <= tolerance)), simulated=self.hardware.simulated)
            with self.lock:
                self.last_verification = report
            if report["reached"]:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("实际反馈未到达关键点；请查看到位误差，后续步骤未执行")
            if self._pause():
                deadline = time.monotonic()+8
            self._send(space, target)
            self.stop_event.wait(self.tick_period)

    def _observation(self, model):
        obs = self.hardware.observation()
        space = "pose" if model.config["action_mode"] in POSE_MODES else "joint"
        # Delta joint models integrate on the last policy target, before the
        # hardware-rate transition filter, as the existing deployer does.
        obs["command_anchor"] = (self.policy_anchor.copy() if self.active_model == model.config["id"] and self.policy_anchor is not None
                                  else self.command.copy() if self.space == space and self.command is not None
                                  else vector(self.hardware.snapshot(), space))
        if model.contract and not self.hardware.simulated:
            cameras = {role: self.hardware.camera(role)[2] for role in self.hardware.serials}
            validate_live_cameras(model.contract, {"cameras": cameras})
        return obs

    def _submit(self, index, prefix=None, prefix_start=1):
        step = self.plan["steps"][index]
        model = self.models.get(step["model"])
        if model.stage == "failed":
            raise RuntimeError(f"模型 {model.config.get('name', step['model'])} 加载失败：{model.detail}")
        if model.stage != "ready":
            return None
        obs = self._observation(model)
        task = step.get("task") or model.config.get("task", "")
        future = self.executor.submit(model.infer, obs, task, prefix, prefix_start,
                                      cancel_event=self.inference_cancel)
        with self.lock:
            self.inferences.add(future)
        future.add_done_callback(self._inference_done)
        return future

    def _inference_done(self, future):
        with self.lock:
            self.inferences.discard(future)

    def _cancel_inferences(self):
        # Running GPU kernels finish normally; queued work must not start after
        # pause/stop or leak into a subsequent workflow run.
        with self.lock:
            self.inference_cancel.set()
            self.inference_cancel = threading.Event()
            pending = list(self.inferences)
            self.inferences.clear()
            self.prefills.clear()
        for future in pending:
            future.cancel()

    def _prefetch_next(self, index, remaining):
        future_index = index+1
        if remaining > self.plan["settings"]["prefetch_s"] or future_index >= len(self.plan["steps"]):
            return
        step = self.plan["steps"][future_index]
        if step["type"] != "vla" or future_index in self.prefills:
            return
        future = self._submit(future_index)
        if future is not None:
            self.prefills[future_index] = future

    def _vla(self, index):
        old_period = self.tick_period
        config = self.models.get(self.plan["steps"][index]["model"]).config
        self.tick_period = 1/float(config.get("control_hz", self.plan["settings"]["control_hz"]))
        try:
            self._vla_body(index)
        finally:
            self.tick_period = old_period

    def _vla_body(self, index):
        step = self.plan["steps"][index]
        model = self.models.get(step["model"])
        settings = self.plan["settings"]
        hz = float(model.config.get("policy_hz", 10))
        rtc = model.config.get("inference_mode", "rtc" if model.config.get("rtc", True) else "sync") == "rtc"
        threshold = int(model.config.get("rtc_queue_threshold",40)) if rtc else 0
        tau = float(model.config.get("smoothing_time_constant", .1))
        space = "pose" if model.config["action_mode"] in POSE_MODES else "joint"
        start = self._seed(space)
        self.active_sides = ("left", "right") if step.get("side", "both") == "both" else (step["side"],)
        self.active_model, self.policy_anchor = step["model"], start.copy()
        desired, velocity = start.copy(), np.zeros(14)
        filtered = start.copy()
        queue = deque()
        pending = self.prefills.pop(index, None)
        pending_consumed = 0
        pending_prefix = False
        conditioned_delay = discarded = 0
        submitted_at = time.monotonic()
        next_policy = elapsed = actions = blend_time = 0
        transition = settings["transition_s"]
        until, value = step.get("until", "duration"), step.get("value", 10)
        started_actions = False
        last_tick = time.monotonic()
        while not self.next_event.is_set():
            self._check()
            if self._pause():
                queue.clear()
                pending = None
                pending_consumed = 0
                start = self._seed(space)
                desired, velocity, blend_time = start.copy(), np.zeros(14), 0
                filtered = start.copy()
                pending_prefix = False
                next_policy = 0
                self.policy_anchor = start.copy()
                last_tick = time.monotonic()
            t = time.monotonic()
            dt = t-last_tick
            last_tick = t
            if pending is not None and pending.done():
                chunk = pending.result()
                # Prefix was anchored when the request was submitted. Remove only
                # actions actually consumed since then, never wall-clock guesses.
                elapsed_frames = int(np.ceil((time.monotonic()-submitted_at)*hz))
                if rtc and model.config.get("rtc_mode") == "trained" and pending_prefix and elapsed_frames > conditioned_delay:
                    discarded += 1
                    if discarded >= 5:
                        raise RuntimeError("trained RTC 连续 5 次推理延迟超过前缀覆盖范围，请检查模型推理耗时与 policy Hz")
                else:
                    discarded = 0
                    if not rtc:
                        chunk = chunk[:model.inference.get("action_steps",50)]
                    queue = deque(chunk[min(pending_consumed, len(chunk)):])
                pending, pending_consumed = None, 0
            if pending is None and len(queue) <= threshold:
                prefix = np.array(queue) if rtc and queue else None
                conditioned_delay = max(1, int(np.ceil(max(model.latencies, default=0)*hz))) if prefix is not None else 0
                if prefix is not None:
                    conditioned_delay = min(conditioned_delay, len(prefix), int(model.config.get("rtc_horizon",20)))
                    if model.config.get("rtc_mode") == "trained":
                        conditioned_delay = min(conditioned_delay, model.inference.get("training_max_delay",0))
                submitted_at = time.monotonic()
                pending = self._submit(index, prefix, conditioned_delay)
                pending_prefix = prefix is not None
                pending_consumed = 0
            if time.monotonic() >= next_policy and queue:
                desired = queue.popleft()
                for side, offset in (("left", 0), ("right", 7)):
                    if side not in self.active_sides:
                        desired[offset:offset+7] = start[offset:offset+7]
                desired[[6, 13]] = np.clip(desired[[6, 13]], 0, .1)
                if space == "joint" and not self.hardware.simulated:
                    for offset in (0, 7):
                        desired[offset:offset+6] = np.clip(desired[offset:offset+6], self.hardware.limits[:, 0], self.hardware.limits[:, 1])
                actions += 1
                self.policy_anchor = desired.copy()
                started_actions = True
                if pending is not None:
                    pending_consumed += 1
                next_policy = time.monotonic() + 1/hz
            if not started_actions:
                self.update(detail=f"等待 {model.config.get('name', step['model'])} 就绪及首个动作", progress=0)
            else:
                # Smooth from the last controller's actual commanded target.
                blended = interpolate(start, desired, min(1, blend_time/transition), space)
                filtered = smooth_target(filtered, blended, self.tick_period, tau, space)
                self._track(filtered, velocity, space)
                blend_time += self.tick_period
                elapsed += dt
                self.update(detail=f"VLA：{model.config.get('name', step['model'])}", actions=actions,
                            queue=len(queue), progress=(min(1, (elapsed if until == "duration" else actions)/value) if until != "manual" else 0))
                remaining = value-elapsed if until == "duration" else ((value-actions)/hz if until == "actions" else float("inf"))
                self._prefetch_next(index, remaining)
                if (until == "duration" and elapsed >= value) or (until == "actions" and actions >= value):
                    break
            self._sleep(t)
        # Brake to zero velocity before a different motion-space controller.
        accel = np.minimum(settings["joint_acceleration"], speed_limits(settings, space)*3)
        braking_s = float(np.max(np.abs(velocity)/accel))
        for _ in range(int(np.ceil(braking_s/self.tick_period)) + 1):
            self._check()
            t = time.monotonic()
            self._track(self.command.copy(), velocity, space)
            self._sleep(t)
        self.active_model = self.policy_anchor = None

    def _track(self, desired, velocity, space):
        current = self.command
        delta = difference(desired, current, space)
        maximum = speed_limits(self.plan["settings"], space)
        accel = np.minimum(self.plan["settings"]["joint_acceleration"], maximum*3)
        wanted = np.sign(delta)*np.minimum(maximum, np.sqrt(2*accel*np.abs(delta)))
        if space == "pose":
            for offset in (0, 3, 7, 10):
                norm = np.linalg.norm(wanted[offset:offset+3])
                wanted[offset:offset+3] *= min(1, maximum[offset]/max(norm, 1e-12))
        dv = np.clip(wanted-velocity, -accel*self.tick_period, accel*self.tick_period)
        if space == "pose":
            for offset in (0, 3, 7, 10):
                norm = np.linalg.norm(dv[offset:offset+3])
                dv[offset:offset+3] *= min(1, accel[offset]*self.tick_period/max(norm, 1e-12))
        velocity += dv
        movement = velocity*self.tick_period
        target = current + movement
        if space == "pose":
            for offset in (0, 7):
                target[offset+3:offset+6] = matrix_rpy(rotvec_matrix(movement[offset+3:offset+6]) @ rpy_matrix(current[offset+3:offset+6]))
        target[[6, 13]] = np.clip(target[[6, 13]], 0, .1)
        if space == "joint" and not self.hardware.simulated:
            for offset in (0, 7):
                bounded = np.clip(target[offset:offset+6], self.hardware.limits[:, 0], self.hardware.limits[:, 1])
                velocity[offset:offset+6][bounded != target[offset:offset+6]] = 0
                target[offset:offset+6] = bounded
        self._send(space, target)

    def close(self):
        self.disconnect()
        self.models.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
