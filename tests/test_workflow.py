"""Offline workflow contracts, geometry, cancellation and real worker IPC."""

import copy
import json
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from easy_embodied_deploy.workflow.plan import (new_plan, validate, PlanStore, interpolate,
                                               tangent, camera_point)
from easy_embodied_deploy.workflow.hardware import MockHardware, PiperHardware
from easy_embodied_deploy.workflow.models import ModelPool, ModelProcess
from easy_embodied_deploy.workflow.runtime import WorkflowController, smooth_target


def point_plan():
    plan = new_plan()
    plan["points"] = [dict(id="home", name="基准点", space="joint",
                           arms=dict(left=[.01, .2, -.2, 0, 0, -.785, .04]))]
    plan["steps"] = [dict(id="first", type="point", name="去基准点", point="home", duration=.1)]
    return plan


def wait_for(predicate, timeout=8):
    end = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError("operation timed out")
        time.sleep(.02)


class GeometryTests(unittest.TestCase):
    def test_joint_command_defaults_and_validation(self):
        plan = point_plan()
        plan['settings'].pop('joint_command')
        self.assertEqual(validate(plan)['settings']['joint_command'], 'move_js')
        plan['settings']['joint_command'] = 'move_j'
        self.assertEqual(validate(plan)['settings']['joint_command'], 'move_j')
        plan['settings']['joint_command'] = 'disable'
        with self.assertRaisesRegex(ValueError, '关节下发接口'):
            validate(plan)

    def test_reference_reuse_roundtrip_and_invalid_dangling_reference(self):
        plan = point_plan()
        plan["steps"].append(dict(id="return", type="point", point="home", duration=.1))
        with tempfile.TemporaryDirectory() as directory:
            store = PlanStore(directory)
            identifier = store.save(plan)
            self.assertEqual(store.load(identifier), validate(plan))
            self.assertEqual(store.listing()[0]["steps"], 2)
            with self.assertRaises(ValueError):
                store.load("../outside")
        plan["points"] = []
        with self.assertRaises(ValueError):
            validate(plan)

    def test_quintic_knots_and_short_rotation_arc(self):
        a, b, c = np.zeros(14), np.ones(14)*.1, np.ones(14)*.2
        v = tangent(a,b,c,2,2)
        epsilon=1e-5
        before=(interpolate(a,b,1,"joint",None,v,2)-interpolate(a,b,1-epsilon,"joint",None,v,2))/(2*epsilon)
        after=(interpolate(b,c,epsilon,"joint",v,None,2)-b)/(2*epsilon)
        np.testing.assert_allclose(before,after,atol=1e-6)
        np.testing.assert_allclose(interpolate(a,b,1),b,atol=1e-12)
        a,b=np.zeros(14),np.zeros(14)
        a[5],b[5]=np.deg2rad(179),np.deg2rad(-179)
        mid=interpolate(a,b,.5,"pose")
        self.assertAlmostEqual(abs(mid[5]),np.pi,places=6)

    def test_depth_pixel_uses_frozen_wrist_and_tcp_transform(self):
        hardware=MockHardware()
        rgb,depth,camera=hardware.camera("left_eye_in_hand")
        snapshot=dict(hardware.snapshot(),depth=depth,camera=camera)
        matrix=np.eye(4);matrix[0,3]=.05
        item=dict(job="left_wrist",camera=camera,mount_id="test",
                  result=dict(transform="T_left_flange_left_camera",matrix=matrix.tolist(),accepted=True))
        bundle=dict(schema="easy_embodied_camera_calibration",calibrations=[item])
        p=camera_point(snapshot,[320,240],bundle,"left",[0,0,.1],[0,0,.05])
        np.testing.assert_allclose(p["arms"]["left"][:3],[.3,.1,.65])
        calibrated=copy.deepcopy(bundle)
        calibrated["calibrations"][0]["camera"]["intrinsics"]["K"][0][0]=300
        p=camera_point(snapshot,[380,240],calibrated,"left",[0,0,.1],[0,0,.05])
        self.assertAlmostEqual(p["arms"]["left"][0],.38)
        snapshot["depth"][240,320]=0
        with self.assertRaises(ValueError):camera_point(snapshot,[320,240],bundle,"left",[0]*3,[0]*3)
        snapshot["depth"][240,320]=.4
        bad=copy.deepcopy(bundle);bad["calibrations"][0]["camera"]["serial"]="different"
        with self.assertRaises(ValueError):camera_point(snapshot,[320,240],bad,"left",[0]*3,[0]*3)

    def test_busy_lease_rejects_before_sdk_import_or_connection(self):
        @contextmanager
        def unavailable():
            raise RuntimeError("device in use")
            yield
        with patch("easy_embodied_deploy.workflow.hardware.hardware_lease",unavailable):
            with self.assertRaisesRegex(RuntimeError,"device in use"):
                PiperHardware({})


class HardwareAdapterTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls
        def message(value):
            return SimpleNamespace(timestamp=time.time(), msg=value)
        class Arm:
            def __init__(self, side):
                self.side, self.enabled, self.powered = side, 0, True
                self.q = [0, .2, -.2, 0, 0, 0]
                self.feedback = True
                self.stale_driver = False
                self._msg_mode = SimpleNamespace(move_spd_rate_ctrl=100)
            @property
            def _parser(self):
                return SimpleNamespace(**{name: message(SimpleNamespace(**{
                    f'joint_{i+1}': self.q[i], f'joint_{i+2}': self.q[i+1]})) if self.feedback else None
                    for name, i in (('joint_12',0), ('joint_34',2), ('joint_56',4))})
            def get_joint_angles(self): return message(list(self.q)) if self.feedback else None
            def get_flange_pose(self): return message([.25, 0, .3, 0, 0, 0]) if self.feedback else None
            def get_arm_status(self): return message(SimpleNamespace(arm_status=0)) if self.feedback else None
            def get_driver_states(self, i):
                m = message(SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=self.powered)))
                if self.stale_driver: m.timestamp -= 100
                return m
            def move_js(self, q): calls.append((self.side, 'joint', list(q))); self.q = list(q)
            def move_j(self, q): calls.append((self.side, 'planned', list(q))); self.q = list(q)
            def move_p(self, q): calls.append((self.side, 'pose', q))
            def set_follower_mode(self): calls.append((self.side, 'follower')); self.feedback = True
            def set_speed_percent(self, speed): self._msg_mode.move_spd_rate_ctrl = speed; calls.append((self.side, 'speed', speed))
            def disable(self): self.powered = False; calls.append((self.side, 'disable')); return True
            def enable(self):
                self.enabled += 1; calls.append((self.side, 'enable'))
                self.powered = self.enabled >= 2
                return self.powered
            def electronic_emergency_stop(self): calls.append((self.side, 'emergency'))
        class Gripper:
            def __init__(self, side): self.width, self.side = .04, side
            def get_gripper_status(self): return message(SimpleNamespace(mode='width', value=self.width))
            def move_gripper_m(self, value, force): self.width = value; calls.append((self.side, 'gripper', value))
        rig = PiperHardware.__new__(PiperHardware)
        rig.lock = __import__('threading').RLock()
        rig.resources = ExitStack(); rig.teaching = rig.emergency = False
        rig.closed = False
        rig.teaching_sides, rig.armed_sides = set(), set()
        rig.channels = dict(left='piper-lo', right='piper-ro')
        rig.arms = {s: Arm(s) for s in ('left', 'right')}
        rig.grippers = {s: Gripper(s) for s in rig.arms}
        rig.limits = np.array([[-3, 3]]*6)
        self.rig = rig
        self.record = SimpleNamespace(HOME_SPEED=30, HOME_TIMEOUT_S=1,
                                      HOME_TOLERANCE_DEG=1, GRIPPER_HOME_TOLERANCE_M=.002)

    def test_joint_interface_routes_sdk_and_preserves_side_and_pose(self):
        target = np.array([.1, .2, -.2, 0, 0, 0, .04]*2)
        for interface, expected in [('move_js', 'joint'), ('move_j', 'planned')]:
            self.calls.clear()
            self.rig.send('joint', target, ('right',), joint_command=interface)
            self.assertEqual([c[:2] for c in self.calls], [('right', expected), ('right', 'gripper')])
        self.calls.clear()
        self.rig.send('pose', target, ('left',), joint_command='move_j')
        self.assertEqual([c[:2] for c in self.calls], [('left', 'pose'), ('left', 'gripper')])
        self.calls.clear()
        with self.assertRaises(ValueError):
            self.rig.send('joint', target, joint_command='disable')
        self.assertEqual(self.calls, [])

    def test_recording_session_never_changes_power_mode_or_targets(self):
        rig = self.rig
        rig.prepare(('left',))
        self.assertEqual(rig.armed_sides, {'left'})
        self.assertEqual(rig.arms['left'].enabled, 2)
        self.assertEqual(self.calls[0][:2], ('left', 'joint'))
        self.assertFalse(any(c[0] == 'right' for c in self.calls))
        self.calls.clear()
        rig.send('pose', np.array([.3, 0, .3, 0, 0, 0, .04]*2), ('left',))
        self.assertEqual([c[1] for c in self.calls], ['pose', 'gripper'])
        self.calls.clear()
        rig.teach(True, 'left')
        self.assertEqual(self.calls, [])
        rig.arms['left'].q[0] = .35
        # Neither opposite-side feedback nor flange data gates a joint capture.
        rig.arms['right'].feedback = False
        rig.arms['left'].get_flange_pose = lambda: None
        recorded = rig.point_values('joint', ('left',))
        self.assertEqual(recorded['left'][0], .35)
        self.assertEqual(recorded['left'][6], .04)
        self.calls.clear()
        rig.teach(False)
        self.assertFalse(rig.teaching)
        self.assertEqual(self.calls, [])
        self.assertFalse(any(c[1] == 'disable' for c in self.calls))
        self.assertFalse(any(c[0] == 'right' for c in self.calls))

    def test_stale_feedback_is_not_accepted_or_replaced_with_old_targets(self):
        self.rig.arms['left'].stale_driver = True
        with self.assertRaisesRegex(RuntimeError, '六轴确认'):
            self.rig._power('left', True, timeout=.03)
        self.rig.arms['left'].feedback = False
        with self.assertRaises(RuntimeError): self.rig.point_values('joint', ('left',))
        self.calls.clear()
        with self.assertRaises(RuntimeError): self.rig.prepare(('left',))
        self.assertEqual(self.calls, [])

    def test_capture_uses_complete_measured_groups_and_keeps_original_values(self):
        arm = self.rig.arms['right']
        raw = np.deg2rad([4.555,-2.231,-41.077,11.353,39.429,-17.841]).tolist()
        arm.q = raw
        # The SDK aggregate can look fresh while containing different cached axes.
        arm.get_joint_angles = lambda: SimpleNamespace(timestamp=time.time(), msg=[0]*6)
        values, source = self.rig.capture_point('joint', ('right',))
        np.testing.assert_array_equal(values['right'][:6], raw)
        np.testing.assert_array_equal(source['feedback']['right']['joint_angles'], raw)
        self.assertEqual(source['feedback']['right']['channel'], 'piper-ro')
        self.assertEqual(len(source['feedback']['right']['received_at_s']), 3)
        self.assertEqual(self.calls, [])

    def test_missing_old_and_skewed_groups_cannot_make_a_recorded_point(self):
        original = self.rig.arms['right']
        for fault in ('missing', 'old', 'skew'):
            groups = original._parser
            if fault == 'missing': groups.joint_12 = None
            else: groups.joint_12.timestamp -= 1 if fault == 'old' else .09
            # Aggregate is still fresh; this must not hide the invalid group.
            self.rig.arms['right'] = SimpleNamespace(_parser=groups,
                get_joint_angles=lambda: SimpleNamespace(timestamp=time.time(), msg=[0]*6))
            with self.subTest(fault=fault):
                with self.assertRaisesRegex(RuntimeError, '六轴反馈'):
                    self.rig.capture_point('joint', ('right',))
        self.rig.arms['right'] = original

    def test_prepare_failure_never_releases_motors(self):
        original = self.rig._power
        def fail_right(side, enabled, **kw):
            if side == 'right': raise RuntimeError('right missing driver feedback')
            return original(side, enabled, **kw)
        self.rig._power = fail_right
        with self.assertRaises(RuntimeError): self.rig.prepare()
        self.assertFalse(self.rig.teaching)
        self.assertFalse(any(c[1] == 'disable' for c in self.calls))

    def test_explicit_recovery_restores_follower_feedback_without_targets(self):
        for arm in self.rig.arms.values(): arm.feedback = False
        self.rig.recover()
        self.assertFalse(self.rig.teaching)
        self.assertFalse(any(c[1] in ('joint', 'planned', 'enable') for c in self.calls))
        for side in ('left', 'right'):
            calls = [c[1] for c in self.calls if c[0] == side]
            self.assertIn('follower', calls)
            self.assertNotIn('disable', calls)
        self.assertEqual(len(self.rig.snapshot()['joints']), 14)

    def test_home_uses_firmware_planning_both_followers_and_restores_speed(self):
        progress = []
        self.rig.teach(True, 'left')
        with patch.dict(sys.modules, record=self.record):
            self.rig.home(lambda: None, progress.append)
        self.assertFalse(self.rig.teaching)
        self.assertEqual([(c[0], c[2]) for c in self.calls if c[1] == 'planned'],
                         [('left', [0.0]*6), ('right', [0.0]*6)])
        self.assertFalse(any(c[1] == 'gripper' for c in self.calls))
        self.assertEqual(progress[-1], 1)
        self.assertTrue(all(a._msg_mode.move_spd_rate_ctrl == 100 for a in self.rig.arms.values()))
        with patch.dict(sys.modules, record=self.record):
            self.rig.home(lambda: None, progress.append, True)
        self.assertTrue(all(g.width == 0 for g in self.rig.grippers.values()))

    def test_record_then_home_rechecks_enable_without_reconnecting(self):
        self.rig.prepare()
        for _ in range(2):
            # Physical teaching can happen without the web teach() button.
            arm = self.rig.arms['right']
            arm.q[0] = .08
            arm.powered = False
            count = arm.enabled
            self.rig.capture_point('joint', ('right',))
            original = arm.move_j
            arm.move_j = lambda q: original(q) if arm.powered else None
            try:
                with patch.dict(sys.modules, record=self.record):
                    self.rig.home(lambda: None, lambda p: None)
            finally:
                arm.move_j = original
            self.assertGreater(arm.enabled, count)
            self.assertTrue(arm.powered)
            self.assertEqual(arm.q, [0.0]*6)
        self.assertFalse(any(c[1] == 'disable' for c in self.calls))

    def test_physical_button_stop_allows_home_with_latched_teach_control_mode(self):
        # Actual right follower 0x2A1: 02 00 01 02 01 00 00 00.
        state = SimpleNamespace(ctrl_mode=2, arm_status=0, teach_status=1)
        self.rig.arms['right'].get_arm_status = lambda: SimpleNamespace(
            timestamp=time.time(), msg=state)
        controller = WorkflowController()
        controller.hardware = self.rig
        self.addCleanup(controller.close)
        self.rig.teach(True, 'right')
        with patch.dict(sys.modules, record=self.record):
            controller.home()
            wait_for(lambda: '机身示教' in controller.status['detail'])
            self.assertEqual(self.calls, [])
            state.teach_status = 2  # physical button ends recording
            wait_for(lambda: not controller.busy())
        self.assertEqual(controller.status['state'], 'completed', controller.snapshot())
        self.assertFalse(self.rig.teaching)
        self.assertEqual([c[0] for c in self.calls if c[1] == 'planned'], ['left', 'right'])
        self.assertFalse(any(c[1] in ('disable', 'follower') for c in self.calls))

    def test_manual_exit_does_not_override_active_teaching_or_faults(self):
        for ctrl, arm, teach in ((2,0,1), (1,0,1), (2,11,2), (2,8,2), (6,0,2), (2,0,0)):
            with self.subTest(ctrl=ctrl, arm=arm, teach=teach):
                state = SimpleNamespace(ctrl_mode=ctrl, arm_status=arm, teach_status=teach)
                self.assertTrue(PiperHardware._manual_mode(state))
                self.rig.arms['right'].get_arm_status = lambda: SimpleNamespace(
                    timestamp=time.time(), msg=state)
                with self.assertRaisesRegex(RuntimeError, '示教'):
                    self.rig.prepare(('right',))
                self.assertEqual(self.calls, [])

    def test_home_cancel_and_emergency_do_not_restart_motion(self):
        def check():
            if any(c[1] == 'planned' for c in self.calls): raise InterruptedError()
        with patch.dict(sys.modules, record=self.record):
            with self.assertRaises(InterruptedError): self.rig.home(check, lambda _: None)
        self.assertTrue(any(c[1] == 'joint' for c in self.calls))
        self.rig.stop(emergency=True)
        self.calls.clear()
        with self.assertRaises(RuntimeError): self.rig.teach(False)
        with self.assertRaises(RuntimeError): self.rig.prepare()
        self.rig.stop()
        self.assertEqual(self.calls, [])

    def test_emergency_attempts_both_sides_even_if_one_send_fails(self):
        def fail(): raise RuntimeError('CAN down')
        self.rig.arms['left'].electronic_emergency_stop = fail
        with self.assertRaisesRegex(RuntimeError, 'CAN down'): self.rig.stop(emergency=True)
        self.assertIn(('right', 'emergency'), self.calls)
        self.assertTrue(self.rig.emergency)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.controller=WorkflowController()
        self.controller.connect(True)
        self.addCleanup(self.controller.close)

    def test_selected_joint_interface_reaches_runtime_sender(self):
        plan = point_plan()
        plan['settings']['joint_command'] = 'move_j'
        hardware = self.controller.hardware
        with patch.object(hardware, 'send', wraps=hardware.send) as send:
            self.controller.start(plan, home_before=False)
            wait_for(lambda: not self.controller.busy())
            self.assertEqual(self.controller.status['state'], 'completed', self.controller.snapshot())
            self.assertGreater(send.call_count, 1)
            self.assertTrue(all(call.kwargs.get('joint_command') == 'move_j' for call in send.call_args_list))

    def test_execute_automatically_homes_before_steps_and_preserves_recorded_points(self):
        plan = point_plan()
        before = copy.deepcopy(plan)
        self.controller.teach(True, 'left')
        homes = []
        original = self.controller.hardware.home
        def home(*args):
            homes.append(True)
            return original(*args)
        self.controller.hardware.home = home
        self.controller.start(plan)
        wait_for(lambda: not self.controller.busy(), timeout=10)
        self.assertEqual(self.controller.status['state'], 'completed', self.controller.snapshot())
        self.assertEqual(homes, [True])
        self.assertFalse(self.controller.hardware.teaching)
        self.assertEqual(plan, before)
        self.assertTrue(any('自动回零完成' in e['message'] for e in self.controller.events))

    def test_already_home_skips_homing_and_waiting_manual_mode_is_cancellable(self):
        self.controller.hardware.joints[:] = 0
        self.controller.hardware.home = lambda *args: self.fail('already at zero')
        plan = point_plan()
        self.controller.start(plan)
        wait_for(lambda: not self.controller.busy())
        self.assertEqual(self.controller.status['state'], 'completed')
        self.controller.hardware.manual_mode_sides = lambda: ['right']
        self.controller.start(plan)
        wait_for(lambda: '机身示教' in self.controller.status['detail'])
        count = len(self.controller.hardware.commands)
        self.controller.stop()
        wait_for(lambda: not self.controller.busy())
        self.assertEqual(self.controller.status['state'], 'stopped')
        self.assertEqual(len(self.controller.hardware.commands), count)

    def test_old_out_of_range_recording_is_not_silently_changed(self):
        self.controller.hardware.simulated = False
        self.controller.hardware.limits = np.deg2rad([
            [-150,150], [0,180], [-170,0], [-100,100], [-70,70], [-120,120]])
        plan = point_plan()
        plan['points'][0]['name'] = '右臂2'
        plan['points'][0]['arms'] = {'right': [*np.deg2rad([4.555,-2.231,-41.077,11.353,39.429,-17.841]), .0011]}
        plan['points'][0]['source'] = {'kind':'feedback', 'simulated':False}
        before = copy.deepcopy(plan)
        with self.assertRaisesRegex(ValueError, '右臂2.*J2=-2.231'):
            self.controller.start(plan)
        self.assertEqual(plan, before)
        self.assertEqual(self.controller.hardware.commands, [])

    def test_home_from_teaching_preserves_points_and_can_close_grippers(self):
        point = self.controller.capture("基准点", "left")
        original = copy.deepcopy(point)
        self.controller.teach(True, "left")
        self.controller.home()
        wait_for(lambda: not self.controller.busy())
        self.assertEqual(self.controller.status["state"], "completed")
        self.assertFalse(self.controller.hardware.teaching)
        np.testing.assert_array_equal(self.controller.hardware.joints[:6], np.zeros(6))
        self.assertEqual(self.controller.hardware.joints[6], .04)
        self.assertEqual(point, original)
        self.controller.home(True)
        wait_for(lambda: not self.controller.busy())
        self.assertEqual(self.controller.hardware.joints[6], 0)

    def test_home_can_stop_and_emergency_remains_latched(self):
        self.controller.home()
        wait_for(lambda: len(self.controller.hardware.commands) > 2)
        with self.assertRaises(ValueError): self.controller.pause()
        self.controller.stop()
        wait_for(lambda: not self.controller.busy())
        self.assertEqual(self.controller.status["state"], "stopped")
        self.controller.home()
        self.controller.stop(emergency=True)
        wait_for(lambda: not self.controller.busy())
        self.assertEqual(self.controller.status["state"], "emergency")
        with self.assertRaises(ValueError): self.controller.home()
        with self.assertRaises(ValueError): self.controller.start(point_plan())

    def test_capture_pose_while_dragging_and_invalid_space_is_rejected(self):
        self.controller.teach(True, "left")
        self.assertEqual(set(self.controller.capture("末端点", "left", "pose")["arms"]), {"left"})
        self.assertEqual(self.controller.snapshot()["teaching_sides"], ["left"])
        with self.assertRaises(ValueError): self.controller.capture("bad", "left", "unknown")

    def test_emergency_send_does_not_wait_for_controller_lock(self):
        import threading
        sent = threading.Event()
        original = self.controller.hardware.stop
        def stop(emergency=False):
            if emergency: sent.set()
            original(emergency)
        self.controller.hardware.stop = stop
        with self.controller.lock:
            thread = threading.Thread(target=lambda: self.controller.stop(emergency=True))
            thread.start()
            self.assertTrue(sent.wait(1))
        thread.join(1)
        self.assertFalse(thread.is_alive())

    def test_repeated_single_arm_point_preserves_other_arm_and_verifies(self):
        original=self.controller.hardware.joints[7:].copy()
        plan=point_plan()
        plan["steps"].append(dict(id="again",type="point",point="home",duration=.1))
        self.controller.start(plan, home_before=False)
        wait_for(lambda:not self.controller.busy())
        self.assertEqual(self.controller.status["state"],"completed",self.controller.snapshot())
        self.assertTrue(self.controller.last_verification["reached"])
        np.testing.assert_array_equal(self.controller.hardware.joints[7:],original)
        self.assertAlmostEqual(self.controller.hardware.joints[0],.01)

    def test_pause_freezes_commands_and_stop_interrupts_wait(self):
        plan=point_plan();plan["steps"][0]["duration"]=2
        self.controller.start(plan, home_before=False)
        wait_for(lambda:len(self.controller.hardware.commands)>3)
        self.controller.pause();time.sleep(.1)
        count=len(self.controller.hardware.commands)
        time.sleep(.1)
        self.assertEqual(len(self.controller.hardware.commands),count)
        self.controller.resume();wait_for(lambda:len(self.controller.hardware.commands)>count)
        self.controller.stop();wait_for(lambda:not self.controller.busy())
        self.assertEqual(self.controller.status["state"],"stopped")

    def test_model_load_does_not_block_points_and_multiple_workers_switch(self):
        plan=point_plan()
        plan["models"]=[dict(id=k,name=k,backend="mock",action_mode="absolute_joint",policy_hz=20,load_delay=.8)
                        for k in ("a","b")]
        plan["steps"] += [dict(id="vla1",type="vla",model="a",until="actions",value=3),
                          dict(id="return",type="point",point="home",duration=.1),
                          dict(id="vla2",type="vla",model="b",until="actions",value=3)]
        self.controller.start(plan, home_before=False)
        wait_for(lambda:len(self.controller.hardware.commands)>2)
        self.assertTrue(any(m["stage"]!="ready" for m in self.controller.models.snapshot()))
        wait_for(lambda:not self.controller.busy(),timeout=12)
        self.assertEqual(self.controller.status["state"],"completed",self.controller.snapshot())
        self.assertEqual(len(self.controller.models.snapshot()),2)
        self.assertEqual(self.controller.status["actions"],3)

    def test_loading_failure_stops_and_manual_next_works(self):
        plan=point_plan()
        plan["models"]=[dict(id="bad",name="bad",backend="mock",action_mode="absolute_joint",policy_hz=10,fail_load=True)]
        plan["steps"]=[dict(id="vla",type="vla",model="bad",until="manual")]
        self.controller.start(plan, home_before=False);wait_for(lambda:not self.controller.busy())
        self.assertEqual(self.controller.status["state"],"failed")
        self.assertIn("模拟加载失败",self.controller.status["error"])
        plan["models"][0].pop("fail_load")
        self.controller.start(plan, home_before=False)
        wait_for(lambda:self.controller.status.get("actions",0)>=2)
        self.controller.next();wait_for(lambda:not self.controller.busy())
        self.assertEqual(self.controller.status["state"],"completed",self.controller.snapshot())

    def test_execute_exits_teaching_and_emergency_prevents_further_commands(self):
        self.controller.teach(True)
        point=self.controller.capture("手动拖动点")
        self.assertEqual(point["name"],"手动拖动点")
        self.controller.start(point_plan(), home_before=False)
        wait_for(lambda:len(self.controller.hardware.commands)>2)
        self.controller.stop(emergency=True)
        wait_for(lambda:not self.controller.busy())
        count=len(self.controller.hardware.commands);time.sleep(.05)
        self.assertEqual(len(self.controller.hardware.commands),count)
        self.assertEqual(self.controller.status["state"],"emergency")


class VLASettingsTests(unittest.TestCase):
    def test_model_unload_does_not_block_progress_snapshot(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        pool = ModelPool()
        model = Mock()
        def close():
            entered.set()
            release.wait(3)
        model.close.side_effect = close
        pool.models['m'] = model
        worker = threading.Thread(target=pool.close)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                self.assertEqual(executor.submit(pool.snapshot).result(timeout=.5), [])
        finally:
            release.set()
            worker.join(2)

    def test_cancelled_inference_never_starts_another_rpc(self):
        import threading
        model = ModelProcess.__new__(ModelProcess)
        model.rpc_lock = threading.Lock()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            model.infer({}, '', cancel_event=cancel)

    def test_preloaded_model_reused_and_unreferenced_draft_does_not_block(self):
        controller = WorkflowController()
        self.addCleanup(controller.close)
        controller.connect(True)
        plan = new_plan()
        plan['settings']['control_hz'] = 80
        plan['models'] = [dict(id='ready', name='已配置模型', backend='mock', action_mode='absolute_joint'),
                          dict(id='draft', name='待编辑模型', backend='lerobot', action_mode='delta_joint', checkpoint='')]
        plan['steps'] = [dict(id='vla', type='vla', model='ready', until='actions', value=1)]
        with tempfile.TemporaryDirectory() as directory:
            store = PlanStore(directory)
            self.assertEqual(len(store.load(store.save(plan))['models']), 2)
        controller.preload(plan, ['ready'])
        model = controller.models.get('ready')
        wait_for(lambda: model.stage in ('ready', 'failed'), timeout=15)
        self.assertEqual(model.stage, 'ready', model.detail)
        self.assertEqual(model.config['control_hz'], 80)
        controller.start(plan, home_before=False)
        wait_for(lambda: not controller.busy(), timeout=15)
        self.assertEqual(controller.status['state'], 'completed', controller.snapshot())
        self.assertIs(controller.models.get('ready'), model)
        plan['models'][0]['name'] = '只改名称'
        controller.preload(plan, ['ready'])
        self.assertIs(controller.models.get('ready'), model)
        self.assertEqual(model.snapshot()['config']['name'], '只改名称')
        plan['steps'][0]['model'] = 'draft'
        with self.assertRaisesRegex(ValueError, '待编辑模型.*checkpoint'):
            controller.start(plan, home_before=False)
        with self.assertRaisesRegex(ValueError, '待编辑模型.*checkpoint'):
            controller.preload(plan, ['draft'])
        with self.assertRaisesRegex(ValueError, '模型不存在'):
            controller.preload(plan, ['missing'])

    def test_runtime_uses_queue_threshold_and_sync_reuse_count(self):
        from concurrent.futures import Future
        for mode in ('rtc','sync'):
            with self.subTest(mode=mode):
                model=SimpleNamespace(config=dict(id='m',action_mode='absolute_joint',
                    inference_mode=mode,rtc_queue_threshold=2,policy_hz=50,control_hz=50,
                    smoothing_time_constant=0),inference={'action_steps':2},latencies=[])
                pool=Mock();pool.get.return_value=model
                controller=WorkflowController(model_pool=pool)
                try:
                    controller.connect(True)
                    controller.plan=new_plan()
                    controller.plan['steps']=[dict(type='vla',model='m',until='actions',value=8)]
                    prefixes=[]
                    def submit(index,prefix=None,delay=0):
                        prefixes.append(None if prefix is None else len(prefix))
                        future=Future();future.set_result(np.tile(controller.hardware.joints,(5,1)))
                        return future
                    with patch.object(controller,'_submit',side_effect=submit):
                        controller._vla(0)
                    self.assertEqual(controller.status['actions'],8)
                    if mode=='rtc':
                        self.assertEqual(prefixes[0],None)
                        self.assertTrue(all(p==2 for p in prefixes[1:]),prefixes)
                        self.assertGreater(len(prefixes),1)
                    else:
                        self.assertEqual(prefixes,[None]*4)
                finally:controller.close()

    def test_model_parameters_roundtrip_and_legacy_rtc(self):
        plan=point_plan()
        model=dict(id='m',name='模型',backend='mock',action_mode='absolute_joint',policy_hz=12,
                   control_hz=80,smoothing_time_constant=0,inference_mode='rtc',rtc_mode='trained',
                   rtc_horizon=15,rtc_queue_threshold=25,sample_steps=8,n_action_steps=7,
                   rtc_schedule='linear',rtc_max_guidance_weight=3,rtc_mask_grippers=False)
        plan['models']=[model]
        with tempfile.TemporaryDirectory() as directory:
            store=PlanStore(directory)
            saved=store.load(store.save(plan))['models'][0]
            for key,value in model.items():self.assertEqual(saved[key],value)
        plan['models']=[dict(id='m',name='模型',backend='mock',action_mode='absolute_joint',rtc=False)]
        self.assertEqual(validate(plan)['models'][0]['inference_mode'],'sync')

    def test_time_constant_filter_and_short_rotation(self):
        start,end=np.zeros(14),np.ones(14)
        np.testing.assert_array_equal(smooth_target(start,end,.02,0,'joint'),end)
        np.testing.assert_allclose(smooth_target(start,end,.02,.1,'joint'),1-np.exp(-.2))
        end=np.zeros(14)
        end[5]=np.deg2rad(-179);start[5]=np.deg2rad(179)
        middle=smooth_target(start,end,np.log(2),1,'pose')
        self.assertAlmostEqual(abs(middle[5]),np.pi)

    def test_vla_control_frequency_is_local_and_restored_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=WorkflowController(model_pool=Mock())
            try:
                controller.plan=new_plan()
                controller.plan['steps']=[dict(model='m')]
                controller.tick_period=.02
                controller.models=Mock()
                controller.models.get.return_value.config=dict(control_hz=80)
                periods=[]
                def run(index):
                    periods.append(controller.tick_period)
                    raise ValueError('test')
                with patch.object(controller,'_vla_body',side_effect=run):
                    with self.assertRaisesRegex(ValueError,'test'):controller._vla(0)
                self.assertEqual(periods,[1/80])
                self.assertEqual(controller.tick_period,.02)
            finally:controller.close()

    def test_openpi_loader_forwards_rtc_and_sampling_settings(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from scripts.OpenpiDeployer import runtime as shared
        from easy_embodied_deploy.workflow import model_worker
        config=dict(backend='openpi',checkpoint='/tmp/fake-checkpoint',action_mode='absolute_joint',
                    inference_mode='rtc',rtc_mode='trained',rtc_horizon=20,rtc_queue_threshold=30,
                    rtc_schedule='linear',rtc_max_guidance_weight=3,rtc_mask_grippers=False,
                    policy_hz=13,sample_steps=7,n_action_steps=9)
        manifest={'config':dict(action_mode='absolute_joint',rtc_max_delay=8)}
        with patch.object(shared,'resolve_checkpoint',side_effect=lambda p:p), \
             patch.object(shared,'load_training_manifest',return_value=(None,manifest)), \
             patch.object(shared,'load_policy') as loader, \
             patch.object(shared,'PolicyChunkSampler') as sampler, \
             patch('easy_embodied_deploy.target_runtime.checkpoint_pose_contract',return_value=None), \
             patch.dict(model_worker.os.environ,{},clear=False):
            _,_,metadata=model_worker.load(config)
            rtc=sampler.call_args.args[2]
            self.assertEqual((rtc.mode,rtc.horizon,rtc.queue_threshold,rtc.policy_hz),('trained',20,30,13))
            self.assertEqual((rtc.schedule,rtc.max_guidance_weight,rtc.mask_grippers,rtc.training_max_delay),('linear',3,False,8))
            self.assertEqual(loader.call_args.kwargs['num_steps'],7)
            self.assertEqual(metadata,dict(training_max_delay=8,action_steps=9))
            sampler.return_value.warmup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
