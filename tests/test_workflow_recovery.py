"""Recovery tests never open real CAN or alter calibration."""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from easy_embodied_deploy.workflow.recovery import inspect_arm, restore_arm
from easy_embodied_deploy.workflow.runtime import WorkflowController

LIMITS = np.radians([[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-120, 120]])


class Arm:
    def __init__(self, emergency=False):
        self.q = np.radians([0, -2, 2, -7, 21, -1])
        self.enabled, self.emergency, self.fault = False, emergency, False
        self.calls = []
        self._msg_mode = SimpleNamespace(move_spd_rate_ctrl=50)
    def msg(self, msg):
        return SimpleNamespace(msg=msg, timestamp=time.time())
    def get_arm_status(self):
        return self.msg(SimpleNamespace(arm_status=int(self.emergency), ctrl_mode=0,
                                       teach_status=0, err_status=SimpleNamespace()))
    def get_joint_angles(self):
        return self.msg(self.q.copy())
    def get_driver_states(self, index):
        return self.msg(SimpleNamespace(foc_status_code=0xA4 if self.fault and index == 2 else 0x40 if self.enabled else 0,
            foc_status=SimpleNamespace(driver_enable_status=self.enabled,
                                       stall_status=self.fault and index == 2)))
    def get_motor_states(self, index):
        return self.msg(SimpleNamespace(velocity=0))
    def set_speed_percent(self, value):
        self.calls.append(('speed', value))
    def move_j(self, q):
        self.calls.append(('move_j', q)); self.target = np.asarray(q)
        if self.enabled and not self.emergency: self.q = self.target.copy()
    def enable(self):
        self.calls.append(('enable',)); self.enabled = True
    def reset(self):
        self.calls.append(('reset',)); self.emergency = False
    def electronic_emergency_stop(self):
        self.calls.append(('emergency',)); self.emergency = True


class RecoveryTests(unittest.TestCase):
    def test_disabled_is_distinct_from_known_zero_fault_and_diagnosis_is_read_only(self):
        arm = Arm()
        result = inspect_arm(arm, LIMITS)
        self.assertTrue(result['eligible'])
        self.assertFalse(result['known_leader_fault'])
        self.assertEqual(arm.calls, [])
        arm.fault = True
        result = inspect_arm(arm, LIMITS)
        self.assertFalse(result['eligible'])
        self.assertTrue(result['known_leader_fault'])

    def test_restore_primes_target_before_enable_preserves_other_joints_and_never_calibrates(self):
        for emergency in (False, True):
            arm = Arm(emergency)
            initial = arm.q.copy()
            with patch('easy_embodied_deploy.workflow.recovery.time.sleep'):
                result = restore_arm(arm, LIMITS, support_confirmed=emergency)
            self.assertTrue(result['recovered'])
            np.testing.assert_allclose(arm.q[[0, 3, 4, 5]], initial[[0, 3, 4, 5]])
            self.assertLessEqual(np.max(np.abs(arm.q-initial)), np.deg2rad(2.001))
            operations = [c[0] for c in arm.calls]
            self.assertLess(operations.index('move_j'), operations.index('enable'))
            if emergency:
                self.assertLess(operations.index('move_j'), operations.index('reset'))
            self.assertEqual(set(operations)-{'speed', 'move_j', 'enable', 'reset'}, set())

    def test_fault_or_large_offset_never_sends_commands(self):
        for reason in ('fault', 'offset', 'stale'):
            arm = Arm()
            if reason == 'fault': arm.fault = True
            if reason == 'offset': arm.q[1] = np.deg2rad(-15)
            if reason == 'stale': arm.get_joint_angles = lambda: SimpleNamespace(msg=arm.q, timestamp=0)
            with self.assertRaises(RuntimeError): restore_arm(arm, LIMITS)
            self.assertEqual(arm.calls, [])

    def test_reset_requires_support_and_waits_for_transient_communication_restart(self):
        arm = Arm(True)
        with self.assertRaisesRegex(RuntimeError, '托稳'): restore_arm(arm, LIMITS)
        self.assertEqual(arm.calls, [])
        original = arm.get_arm_status
        remaining = 0
        def reset():
            nonlocal remaining
            arm.emergency = False; arm.enabled = False; remaining = 3
        def status():
            nonlocal remaining
            message = original()
            if remaining:
                remaining -= 1; message.msg.arm_status = 5
            return message
        arm.reset, arm.get_arm_status = reset, status
        with patch('easy_embodied_deploy.workflow.recovery.time.sleep'):
            self.assertTrue(restore_arm(arm, LIMITS, support_confirmed=True)['recovered'])

    def test_abort_does_not_restore_high_speed_or_enable(self):
        arm = Arm()
        checks = 0
        def check():
            nonlocal checks
            checks += 1
            if checks == 3: raise InterruptedError()
        with self.assertRaises(InterruptedError): restore_arm(arm, LIMITS, check)
        self.assertEqual([c[0] for c in arm.calls], ['speed', 'move_j'])
        self.assertEqual(arm.calls[0], ('speed', 5))

    def test_excess_motion_triggers_selected_arm_estop(self):
        arm = Arm()
        def enable():
            arm.enabled = True; arm.q[4] += .2
        arm.enable = enable
        with patch('easy_embodied_deploy.workflow.recovery.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, '3.5°'):
                restore_arm(arm, LIMITS)
        self.assertEqual(arm.calls[-1], ('emergency',))

    def test_controller_rechecks_eligibility_and_blocks_concurrent_operations(self):
        controller = WorkflowController()
        controller.hardware = Mock(simulated=False)
        controller.hardware.diagnose.return_value = dict(eligible=False, reasons=['堵转'])
        try:
            with self.assertRaisesRegex(ValueError, '堵转'): controller.restore_arm('right')
            controller.hardware.restore.assert_not_called()
            controller.thread = Mock()
            controller.thread.is_alive.return_value = True
            with self.assertRaises(ValueError): controller.diagnose_arm('right')
        finally:
            controller.thread = None
            controller.hardware = None
            controller.close()


if __name__ == '__main__':
    unittest.main()
