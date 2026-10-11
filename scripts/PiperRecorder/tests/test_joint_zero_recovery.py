import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/PiperRecorder"))

import joint_zero_recovery as recovery
from easy_embodied_deploy import recorder_console


def snapshot(*, j2_deg=4.9, code=0xA4, enabled=False, velocity=0.0, other=0.0):
    joints = [other, j2_deg, other, other, other, other]
    drivers = [
        {"joint": index, "code": 0x40, "enabled": True}
        for index in range(1, 7)
    ]
    drivers[1].update(code=code, enabled=enabled)
    motors = [
        {
            "joint": index,
            "position_rad": 0.0,
            "velocity_rad_s": 0.0,
            "current_a": 0.0,
            "torque_nm": 0.0,
        }
        for index in range(1, 7)
    ]
    motors[1]["velocity_rad_s"] = velocity
    return {"joints_deg": joints, "joints_rad": joints, "drivers": drivers, "motors": motors}


class JointZeroRecoveryTests(unittest.TestCase):
    def test_only_known_stationary_j2_fault_is_eligible(self):
        eligible, reasons = recovery.assess_snapshot(snapshot())
        self.assertTrue(eligible, reasons)

        cases = (
            snapshot(j2_deg=0),
            snapshot(j2_deg=15),
            snapshot(code=0x40, enabled=True),
            snapshot(velocity=0.2),
            snapshot(other=3),
        )
        for value in cases:
            with self.subTest(value=value):
                self.assertFalse(recovery.assess_snapshot(value)[0])

    def test_repair_uses_no_position_motion_and_verifies_healthy_zero(self):
        arm = mock.Mock()
        arm.clear_joint_error.return_value = True
        arm.calibrate_joint.return_value = True
        before = snapshot()
        after = snapshot(j2_deg=0, code=0x40, enabled=True)
        with (
            mock.patch.object(recovery, "verify_can_interface"),
            mock.patch.object(
                recovery.AgxArmFactory, "create_arm", return_value=arm
            ),
            mock.patch.object(
                recovery, "_wait_regular_feedback", return_value=[0, math.radians(4.9), 0, 0, 0, 0]
            ),
            mock.patch.object(recovery, "_wait_snapshot", side_effect=[before, after]),
            mock.patch.object(recovery, "_wait_joint_enabled"),
            mock.patch.object(recovery, "_wait_zero_feedback"),
        ):
            result = recovery.run("right", repair=True, confirmed=True)
        self.assertTrue(result["repaired"])
        arm.calibrate_joint.assert_called_once_with(2, timeout=2.0)
        arm.move_j.assert_not_called()
        arm.set_leader_mode.assert_called_once()
        arm.restore_leader_drag_mode.assert_called_once()
        arm.disconnect.assert_called_once()

    def test_console_requires_confirmation_and_parses_structured_result(self):
        session = recorder_console.RecorderSession()
        with self.assertRaisesRegex(ValueError, "确认"):
            session.joint_zero("repair", {"side": "right"})

        expected = {
            "ok": True,
            "operation": "inspect",
            "side": "right",
            "arm": "piper-rt",
            "eligible": False,
            "reasons": ["healthy"],
            "before": snapshot(j2_deg=0, code=0x40, enabled=True),
        }
        process = mock.Mock(returncode=0)
        process.communicate.return_value = (
            "diagnostic line\n"
            + recorder_console.ZERO_RESULT_PREFIX
            + json.dumps(expected)
            + "\n",
            None,
        )
        with mock.patch.object(recorder_console.subprocess, "Popen", return_value=process):
            result = session.joint_zero("inspect", {"side": "right"})
        self.assertEqual(result, expected)
        self.assertFalse(session.maintenance)
        self.assertIn("diagnostic line", session.logs[-1])

    def test_console_refuses_maintenance_during_recording(self):
        session = recorder_console.RecorderSession()
        session.process = mock.Mock()
        session.process.poll.return_value = None
        with self.assertRaisesRegex(RuntimeError, "结束采集"):
            session.joint_zero("inspect", {"side": "left"})

    def test_enable_timeout_identifies_faulted_joint(self):
        from piper_recorder.robot import PiperPair

        arm = mock.Mock()
        arm.enable.return_value = False
        arm.get_joint_angles.return_value = SimpleNamespace(
            msg=[0, math.radians(4.94), 0, 0, 0, 0]
        )
        healthy_foc = SimpleNamespace(driver_enable_status=True)
        fault_foc = SimpleNamespace(
            driver_enable_status=False,
            voltage_too_low=False,
            motor_overheating=False,
            driver_overcurrent=True,
            driver_overheating=False,
            collision_status=False,
            driver_error_status=True,
            stall_status=True,
        )
        arm.get_driver_states.side_effect = lambda index: SimpleNamespace(
            msg=SimpleNamespace(
                foc_status= fault_foc if index == 2 else healthy_foc,
                foc_status_code=0xA4 if index == 2 else 0x40,
            )
        )
        arm.get_motor_states.return_value = SimpleNamespace(
            msg=SimpleNamespace(current=-11.679, torque=-55.183)
        )
        with self.assertRaises(RuntimeError) as raised:
            PiperPair._wait_enabled(arm, "piper-rt", timeout=0)
        message = str(raised.exception)
        self.assertIn("piper-rt 使能失败", message)
        self.assertIn("J2:位置+4.940°", message)
        self.assertIn("状态码0xA4", message)
        self.assertIn("过流/驱动器错误/堵转", message)

    def test_parallel_failures_are_aggregated(self):
        from piper_recorder.robot import DualPiper

        futures = []
        for message in ("left failed", "right J2 0xA4"):
            future = mock.Mock()
            future.result.side_effect = RuntimeError(message)
            futures.append(future)
        with self.assertRaisesRegex(RuntimeError, "left failed；right J2 0xA4"):
            DualPiper._finish_parallel(futures)


if __name__ == "__main__":
    unittest.main()
