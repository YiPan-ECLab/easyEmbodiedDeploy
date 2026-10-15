import io
import queue
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import record
from piper_recorder.control import WebOperator
from piper_recorder.camera import CameraFrame
from piper_recorder.robot import ArmSample, ArmCommand


class RecordingTests(unittest.TestCase):
    def test_recording_accepts_five_degree_and_five_mm_boundary_overshoot(self):
        from piper_recorder.dataset import _clip_side_values
        from piper_recorder.robot import JOINT_LIMITS_RAD

        lower = np.array([pair[0] for pair in JOINT_LIMITS_RAD] + [0.0])
        upper = np.array([pair[1] for pair in JOINT_LIMITS_RAD] + [0.1])
        allowance = np.array([np.deg2rad(5.0)] * 6 + [0.005])
        for label, values, expected in (
            ("lower", lower - allowance, lower),
            ("upper", upper + allowance, upper),
        ):
            with self.subTest(boundary=label):
                clipped, count = _clip_side_values(values[None, :], "left state")
                np.testing.assert_array_equal(clipped[0], expected)
                self.assertEqual(count, 7)

        # Regression: this actual J6 feedback previously rejected the episode.
        values = ((lower + upper) / 2)[None, :]
        values[0, 5] = np.deg2rad(-122.185)
        clipped, count = _clip_side_values(values, "left state")
        self.assertEqual(clipped[0, 5], lower[5])
        self.assertEqual(count, 1)
        self.assertEqual(values[0, 5], np.deg2rad(-122.185))

    def test_recording_rejects_values_beyond_new_boundary_allowance(self):
        from piper_recorder.dataset import _clip_side_values
        from piper_recorder.robot import JOINT_LIMITS_RAD

        lower = np.array([pair[0] for pair in JOINT_LIMITS_RAD] + [0.0])
        upper = np.array([pair[1] for pair in JOINT_LIMITS_RAD] + [0.1])
        excess = np.array([np.deg2rad(5.001)] * 6 + [0.005001])
        for axis in range(7):
            for boundary, direction in ((lower, -1), (upper, 1)):
                with self.subTest(axis=axis, direction=direction):
                    values = ((lower + upper) / 2)[None, :]
                    values[0, axis] = boundary[axis] + direction * excess[axis]
                    with self.assertRaisesRegex(ValueError, "超出允许截断范围"):
                        _clip_side_values(values, "right action")

    def operator(self):
        operator = WebOperator.__new__(WebOperator)
        operator.messages = queue.Queue()
        operator.disconnected = threading.Event()
        operator.token = 0
        operator.phase_name = "idle"
        operator.accepts = []
        operator.last_progress = 0
        operator.emit = mock.Mock()
        return operator

    def test_stale_and_duplicate_pedal_messages_do_not_cross_phases(self):
        operator = self.operator()
        operator.phase("instruction", accepts=("submit",))
        command = {"token": operator.token, "action": "submit", "value": "task"}
        operator.messages.put(command)
        operator.messages.put(command)
        self.assertEqual(operator._take()["value"], "task")
        operator.phase("waiting_motion", accepts=("finish",))
        self.assertFalse(operator.stop_requested())
        operator.messages.put({"token": operator.token, "action": "finish"})
        operator.update("recording")
        self.assertTrue(operator.stop_requested())
        self.assertFalse(operator.stop_requested())

    def test_disconnect_surfaces_instead_of_leaving_uncontrolled_input(self):
        operator = self.operator()
        operator.disconnected.set()
        with self.assertRaises(EOFError):
            operator._take()

    def test_real_record_episode_keeps_task_then_home_and_skips_wait(self):
        now = [100.0]
        samples = [0]
        calls = []

        def sample():
            samples[0] += 1
            value = 0 if samples[0] < 4 else 0.02 + samples[0] * 0.001
            action = ArmCommand((value, 0, 0, 0, 0, 0), 0, now[0])
            item = ArmSample(action.joints, (0,) * 6, 0, 1, action)
            idle_action = ArmCommand((0,) * 6, 0, 99.0)
            idle = ArmSample(idle_action.joints, (0,) * 6, 0, 1, idle_action)
            return {"left": item, "right": idle}

        def ticks(_):
            for _ in range(7):
                now[0] += 0.04
                yield now[0]

        def home(on_sample, sample_hz):
            calls.append("home")
            for _ in range(3):
                now[0] += 0.04
                on_sample()

        robot = mock.Mock(sample=sample, home=home, sample_diagnostics=lambda: {})
        robot.preposition_and_start.side_effect = lambda: calls.append("preposition")
        cameras = {
            name: mock.Mock(
                read=lambda: CameraFrame(np.zeros((224, 224, 3), np.uint8), now[0])
            )
            for name in record.CAMERA_NAMES
        }
        operator = mock.Mock()
        operator.stop_requested.side_effect = [False] * 6 + [True]
        with (
            mock.patch.object(record, "ticks", side_effect=ticks),
            mock.patch.object(record.time, "monotonic", side_effect=lambda: now[0]),
            mock.patch.object(record, "terminal_bell"),
            mock.patch.object(record, "clear_pending_input"),
            mock.patch("sys.stdout", new=io.StringIO()),
        ):
            episode = record.record_episode(robot, cameras, 30, operator=operator)
        self.assertEqual(calls, ["preposition", "home"])
        self.assertEqual(episode.phases, [0] * 4 + [1] * 3)
        self.assertEqual(len(episode), 7)
        self.assertEqual(episode.recording_metadata["start_joint_delta_rad"], 0.01)

    def test_either_arm_or_gripper_can_start_recording_at_threshold(self):
        baseline = {side: ((0,) * 6, 0) for side in ("left", "right")}
        for side in baseline:
            for gripper_only in (False, True):
                for scale, expected in ((0.99, False), (1.0, True)):
                    samples = {}
                    for arm in baseline:
                        joints = (.01 * scale if arm == side and not gripper_only else 0,) + (0,) * 5
                        width = .002 * scale if arm == side and gripper_only else 0
                        command = ArmCommand(joints, width, 0)
                        samples[arm] = ArmSample(joints, (0,) * 6, width, 1, command)
                    with self.subTest(side=side, gripper_only=gripper_only, scale=scale):
                        self.assertEqual(record._operator_started(samples, baseline, .01, .002), expected)

    def test_nonfinite_feedback_is_rejected_before_clamping(self):
        from types import SimpleNamespace
        from piper_recorder.robot import PiperPair, _message_values, _normalize_gripper_width

        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
                    _normalize_gripper_width(value, .1, "right")
                with self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
                    _message_values(SimpleNamespace(msg=[value] + [0] * 5), 6, "right")
                with self.assertRaisesRegex(RuntimeError, "有限"):
                    PiperPair._clamp_joints([value] + [0] * 5)
        self.assertEqual(_normalize_gripper_width(-.0001, .1, "right"), 0)
        self.assertEqual(_normalize_gripper_width(.1001, .1, "right"), .1)

    def test_idle_arm_does_not_require_new_leader_commands(self):
        from piper_recorder.robot import PiperPair

        pair = PiperPair.__new__(PiperPair)
        pair.status_period_s = .5
        pair.follow_hz = 200
        pair._stop = threading.Event()
        pair._lock = threading.Lock()
        pair._error = None
        pair._complete_leader_joints = mock.Mock(return_value=None)
        pair.leader_gripper = mock.Mock()
        pair.leader_gripper.get_gripper_ctrl_states.return_value = None
        pair.follower = mock.Mock()
        pair.follower_gripper = mock.Mock()
        command = ArmCommand((0, 0, 0, 0, 0, .785398), 0, -100)
        with (
            mock.patch("piper_recorder.robot.time.monotonic", return_value=0),
            mock.patch("piper_recorder.robot.ticks", return_value=iter([0, .005, .01])),
        ):
            pair._follow_loop(command, (0, 0, 0), 0)
        self.assertIsNone(pair._error)
        self.assertEqual(pair._command, command)
        pair.follower.move_js.assert_not_called()
        pair.follower_gripper.move_gripper_m.assert_not_called()

    def test_invalid_options_are_rejected_before_movement(self):
        robot = mock.Mock()
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                record.record_episode(robot, {}, value)
        robot.preposition_and_start.assert_not_called()

    def test_mode_switch_accepts_intermittent_fresh_feedback(self):
        from piper_recorder.robot import PiperPair
        from types import SimpleNamespace

        pair = PiperPair.__new__(PiperPair)
        pair.leader_name = "test"
        pair.leader = mock.Mock()
        # Hardware trace: repeated mode requests interrupt publication, with
        # fresh feedback only every few polls. Do not keep reconfiguring once
        # the controller has supplied the first fresh sample.
        stamps = [1, 1, 1, 1, 1, 1, 2]
        pair.leader.get_joint_angles.side_effect = [
            SimpleNamespace(timestamp=n) for n in stamps
        ]
        with (
            mock.patch(
                "piper_recorder.robot.ticks", return_value=iter([0, .05, .1, .15, .2, .25])
            ),
            mock.patch("piper_recorder.robot.time.monotonic", return_value=0),
        ):
            result = pair._enter_leader_control_mode(timeout=10)
        self.assertEqual(result.timestamp, 2)
        self.assertEqual(pair.leader.set_follower_mode.call_count, 6)
        pair.leader.move_j.assert_not_called()

    def test_mode_switch_still_rejects_missing_or_stale_feedback(self):
        from piper_recorder.robot import PiperPair
        from types import SimpleNamespace

        for feedback in (None, SimpleNamespace(timestamp=1)):
            with self.subTest(feedback=feedback):
                pair = PiperPair.__new__(PiperPair)
                pair.leader_name = "test"
                pair.leader = mock.Mock()
                pair.leader.get_joint_angles.return_value = feedback
                with (
                    mock.patch("piper_recorder.robot.ticks", return_value=iter([0, 1, 2])),
                    mock.patch("piper_recorder.robot.time.monotonic", return_value=0),
                ):
                    with self.assertRaisesRegex(RuntimeError, "切换 follower 后没有关节反馈"):
                        pair._enter_leader_control_mode(timeout=2)
                self.assertTrue(pair._leader_is_follower)
                pair.leader.enable.assert_not_called()
                pair.leader.move_j.assert_not_called()


class RecordingSessionTests(unittest.TestCase):
    def test_save_failure_retry_discard_and_instruction_reuse(self):
        from contextlib import nullcontext

        robot = mock.Mock()
        episode = mock.MagicMock()
        episode.__len__.return_value = 12
        operator = mock.Mock()
        operator.prompt.side_effect = ["test task", "", "y", "", "n", ":q"]
        with (
            mock.patch.object(record, "hardware_lease", return_value=nullcontext()),
            mock.patch.object(record, "open_cameras", return_value={}),
            mock.patch.object(record, "DualPiper", return_value=robot),
            mock.patch.object(
                record, "record_episode", return_value=episode
            ) as capture,
            mock.patch.object(
                record,
                "write_episode",
                side_effect=[OSError("disk unavailable"), Path("/tmp/test.hdf5")],
            ) as writer,
            mock.patch("sys.stdout", new=io.StringIO()),
            mock.patch("sys.stderr", new=io.StringIO()),
        ):
            record.run(Path("/tmp/unused-recording"), operator=operator)
        self.assertEqual(capture.call_count, 2)
        self.assertEqual(writer.call_count, 2)
        self.assertTrue(
            all(call.args[1] == "test task" for call in writer.call_args_list)
        )
        robot.close.assert_called_once()

    def test_bridge_bind_failure_releases_transport_without_hardware(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LerobotDeployer"))
        import hardware_bridge as bridge

        context = mock.Mock()
        socket = context.socket.return_value
        socket.bind.side_effect = RuntimeError("address already in use")
        with (
            mock.patch.object(bridge.zmq, "Context", return_value=context),
            mock.patch.object(bridge, "open_cameras") as cameras,
        ):
            with self.assertRaisesRegex(RuntimeError, "address already"):
                bridge._serve("tcp://127.0.0.1:6007")
        socket.close.assert_called_once()
        context.term.assert_called_once()
        cameras.assert_not_called()

    def test_bridge_failed_preposition_keeps_original_cleanup_sequence(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LerobotDeployer"))
        import hardware_bridge as bridge

        robot = mock.Mock()
        order = []
        robot.close.side_effect = lambda: order.append("disconnect")
        with (
            mock.patch.object(bridge.zmq, "Context"),
            mock.patch.object(bridge, "open_cameras"),
            mock.patch.object(bridge, "DualPiper", return_value=robot),
            mock.patch.object(bridge, "connect_followers"),
            mock.patch.object(
                bridge,
                "prepare_followers",
                side_effect=RuntimeError("preposition failed"),
            ),
            mock.patch.object(
                bridge, "home_followers", side_effect=lambda robot: order.append("home")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "preposition failed"):
                bridge._serve("tcp://127.0.0.1:6007")
        self.assertEqual(order, ["disconnect"])


if __name__ == "__main__":
    unittest.main()
