"""Follower role handoff regression; no SDK connection or physical commands."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from piper_recorder.robot import PiperPair


class FollowerConnectionTests(unittest.TestCase):
    def pair(self):
        pair = PiperPair.__new__(PiperPair)
        pair.side, pair.follower_name, pair.leader_name = 'left', 'piper-lo', 'piper-lt'
        pair.follower, pair.leader = mock.Mock(), mock.Mock()
        pair.follower_gripper, pair.leader_gripper = mock.Mock(), mock.Mock()
        pair._create_arm = mock.Mock(side_effect=[
            (pair.follower, pair.follower_gripper), (pair.leader, pair.leader_gripper)])
        return pair

    def test_role_is_restored_before_waiting_for_follower_feedback(self):
        pair = self.pair()
        ready = False
        def switch():
            nonlocal ready
            ready = True
        def feedback():
            return SimpleNamespace(timestamp=11) if ready else None
        pair.follower.set_follower_mode.side_effect = switch
        pair.follower.get_joint_angles.side_effect = feedback
        pair.follower.get_flange_pose.side_effect = feedback
        pair.follower_gripper.get_gripper_status.side_effect = feedback
        with mock.patch('piper_recorder.robot.time.time', return_value=10):
            pair.connect()
        pair.follower.set_follower_mode.assert_called_once()
        pair.follower.enable.assert_not_called()
        pair.follower.move_j.assert_not_called()
        pair.follower.move_js.assert_not_called()
        pair.leader.set_leader_mode.assert_called_once()

    def test_stale_pose_or_gripper_cannot_complete_handshake(self):
        for stale in ('关节', '法兰', '夹爪'):
            with self.subTest(stale=stale):
                pair = self.pair()
                for name, getter in [('关节', pair.follower.get_joint_angles),
                                     ('法兰', pair.follower.get_flange_pose),
                                     ('夹爪', pair.follower_gripper.get_gripper_status)]:
                    getter.return_value = SimpleNamespace(timestamp=9 if name == stale else 11)
                with mock.patch('piper_recorder.robot.time.monotonic', return_value=0), \
                     mock.patch('piper_recorder.robot.ticks', return_value=iter([0, 3])):
                    with self.assertRaisesRegex(RuntimeError, stale):
                        pair._wait_follower_feedback(timeout=3, after=10)

    def test_silent_bus_still_fails_without_enabling_or_moving(self):
        pair = self.pair()
        pair.follower.get_joint_angles.return_value = None
        pair.follower.get_flange_pose.return_value = None
        pair.follower_gripper.get_gripper_status.return_value = None
        with mock.patch('piper_recorder.robot.time.monotonic', return_value=0), \
             mock.patch('piper_recorder.robot.ticks', return_value=iter([0, 3])):
            with self.assertRaisesRegex(RuntimeError, '已请求从臂模式'):
                pair.connect()
        pair.follower.set_follower_mode.assert_called_once()
        pair.follower.enable.assert_not_called()
        pair.follower.move_j.assert_not_called()
        pair.leader.set_leader_mode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
