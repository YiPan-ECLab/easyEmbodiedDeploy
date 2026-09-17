import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/LerobotDeployer"))
from hardware_bridge import PoseCommandLoop
from easy_embodied_deploy.pose_actions import matrix_rotvec, rpy_matrix


class PoseHardwareTests(unittest.TestCase):
    def test_move_p_units_gripper_and_short_rotation_path(self):
        pairs = {s: SimpleNamespace(follower=Mock(), follower_gripper=Mock(), gripper_force=1)
                 for s in ("left", "right")}
        initial = np.array([.2, .1, .3, 0, 0, np.pi-.01, .02]*2)
        target = initial.copy(); target[[5, 12]] = -np.pi+.01; target[[0, 7]] += .001
        loop = PoseCommandLoop(SimpleNamespace(pairs=pairs), initial)
        try:
            loop.configure(control_hz=30, smoothing_time_constant_s=.1)
            loop.set_target(target)
            deadline = time.monotonic()+1
            while not pairs["left"].follower.move_p.called and time.monotonic() < deadline:
                time.sleep(.005)
            for pair in pairs.values():
                pair.follower.move_p.assert_called()
                pair.follower.move_js.assert_not_called()
                pose = np.array(pair.follower.move_p.call_args.args[0])
                self.assertTrue(.2 <= pose[0] <= .201)
                change = np.linalg.norm(matrix_rotvec(rpy_matrix(pose[3:]) @ rpy_matrix(initial[3:6]).T))
                self.assertLessEqual(change, .020001)
                pair.follower_gripper.move_gripper_m.assert_called_with(.02, force=1)
            with self.assertRaises(ValueError):
                bad = target.copy(); bad[6] = .11; loop.set_target(bad)
            self.assertEqual(loop.stats()["motion_space"], "pose")
        finally:
            loop.stop()


if __name__ == "__main__":
    unittest.main()
