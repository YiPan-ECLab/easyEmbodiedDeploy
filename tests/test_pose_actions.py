"""Geometry and chunk semantics, independent of cameras and robot motion."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from easy_embodied_deploy.pose_actions import (PoseCodec, calibration_for_mode, flange_matrix,
    matrix_6d, matrix_rotvec, matrix_rpy, rotation_6d, rotvec_matrix, rpy_matrix)


def calibration():
    rows = []
    for side in ("left", "right"):
        for kind in ("wrist", "global"):
            transform = flange_matrix([.1, -.05, .03, .4, -.2, .7])
            rows.append(dict(job=f"{side}_wrist" if kind == "wrist" else f"global_{side}",
                mount_id="synthetic-unit-test", camera=dict(role=f"{side}_eye_in_hand" if kind == "wrist" else "agentview", serial=side if kind == "wrist" else "global"),
                result=dict(accepted=True, transform=f"T_{side}_flange_{side}_camera" if kind == "wrist" else f"T_{side}_base_global_camera", matrix=transform.tolist())))
    return dict(schema="easy_embodied_camera_calibration", calibrations=rows)


class PoseActionsTest(unittest.TestCase):
    def test_roundtrip_noncommuting_rotations(self):
        rng = np.random.default_rng(724)
        for mode in ("absolute_base_pose", "camera_delta_wrist", "camera_delta_global"):
            codec = PoseCodec(mode, calibration())
            for _ in range(15):
                anchor, target = rng.normal(size=(2, 12))
                action = codec.encode(anchor, target, [.02, .08])
                decoded, grip = codec.decode(anchor, action)
                np.testing.assert_allclose(grip, [.02, .08])
                for offset in (0, 6):
                    np.testing.assert_allclose(flange_matrix(decoded[offset:offset+6]), flange_matrix(target[offset:offset+6]), atol=1e-7)

    def test_eq5_frozen_anchor_and_absolute_gripper(self):
        codec = PoseCodec("camera_delta_wrist", calibration())
        poses = np.zeros((4, 12)); poses[:, 0] = [0, .02, .04, .06]
        poses[:, 5] = [0, .2, -.3, .5]
        grips = np.array([[.01, .02], [.02, .03], [.04, .05], [.06, .07]])
        chunk, valid = codec.chunk(poses, grips, 0, 5)
        np.testing.assert_array_equal(valid, [1, 1, 1, 0, 0])
        camera = rpy_matrix([.4, -.2, .7])
        for i in range(3):
            np.testing.assert_allclose(chunk[i, :3], camera.T @ [(.02*(i+1)), 0, 0], atol=1e-8)
            expected = camera.T @ rpy_matrix([0, 0, poses[i+1, 5]]) @ camera
            np.testing.assert_allclose(rotvec_matrix(chunk[i, 3:6]), expected, atol=1e-7)
        np.testing.assert_allclose(chunk[:3, 6], grips[1:, 0])
        np.testing.assert_array_equal(chunk[3:], np.repeat(chunk[2:3], 2, axis=0))

    def test_pi_gimbal_and_6d_column_order(self):
        for value in ([0, np.pi/2, .7], [.2, -np.pi/2, -.4], [np.pi, 0, 0], [.1, .2, .3]):
            matrix = rpy_matrix(value)
            np.testing.assert_allclose(rotation_6d(matrix), np.r_[matrix[:, 0], matrix[:, 1]])
            np.testing.assert_allclose(matrix_6d(rotation_6d(matrix)), matrix, atol=1e-8)
            np.testing.assert_allclose(rotvec_matrix(matrix_rotvec(matrix)), matrix, atol=1e-7)
            np.testing.assert_allclose(rpy_matrix(matrix_rpy(matrix)), matrix, atol=1e-7)

    def test_calibration_contract(self):
        self.assertEqual(calibration_for_mode(None, "absolute_base_pose"), {})
        with self.assertRaises(ValueError): PoseCodec("camera_delta_wrist")
        with self.assertRaises(ValueError): calibration_for_mode(calibration(), "camera_delta_global", "other-installation")
        bad = calibration(); bad["calibrations"][0]["result"]["matrix"][0][0] = 5
        with self.assertRaises(ValueError): PoseCodec("camera_delta_wrist", bad)
        with self.assertRaises(ValueError): matrix_6d([1, 0, 0, 2, 0, 0])


if __name__ == "__main__":
    unittest.main()
