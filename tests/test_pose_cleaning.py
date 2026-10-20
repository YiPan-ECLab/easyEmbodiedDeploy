"""Offline geometry and segmentation checks; no hardware or sampling-rate assumptions."""
import unittest

import numpy as np

from easy_embodied_deploy.pose_cleaning import (
    clean_pose_indices, episode_sources, geometric_jacobian, piper_mdh, singularity_ratio,
)


def forward(q):
    transform = np.eye(4)
    for angle, (d, a, alpha, offset) in zip(q, piper_mdh()):
        c, s = np.cos(angle+offset), np.sin(angle+offset)
        ca, sa = np.cos(alpha), np.sin(alpha)
        transform = transform @ np.array([
            [c, -s, 0, a], [ca*s, ca*c, -sa, -sa*d],
            [sa*s, sa*c, ca, ca*d], [0, 0, 0, 1],
        ])
    return transform


def regular_states(count):
    return np.tile([0, .7, -1, 0, .5, 0, .01]*2, (count, 1))


class PoseCleaningTests(unittest.TestCase):
    def test_jacobian_matches_finite_differences(self):
        rng = np.random.default_rng(31)
        for q in rng.uniform(-1, 1, (12, 6)):
            expected = np.empty((6, 6))
            rotation = forward(q)[:3, :3]
            for axis in range(6):
                step = np.eye(6)[axis]*1e-6
                derivative = (forward(q+step)-forward(q-step))/2e-6
                omega = derivative[:3, :3] @ rotation.T
                expected[:, axis] = [*derivative[:3, 3], omega[2, 1], omega[0, 2], omega[1, 0]]
            np.testing.assert_allclose(geometric_jacobian(q[None])[0], expected, atol=1e-9)

    def test_wrist_singularity_and_regular_configuration(self):
        regular = regular_states(1)[0, :6]
        singular = regular.copy()
        singular[4] = 0
        self.assertGreater(singularity_ratio(regular[None])[0], .01)
        self.assertLess(singularity_ratio(singular[None])[0], 1e-12)

    def test_either_arm_splits_and_isolated_frames_are_counted(self):
        states = regular_states(10)
        states[2, 4] = 0
        states[4, 11] = 0
        runs, report = clean_pose_indices(states, np.arange(10))
        self.assertEqual([r.tolist() for r in runs], [[0, 1], [5, 6, 7, 8, 9]])
        self.assertEqual(report['removed_singular_frames'], 2)
        self.assertEqual(report['isolated_frames'], 1)
        self.assertEqual(report['output_frames'], 5)
        jobs = episode_sources([{'name': 'one.hdf5', 'pose_cleaning': report}])
        self.assertEqual([j['clean_segment'] for j in jobs], [[0, 2], [5, 10]])

    def test_downsampling_does_not_bridge_unsampled_singularity(self):
        states = regular_states(9)
        states[3, 11] = 0
        runs, report = clean_pose_indices(states, [0, 2, 4, 6, 8])
        self.assertEqual([r.tolist() for r in runs], [[0, 2], [4, 6, 8]])
        self.assertEqual(report['removed_singular_frames'], 0)
        self.assertEqual(report['output_episodes'], 2)

    def test_all_bad_and_no_bad(self):
        states = regular_states(5)
        self.assertEqual(clean_pose_indices(states, np.arange(5))[1]['output_frames'], 4)
        states[:, 4] = 0
        runs, report = clean_pose_indices(states, np.arange(5))
        self.assertEqual(runs, [])
        self.assertEqual(report['removed_singular_frames'], 5)
        self.assertEqual(report['output_frames'], 0)


if __name__ == '__main__':
    unittest.main()
