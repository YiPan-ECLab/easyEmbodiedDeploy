import sys
import unittest
from collections import deque
from pathlib import Path
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from easy_embodied_deploy.pose_actions import PoseCodec, flange_matrix
from easy_embodied_deploy.target_runtime import COMMAND_KEY, POSE_KEY, TargetSampler
from easy_embodied_deploy.pose_quality import PoseThresholds, analyze_poses
from easy_embodied_deploy.checkpoint_runtime import initialize_for_resume, copy_resume_manifest
from test_pose_actions import calibration


class Sampler:
    latencies = deque([.2])
    last_guidance_horizon = None

    def __init__(self, chunk):
        self.chunk, self.previous = chunk, None

    def infer(self, observation, *, prev_action_chunk=None, **kwargs):
        self.previous = prev_action_chunk
        return self.chunk.copy()


class PoseRuntimeTests(unittest.TestCase):
    def test_resumed_checkpoint_keeps_portable_action_contract(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "training_manifest.json"
            contract = {"action_mode": "camera_delta_wrist"}
            source.write_text(json.dumps({"config": {"steps": 1}, "pose_contract": contract}))
            checkpoint = root / "1"
            checkpoint.mkdir()
            copy_resume_manifest(source, root, [checkpoint], status="completed", command=["resume"], steps=2)
            actual = json.loads((checkpoint / "training_manifest.json").read_text())
            self.assertEqual(actual["pose_contract"], contract)
            self.assertEqual(actual["config"]["steps"], 2)
            self.assertEqual(actual["status"], "completed")

    def test_step_zero_resume_restores_and_empty_resume_does_not_restart(self):
        from unittest.mock import Mock
        manager = Mock()
        manager.all_steps.return_value = [0]
        initialize = Mock(return_value=(manager, False))
        self.assertEqual(initialize_for_resume(initialize, resume=True), (manager, True))
        manager.close.assert_not_called()
        manager.all_steps.return_value = []
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            initialize_for_resume(initialize, resume=True)
        manager.close.assert_called_once()
        self.assertEqual(initialize_for_resume(initialize, resume=False), (manager, False))

    def test_rtc_snapshot_includes_the_target_dispatched_with_its_observation(self):
        sys.path.insert(0, str(ROOT))
        from scripts.OpenpiDeployer.runtime import AsyncRTCActionEngine, RTCSettings
        ready = threading.Event()
        class CaptureSampler:
            absolute_targets = True
            latencies = deque([.001])
            last_guidance_horizon = None
            previous = None
            def infer(self, observation, *, prev_action_chunk, **kwargs):
                self.previous = prev_action_chunk.copy()
                self.anchor = observation[COMMAND_KEY].copy()
                ready.set()
                return np.repeat(prev_action_chunk[:1], 50, axis=0)
        sampler = CaptureSampler()
        engine = AsyncRTCActionEngine(sampler, RTCSettings(queue_threshold=40))
        try:
            with engine._condition:
                engine._queue.extend(np.full(14, i) for i in range(1, 5))
            action = engine.get_action({COMMAND_KEY: np.zeros(14)})
            self.assertTrue(ready.wait(2))
            np.testing.assert_array_equal(sampler.previous[0], action)
            np.testing.assert_array_equal(sampler.previous[:, 0], [1, 2, 3, 4])
            np.testing.assert_array_equal(sampler.anchor, np.zeros(14))
        finally:
            engine.close()

    def test_delta_elapsed_steps_keep_total_displacement(self):
        anchor = np.zeros(14)
        chunk = np.zeros((50, 14)); chunk[:, 0] = .01
        native = Sampler(chunk)
        sampler = TargetSampler(native, "delta_joint")
        targets = sampler.infer({COMMAND_KEY: anchor})
        # A 3-frame inference delay used to lose 0.03 rad permanently.
        remaining = targets[3:]
        self.assertAlmostEqual(remaining[0, 0], .04)
        self.assertAlmostEqual(remaining[-1, 0], .5)
        new_anchor = targets[9]
        sampler.infer({COMMAND_KEY: new_anchor}, prev_action_chunk=targets[10:])
        np.testing.assert_allclose(native.previous[:, 0], .01, atol=1e-7)

    def test_pose_rtc_prefix_reanchors_same_base_targets(self):
        old = np.array([.3, .2, .1, .2, -.4, .3]*2)
        current = old + [.01, -.03, .02, -.4, .2, -.1]*2
        goals = np.array([old + [.001*i, 0, 0, .002*i, -.001*i, 0]*2 for i in range(1, 51)])
        for mode in ("absolute_base_pose", "camera_delta_wrist", "camera_delta_global"):
            codec = PoseCodec(mode, calibration())
            native = Sampler(np.array([codec.encode(old, goal, [.04, .06]) for goal in goals]))
            sampler = TargetSampler(native, mode, {"calibration": calibration()})
            targets = sampler.infer({POSE_KEY: old})
            sampler.infer({POSE_KEY: current}, prev_action_chunk=targets[8:])
            for i, action in enumerate(native.previous):
                recovered, grip = codec.decode(current, action)
                for offset in (0, 6):
                    np.testing.assert_allclose(flange_matrix(recovered[offset:offset+6]),
                                               flange_matrix(goals[i+8, offset:offset+6]), atol=1e-7)
                np.testing.assert_allclose(grip, [.04, .06], atol=1e-7)

    def test_pose_quality_uses_rotation_geometry_and_detects_freeze(self):
        poses = np.zeros((8, 12)); states = np.zeros((8, 14)); times = np.arange(8)*.1
        poses[:4, 5] = np.pi-.01
        poses[4:, 5] = -np.pi+.01
        states[1:, 7] = np.arange(1, 8)*.01
        report = analyze_poses(poses, times, states)
        self.assertAlmostEqual(report["sides"]["left"]["max_rotation_step_rad"], .02)
        self.assertEqual(report["sides"]["left"]["rotvec_branch_crossing_frames"], [4])
        self.assertEqual(len(report["sides"]["right"]["frozen_pose_moving_joint_frames"]), 7)
        with self.assertRaises(ValueError):
            PoseThresholds(max_speed_m_s=float("nan"))


if __name__ == "__main__":
    unittest.main()
