import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from test_data_pipeline import make_episode, processor, analyze_lerobot
from test_pose_actions import calibration
from easy_embodied_deploy.pose_dataset import PoseDataset, copy_retained_episodes, training_stats


class PosePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.file = make_episode(self.source)
        self.calibration = self.root / "calibration.json"
        self.calibration.write_text(json.dumps(calibration()))
        with h5py.File(self.file, "a") as f:
            for side in ("left", "right"):
                pose = f[f"data/demo_0/obs/{side}_ee_states"]
                values = np.zeros(pose.shape)
                values[:, :3] = [.2, .1, .3]
                values[:, 0] += np.minimum(np.arange(len(values)), 22)*.001
                values[:, 5] = np.minimum(np.arange(len(values)), 22)*.003
                pose[...] = values

    def tearDown(self):
        self.temp.cleanup()

    def settings(self, mode, **kwargs):
        return processor.Settings(str(self.source), output_root=str(self.root/mode),
            action_mode=mode, calibration_bundle=str(self.calibration),
            calibration_mount_id="synthetic-unit-test", **kwargs)

    def test_all_pose_modes_convert_analyze_mask_and_preserve_state(self):
        for mode in ("absolute_base_pose", "camera_delta_wrist", "camera_delta_global"):
            result = processor.run(self.settings(mode))
            self.assertTrue(result["validation"]["valid"], result["validation"])
            root = self.root/mode/"lerobot"
            data = PoseDataset(root)
            # The follower moved after the leader stopped. Keep that endpoint.
            self.assertEqual(data.episode(0)["source_indices"][-1], 21)
            action, valid = data.chunk(data.rows-1)
            self.assertEqual(valid.sum(), 1)
            np.testing.assert_array_equal(action[1:], np.repeat(action[:1], 49, axis=0))
            report = analyze_lerobot.analyze(str(root.parent), None)
            self.assertEqual(report["status"], "PASS", report["errors"])
            self.assertEqual(report["source_verification"]["max_state_error"], 0)
            count = training_stats(root, {})["action"]["count"][0]
            self.assertEqual(count, sum(range(1, data.rows+1)))

    def test_camera_modes_require_mount_and_gaps_cannot_be_compacted(self):
        with self.assertRaisesRegex(ValueError, "安装版本"):
            processor.scan(processor.Settings(str(self.source), action_mode="camera_delta_wrist",
                                              calibration_bundle=str(self.calibration)))
        with self.assertRaisesRegex(ValueError, "reject"):
            processor.scan(self.settings("absolute_base_pose", gap_policy="compact"))

    def test_pose_sidecar_tampering_is_reported(self):
        processor.run(self.settings("absolute_base_pose"))
        root = self.root/"absolute_base_pose/lerobot"
        file = root/"meta/pose/episode-000000.npz"
        with np.load(file) as original:
            values = dict(original)
        values["poses"][2, 0] += .02
        np.savez_compressed(file, **values)
        report = analyze_lerobot.analyze(str(root.parent), None)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("位姿" in e for e in report["errors"]))

    def test_deletion_reindexes_sidecars_and_recomputes_training_statistics(self):
        make_episode(self.source, offset=.01)
        processor.run(self.settings("absolute_base_pose"))
        root = self.root/"absolute_base_pose/lerobot"
        original = PoseDataset(root)
        destination = self.root/"retained"
        (destination/"meta").mkdir(parents=True)
        copy_retained_episodes(root, destination, {0})
        retained = PoseDataset(destination)
        self.assertEqual(len(retained.episodes), 1)
        np.testing.assert_array_equal(retained.episode(0)["poses"], original.episode(1)["poses"])
        self.assertGreater(training_stats(destination, {})["action"]["count"][0], 0)


if __name__ == "__main__":
    unittest.main()
