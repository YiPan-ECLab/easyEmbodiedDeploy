"""Review individual source failures, scoped timing consent and reversible deletion."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from test_data_pipeline import make_episode, processor, analyze_lerobot
from easy_embodied_deploy.trajectory_files import revision, trash_trajectory, restore_trajectory, hold_trajectories


class TrajectoryReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "raw"
        self.good = make_episode(self.source)
        self.gap = make_episode(self.source, gap=True)
        self.settings = processor.Settings(str(self.source), target_fps=30)

    def test_collect_all_files_and_allow_only_reviewed_timing(self):
        corrupt = self.source / "broken.hdf5"
        corrupt.write_bytes(b"not hdf5")
        plan = processor.scan(self.settings, collect_errors=True)
        self.assertEqual(plan["episodes"], 3)
        self.assertEqual(plan["invalid_files"], 2)
        rows = {row["name"]: row for row in plan["files"]}
        self.assertFalse(rows[corrupt.name]["can_ignore_timing"])
        self.assertTrue(rows[self.gap.name]["can_ignore_timing"])
        approved = replace(self.settings, timing_overrides={str(self.gap): revision(self.gap), str(corrupt): revision(corrupt)})
        plan = processor.scan(approved, collect_errors=True)
        self.assertEqual(plan["invalid_files"], 1)
        self.assertEqual(plan["sampling_fps"], "mixed")
        with self.assertRaises(ValueError):
            processor.run(approved, prepared_plan=plan)
        corrupt.unlink()
        self.assertEqual(processor.scan(approved)["invalid_files"], 0)
        with h5py.File(self.gap, "r+") as h:
            h.attrs["changed"] = 1
        with self.assertRaises(processor.TimingError):
            processor.scan(approved)

    def test_numeric_errors_cannot_be_ignored(self):
        with h5py.File(self.gap, "r+") as h:
            h["data/demo_0/obs/left_ee_states"][3, 0] = np.nan
        approved = replace(self.settings, timing_overrides={str(self.gap): revision(self.gap)})
        plan = processor.scan(approved, collect_errors=True)
        row = next(x for x in plan["files"] if x["name"] == self.gap.name)
        self.assertFalse(row["can_ignore_timing"])
        self.assertIn("NaN/Inf", row["errors"][0])

    def test_delete_restore_version_and_active_read_protection(self):
        token = revision(self.good)
        with self.assertRaisesRegex(ValueError, "文件已变化"):
            trash_trajectory(self.good, "stale")
        with hold_trajectories([(self.good, token)]):
            with self.assertRaisesRegex(RuntimeError, "正在"):
                trash_trajectory(self.good, token)
        result = trash_trajectory(self.good, token)
        self.assertFalse(self.good.exists())
        restore_trajectory(self.good, result["id"])
        self.assertEqual(revision(self.good), token)
        with self.assertRaises(ValueError):
            trash_trajectory(self.source, token)
        link = self.source / "link.hdf5"
        link.symlink_to(self.good)
        with self.assertRaises(ValueError):
            trash_trajectory(link, token)

    def test_real_joint_and_pose_conversion_preserves_physical_time(self):
        # One source with a real time gap, approved explicitly. No hardware or
        # existing user recordings are touched by conversion or deletion tests.
        self.good.unlink()
        with h5py.File(self.gap, "r") as h:
            original = h["data/demo_0/timestamps"][:]
        for mode in ("delta_joint", "absolute_base_pose"):
            settings = replace(self.settings, action_mode=mode, output_root=str(self.root / mode),
                               timing_overrides={str(self.gap): revision(self.gap)})
            with patch.object(processor, "available_cpu_count", return_value=1):
                result = processor.run(settings)
            self.assertTrue(result["validation"]["valid"], result["validation"])
            self.assertTrue(result["plan"]["files"][0]["timing_ignored"])
            report = analyze_lerobot.analyze(str(self.root / mode), None)
            self.assertEqual(report["status"], "PASS", report["errors"])
            if mode == "absolute_base_pose":
                from easy_embodied_deploy.pose_dataset import PoseDataset
                data = PoseDataset(self.root / mode / "lerobot")
                self.assertEqual(data.contract["fps"], "mixed")
                arrays = data.episode(0)
                np.testing.assert_array_equal(arrays["timestamps"], original[arrays["source_indices"]])
        with h5py.File(self.gap, "r") as h:
            np.testing.assert_array_equal(h["data/demo_0/timestamps"][:], original)


if __name__ == "__main__":
    unittest.main()
