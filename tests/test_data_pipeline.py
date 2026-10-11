"""Small real HDF5/video fixtures protect the verified action semantics."""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for module in ("PiperRecorder", "PiperViewer", "LerobotDataProcessor", "DataAnalysis"):
    sys.path.insert(0, str(ROOT / "scripts" / module))
from piper_recorder.camera import CAMERA_NAMES, CameraFrame, resize_with_padding
from piper_recorder.robot import ArmCommand, ArmSample
from piper_recorder.dataset import EpisodeData, write_episode
from piper_viewer.server import (
    _plain,
    read_episode,
    read_frame,
    scan_directory,
    _safe_hdf5_path,
)
import processor
import app as processor_app
import analyze_hdf5
import analyze_lerobot
import analysis_app


def make_episode(directory, *, gap=False, offset=0.0, active_side=None,
                 gripper_only=False, frames=27, task_frames=24,
                 last_active=19, image_size=224):
    episode = EpisodeData()
    baseline = [
        0,
        0.2,
        -0.2,
        0,
        0,
        -0.7853981634,
        0.01,
        0,
        0.2,
        -0.2,
        0,
        0,
        0.7853981634,
        0.01,
    ]
    episode.recording_metadata = dict(
        start_baseline=baseline, start_joint_delta_rad=0.01, start_gripper_delta_m=0.002
    )
    for index in range(frames):
        timestamp = 100 + index / 30 + (0.15 if gap and index >= 9 else 0)
        sample = {}
        for side, sign in [("left", -1), ("right", 1)]:
            step = min(index, last_active) if active_side in (None, side) else 0
            joints = (
                0.02 + (0 if gripper_only else step * 0.002) + offset,
                0.2,
                -0.2,
                0,
                0,
                sign * 0.7853981634,
            )
            width = 0.01 + (step * 0.00005 if gripper_only else 0)
            # An unused arm legitimately retains its last accepted command.
            action_time = 99.0 if active_side not in (None, side) else timestamp - 0.001
            command = ArmCommand(joints, width, action_time)
            sample[side] = ArmSample(joints, (0,) * 6, width, 1.0, command)
        frames = {
            name: CameraFrame(
                np.full((image_size, image_size, 3), index * 5 % 256, dtype=np.uint8), timestamp - 0.005
            )
            for name in CAMERA_NAMES
        }
        episode.append(
            frames,
            sample,
            0 if index < task_frames else 1,
            timestamp=timestamp,
            scheduled_timestamp=timestamp,
            capture_duration_s=0.001,
        )
    return write_episode(
        episode,
        "test task",
        directory,
        pyagxarm_version="test",
        firmware_profile="test",
        firmware_version="test",
        gripper_force=1,
        record_hz=30,
    )


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.file = make_episode(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def test_malformed_dataset_has_json_failure_report(self):
        bad = self.root / "bad/lerobot/meta"
        bad.mkdir(parents=True)
        (bad / "info.json").write_text('{"fps": NaN}')
        (bad / "stats.json").write_text("{}")
        result = analyze_lerobot.analyze(str(bad.parent), None)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["errors"])
        json.dumps(result, allow_nan=False)

    def test_writer_viewer_and_analysis_round_trip(self):
        episode = read_episode(self.file, None)
        self.assertEqual(episode["frames"], 27)
        self.assertEqual(len(episode["cameras"]), 3)
        self.assertEqual(len(scan_directory(self.source)["files"]), 1)
        self.assertTrue(
            read_frame(self.file, None, "agentview", 0).startswith(b"\xff\xd8")
        )
        report = analyze_hdf5.analyze_file(self.file, 30, 1.5)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["metrics"]["trailing_idle_frames"], 4)

    def test_viewer_handles_numpy_nan_and_literal_percent_path(self):
        self.assertIsNone(_plain(np.float32(float("nan"))))
        special = self.source / "episode%20name.hdf5"
        self.file.rename(special)
        self.assertEqual(_safe_hdf5_path(str(special)), special)
        json.dumps(_plain({"value": np.array([np.nan, 1])}), allow_nan=False)

    def test_resize_preserves_full_view_with_padding(self):
        rgb = np.full((60, 120, 3), 255, np.uint8)
        result = resize_with_padding(rgb, (224, 224))
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertTrue(np.all(result[:56] == 0))
        self.assertTrue(np.all(result[56:168] == 255))

    def test_phase_grid_trim_and_delta_contract(self):
        settings = processor.Settings(str(self.source), target_fps=10)
        report = processor.inspect_file(self.file, settings)
        self.assertEqual(report["trimmed_idle_frames"], 4)
        with h5py.File(self.file) as handle:
            selected, _, _ = processor._task_sample_indices(handle["data/demo_0"], 3)
        np.testing.assert_array_equal(selected, np.arange(0, 22, 3))
        self.assertEqual(report["output_frames"], 7)

    def test_homing_cadence_does_not_determine_task_fps(self):
        with h5py.File(self.file, "a") as handle:
            demo = handle["data/demo_0"]
            demo["trajectory_phase"][...] = [0] * 8 + [1] * 19
            times = np.r_[np.arange(8) / 30, 7 / 30 + np.arange(1, 20) / 15]
            demo["timestamps"][...] = times
            demo["camera_timestamps"][...] = times[:, None] - np.full((27, 3), .005)
        info = processor.inspect_file(self.file, processor.Settings(str(self.source), target_fps=30))
        self.assertEqual(info["source_fps"], 30)
        self.assertEqual(info["output_frames"], 7)

    def test_home_cannot_supply_missing_task_frames(self):
        with self.assertRaisesRegex(ValueError, "至少需要两帧 task"):
            make_episode(self.root / "one-task-frame", task_frames=1)
        with h5py.File(self.file, "a") as handle:
            handle["data/demo_0/trajectory_phase"][...] = [0] + [1] * 26
        with self.assertRaisesRegex(ValueError, "task 阶段至少需要两帧"):
            processor.inspect_file(self.file, processor.Settings(str(self.source)))

    def test_final_off_grid_change_is_reported_without_crossing_home(self):
        with h5py.File(self.file, "a") as handle:
            demo = handle["data/demo_0"]
            demo["actions"][23, 0] += .01
        info = processor.inspect_file(self.file, processor.Settings(str(self.source)))
        self.assertTrue(any("末次任务变化" in warning for warning in info["warnings"]))
        with h5py.File(self.file) as handle:
            selected, _, _ = processor._task_sample_indices(handle["data/demo_0"], 3)
        self.assertEqual(selected[-1], 21)

    def test_writer_accepts_reused_camera_frames_but_rejects_backwards_time(self):
        from piper_recorder.dataset import _validate_episode
        episode = EpisodeData()
        command = ArmCommand((0,) * 6, 0, 0)
        sample = ArmSample(command.joints, (0,) * 6, 0, 1, command)
        for index in range(2):
            episode.append(
                {name: CameraFrame(np.zeros((16, 16, 3), np.uint8), 0) for name in CAMERA_NAMES},
                {side: sample for side in ("left", "right")}, 0, timestamp=index / 30)
        _validate_episode(episode)
        episode.camera_timestamps[1][0] = -.1
        with self.assertRaisesRegex(ValueError, "不能倒退"):
            _validate_episode(episode)

    def test_single_arm_and_gripper_only_conversion(self):
        for side in ("left", "right"):
            for gripper_only in (False, True):
                source = self.root / f"{side}-{gripper_only}"
                file = make_episode(source, active_side=side, gripper_only=gripper_only,
                                    frames=1000, task_frames=960, last_active=944, image_size=64)
                for mode in ("absolute_joint", "delta_joint"):
                    with self.subTest(side=side, gripper_only=gripper_only, mode=mode):
                        target = self.root / f"{side}-{gripper_only}-{mode}"
                        result = processor.run(processor.Settings(
                            str(source), output_root=str(target), target_fps=30,
                            action_mode=mode, task="single arm test"))
                        self.assertTrue(result["validation"]["valid"], result["validation"])
                        columns, rows = processor._read_parquet_rows(target / "lerobot")
                        with h5py.File(file) as handle:
                            demo = handle["data/demo_0"]
                            selected, _, _ = processor._task_sample_indices(demo, 1)
                            actions = np.asarray(demo["actions"][selected], np.float32)
                            if mode == "delta_joint":
                                actions = np.diff(actions, axis=0)
                            np.testing.assert_array_equal(columns["action"], actions)
                            np.testing.assert_array_equal(columns["observation.state"],
                                np.asarray(demo["states"][selected[:rows]], np.float32))
                        stationary = slice(7, 14) if side == "left" else slice(0, 7)
                        if mode == "delta_joint":
                            np.testing.assert_array_equal(columns["action"][:, stationary], 0)
                        self.assertEqual(rows, 945 - (mode == "delta_joint"))

    def test_statistics_long_constant_and_nearly_constant_axes(self):
        from lerobot.datasets.compute_stats import RunningQuantileStats, aggregate_stats

        values = np.tile(np.array([2.437108, -1.940876, .785398, .01], np.float32), (10000, 1))
        values[::3, 0] += .001
        truth = values.astype(np.float64)
        episode_stats = []
        for part in np.array_split(values, 2):
            accumulator = RunningQuantileStats()
            for batch in np.array_split(part, 5):
                accumulator.update(batch)
            episode_stats.append({"action": accumulator.get_statistics()})
        stats = aggregate_stats(episode_stats)["action"]
        np.testing.assert_allclose(stats["mean"], truth.mean(0), atol=1e-10, rtol=0)
        np.testing.assert_allclose(stats["std"], truth.std(0), atol=2e-6, rtol=0)

    def test_rejects_gaps_and_invalid_fps_without_creating_output(self):
        make_episode(self.source, gap=True)
        with self.assertRaisesRegex(ValueError, "采样间隔异常"):
            processor.scan(processor.Settings(str(self.source)))
        for fps in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                processor.scan(processor.Settings(str(self.source), target_fps=fps))
        self.assertEqual(processor_app.settings_from({"target_fps": 0}).target_fps, 0)
        with self.assertRaises(ValueError):
            processor_app.settings_from({"target_fps": 10.5})

    def test_cancel_before_output_creation(self):
        stop = threading.Event()
        stop.set()
        target = self.root / "cancelled"
        with self.assertRaises(processor.ProcessingCancelled):
            processor.run(
                processor.Settings(str(self.source), output_root=str(target)),
                cancel_event=stop,
            )
        self.assertFalse(target.exists())

    def test_duplicate_source_override_names_are_rejected(self):
        other = self.root / "other"
        other.mkdir()
        (other / self.file.name).touch()
        with self.assertRaisesRegex(ValueError, "同名文件"):
            analyze_lerobot.source_paths(
                [str(self.source), str(other)], [], str(self.source) + "\n" + str(other)
            )

    def test_real_conversion_both_action_modes_and_analysis(self):
        other = self.root / "second"
        make_episode(other, offset=0.01)
        for mode in ("delta_joint", "absolute_joint"):
            target = self.root / mode
            settings = processor.Settings(
                str(self.source),
                input_dirs=[str(self.source), str(other)],
                output_root=str(target),
                action_mode=mode,
                task="test task",
            )
            result = processor.run(settings)
            self.assertTrue(result["validation"]["valid"], result["validation"])
            report = analyze_lerobot.analyze(str(target), None)
            self.assertEqual(report["status"], "PASS", report["errors"])
            self.assertEqual(report["source_verification"]["max_action_error"], 0)
            self.assertEqual(report["source_verification"]["home_frames_in_output"], 0)
            self.assertEqual(report["summary"]["episodes"], 2)

    def test_gap_analysis_and_official_editor_keep_valid_episode(self):
        make_episode(self.source, gap=True)
        target = self.root / "gaps"
        processor.run(
            processor.Settings(
                str(self.source),
                output_root=str(target),
                gap_policy="compact",
                task="test task",
            )
        )
        report = analyze_lerobot.analyze(str(target), None)
        gaps = analysis_app._physical_gap_episodes(report)
        self.assertEqual(gaps, {1})
        with analysis_app.LOCK:
            analysis_app.STATE.update(status="complete", report=report, job_id=99)
        result = analysis_app.delete_error_episodes(
            {"path": str(target), "episodes": [1]}
        )
        self.assertEqual(result["remaining"], 1)
        self.assertTrue(Path(result["backup"]).exists())
        remaining = analyze_lerobot.analyze(str(target), None)
        self.assertEqual(remaining["status"], "PASS", remaining["errors"])
        self.assertEqual(remaining["summary"]["episodes"], 1)


if __name__ == "__main__":
    unittest.main()
