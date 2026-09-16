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


def make_episode(directory, *, gap=False, offset=0.0):
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
    for index in range(27):
        timestamp = 100 + index / 30 + (0.15 if gap and index >= 9 else 0)
        sample = {}
        for side, sign in [("left", -1), ("right", 1)]:
            joints = (
                0.02 + min(index, 19) * 0.002 + offset,
                0.2,
                -0.2,
                0,
                0,
                sign * 0.7853981634,
            )
            command = ArmCommand(joints, 0.01, timestamp - 0.001)
            sample[side] = ArmSample(joints, (0,) * 6, 0.01, 1.0, command)
        frames = {
            name: CameraFrame(
                np.full((224, 224, 3), index * 5, dtype=np.uint8), timestamp - 0.005
            )
            for name in CAMERA_NAMES
        }
        episode.append(
            frames,
            sample,
            0 if index < 24 else 1,
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
