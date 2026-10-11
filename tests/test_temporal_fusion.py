"""Real HDF5 -> fusion -> all action modes, including video and provenance."""

import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import h5py
import numpy as np

from test_data_pipeline import make_episode, processor, analyze_lerobot
from test_pose_actions import calibration
from easy_embodied_deploy import temporal_fusion as fusion
from easy_embodied_deploy.pose_dataset import PoseDataset, new_contract
from easy_embodied_deploy.target_runtime import checkpoint_pose_contract
from piper_viewer.server import ViewerServer


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = make_episode(self.root/"raw", active_side="left", image_size=224)
        self.demo = "/data/demo_0"
        self.plan = dict(base_stride=1, playback_fps=30,
                         segments=[dict(start=3, end=12, factor=3)])
        self.expected = np.array([0, 1, 2, 3, 6, 9, 12, *range(13, 24)])
        with h5py.File(self.source, "a") as handle:
            pose = handle[self.demo+"/obs/left_ee_states"]
            values = np.zeros(pose.shape)
            values[:, :3] = [.2, .1, .3]
            values[:, 0] += np.arange(len(values))*.001
            values[:, 5] = np.arange(len(values))*.003
            pose[...] = values

    def tearDown(self):
        self.temp.cleanup()

    def test_selection_boundaries_base_stride_and_invalid_ranges(self):
        result = fusion.preview(self.source, self.demo, self.plan)
        np.testing.assert_array_equal(result["indices"], self.expected)
        plan = dict(base_stride=3, playback_fps=10, segments=[dict(start=4, end=17, factor=2)])
        result = fusion.preview(self.source, self.demo, plan)
        self.assertEqual(result["indices"], [0, 3, 4, 10, 16, 17, 18, 21, 23])
        for segments in ([dict(start=1, end=24, factor=2)],
                         [dict(start=1, end=8, factor=2), dict(start=7, end=12, factor=3)],
                         [dict(start=2, end=2, factor=3)]):
            with self.assertRaises(ValueError):
                fusion.preview(self.source, self.demo, dict(segments=segments))
        adjacent = dict(segments=[dict(start=0, end=9, factor=100), dict(start=9, end=23, factor=100)])
        self.assertEqual(fusion.preview(self.source, self.demo, adjacent)["indices"], [0, 9, 23])

    def test_export_alignment_source_immutable_and_repeated_edit_provenance(self):
        before = hashlib.sha256(self.source.read_bytes()).digest()
        result = fusion.export_episode(self.source, self.demo, self.plan)
        with h5py.File(self.source, "r") as source, h5py.File(result["path"], "r") as output:
            raw, out = source[self.demo], output[self.demo]
            self.assertEqual(out.attrs["fps"], "mixed")
            np.testing.assert_array_equal(out["timestamps"], raw["timestamps"][self.expected])
            self.assertEqual(out["states"].id, out["robot_states"].id)
            for key in ["actions", "states", "obs/agentview", "obs/left_eye_in_hand",
                        "obs/right_eye_in_hand", "obs/left_ee_states", "obs/right_ee_states"]:
                np.testing.assert_array_equal(out[key], raw[key][self.expected])
            self.assertTrue(np.all(out["trajectory_phase"][:] == 0))
            self.assertEqual(np.flatnonzero(out["dones"][:]).tolist(), [len(self.expected)-1])
            for name in fusion.TIME_FIELDS:
                if name in raw:
                    np.testing.assert_array_equal(out[f"{fusion.PROVENANCE}/{name}"], raw[name][self.expected])
            np.testing.assert_array_equal(out[f"{fusion.PROVENANCE}/source_frame_indices"], self.expected)
        second = fusion.export_episode(result["path"], self.demo, dict(base_stride=2, playback_fps=7))
        with h5py.File(second["path"], "r") as handle:
            out = handle[self.demo]
            selected = np.unique(np.r_[np.arange(0, len(self.expected), 2), len(self.expected)-1])
            np.testing.assert_array_equal(out[f"{fusion.PROVENANCE}/source_frame_indices"], self.expected[selected])
            self.assertIsNotNone(fusion.metadata(out)["previous"])
        self.assertEqual(before, hashlib.sha256(self.source.read_bytes()).digest())

    def test_draft_and_http_roundtrip_and_validation(self):
        server = ViewerServer(("127.0.0.1", 0), self.root/"raw")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            payload = dict(path=str(self.source), demo="demo_0", plan=self.plan)
            def post(endpoint, body):
                request = Request(base+endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urlopen(request) as response:
                    return json.load(response)
            saved = post("/api/fusion/save", payload)
            self.assertTrue(Path(saved["draft_path"]).exists())
            self.assertEqual(fusion.load_plan(self.source, self.demo)["indices"], self.expected.tolist())
            exported = post("/api/fusion/export", payload)
            self.assertTrue(Path(exported["path"]).is_file())
            with self.assertRaises(HTTPError) as context:
                post("/api/fusion/preview", dict(payload, plan=dict(base_stride=0)))
            self.assertEqual(context.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_all_action_modes_keep_sequence_at_unrelated_fps_and_recompute_targets(self):
        with h5py.File(self.source, "a") as handle:
            for key in ("actions", "states", "obs/left_ee_states", "obs/right_ee_states"):
                field = handle[self.demo + "/" + key]
                field[:4] = np.repeat(field[0][None], 4, axis=0)
        exported = fusion.export_episode(self.source, self.demo, self.plan)
        calib = self.root/"calibration.json"
        calib.write_text(json.dumps(calibration()))
        for mode in ("absolute_joint", "delta_joint", "absolute_base_pose", "camera_delta_wrist", "camera_delta_global"):
            with self.subTest(mode=mode):
                settings = processor.Settings(exported["output_directory"], output_root=str(self.root/mode),
                    target_fps=17, source_fps=99, action_mode=mode,
                    calibration_bundle=str(calib), calibration_mount_id="synthetic-unit-test")
                report = processor.run(settings)
                self.assertTrue(report["validation"]["valid"], report["validation"])
                root = self.root/mode/"lerobot"
                kept = self.expected if mode in processor.POSE_MODES else self.expected[self.expected <= 19]
                kept = kept[kept >= 3]
                count = len(kept) - int(mode != "absolute_joint")
                self.assertEqual(report["plan"]["files"][0]["trimmed_leading_idle_frames"], 3)
                info = json.loads((root/"meta/info.json").read_text())
                self.assertEqual(info["total_frames"], count)
                self.assertEqual(info["fps"], "mixed")
                self.assertEqual(info["video_fps"], 17)
                analyzed = analyze_lerobot.analyze(str(root.parent), None)
                self.assertEqual(analyzed["status"], "PASS", analyzed["errors"])
                self.assertEqual(analyzed["source_verification"]["max_state_error"], 0)
                from scripts.LerobotTrainer import trainer as torch_trainer
                from scripts.OpenpiTrainer import trainer as jax_trainer
                for trainer in (torch_trainer, jax_trainer):
                    checked = trainer.validate_dataset(root, normalization_mode="quantiles", action_mode=mode)
                    self.assertEqual(checked["fps"], "mixed")
                from easy_embodied_deploy.pose_training_torch import PoseLeRobotDataset
                from scripts.OpenpiTrainer import dataset_v3
                dataset_v3.VIDEO_BACKEND = "pyav"
                torch_data = PoseLeRobotDataset("local/fusion", root=root, download_videos=False,
                    delta_timestamps={"action": [i/17 for i in range(50)]}, video_backend="pyav")
                jax_data = dataset_v3.LeRobotV3Dataset(root, 50)
                torch_sample, jax_sample = torch_data[2], jax_data[2]
                np.testing.assert_allclose(torch_sample["action"], jax_sample["action"], atol=1e-6)
                for camera in ("agentview", "left_eye_in_hand", "right_eye_in_hand"):
                    key = f"observation.images.{camera}"
                    np.testing.assert_allclose(torch_sample[key], jax_sample[key], atol=1e-6)
                torch_data.finalize()
                if mode in processor.POSE_MODES:
                    data = PoseDataset(root)
                    with h5py.File(self.source, "r") as raw:
                        np.testing.assert_array_equal(data.episode(0)["poses"][:, :6], raw[self.demo+"/obs/left_ee_states"][kept])
                    chunk, valid = data.chunk(2)
                    arrays = data.episode(0)
                    for i in range(int(valid.sum())):
                        target, grip = data.codec.decode(arrays["poses"][2], chunk[i])
                        np.testing.assert_allclose(target, arrays["poses"][3+i], atol=1e-6)
                    self.assertEqual(data.chunk(count-1)[1].sum(), 1)

    def test_deployment_clock_is_independent(self):
        contract = new_contract("absolute_base_pose", 30, None, "")
        (self.root/"training_manifest.json").write_text(json.dumps(dict(
            config=dict(action_mode="absolute_base_pose"), pose_contract=contract)))
        actual = checkpoint_pose_contract(self.root, "absolute_base_pose", policy_hz=17)
        self.assertEqual(actual["fps"], 30)

    def test_fused_trim_edges_keep_internal_pauses_single_arm_and_home_boundary(self):
        with h5py.File(self.source, "a") as handle:
            demo = handle[self.demo]
            demo.attrs["temporal_fusion"] = json.dumps(dict(version=1))
            baseline = demo["actions"][0]
            for channel in (0, 7, 6, 13):
                with self.subTest(channel=channel):
                    actions = np.repeat(baseline[None], 27, axis=0)
                    steps = np.clip(np.arange(27)-3, 0, 16)
                    steps[8:11] = steps[7]  # Deliberate middle pause must survive.
                    actions[:, channel] += steps * .0001
                    actions[24:, channel] += .01  # Home movement cannot extend task.
                    demo["actions"][...] = actions
                    selected, home, tail = processor._task_sample_indices(demo, 5)
                    np.testing.assert_array_equal(selected, np.arange(3, 20))
                    self.assertEqual((home, tail), (3, 4))
                    verified, _ = analyze_lerobot.source_indices(demo, 5)
                    np.testing.assert_array_equal(verified, selected)
            # A stationary task has one anchor, never targets from HOME.
            demo["actions"][:24] = np.repeat(baseline[None], 24, axis=0)
            selected, home, tail = processor._task_sample_indices(demo, 1)
            np.testing.assert_array_equal(selected, [0])
            self.assertEqual((home, tail), (3, 23))

    def test_pose_edges_retain_follower_motion_after_command_stops(self):
        with h5py.File(self.source, "a") as handle:
            demo = handle[self.demo]
            demo.attrs["temporal_fusion"] = json.dumps(dict(version=1))
            for key in ("actions", "states", "obs/left_ee_states", "obs/right_ee_states"):
                values = demo[key][:]
                values[:4] = values[0]
                values[20:24] = values[19]
                demo[key][...] = values
            # Follower continues moving through frame 21; retain it in pose modes.
            poses = demo["obs/left_ee_states"][:]
            poses[20:24, 0] = poses[19, 0] + np.array([.001, .002, .002, .002])
            demo["obs/left_ee_states"][...] = poses
            joint, _, _ = processor._task_sample_indices(demo, 1)
            pose, home, tail = processor._task_sample_indices(demo, 5, pose=True)
            np.testing.assert_array_equal(joint, np.arange(3, 20))
            np.testing.assert_array_equal(pose, np.arange(3, 22))
            self.assertEqual((home, tail), (3, 2))


if __name__ == "__main__":
    unittest.main()
