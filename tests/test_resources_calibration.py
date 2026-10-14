"""Filesystem boundaries, HTTP contracts and known-truth calibration geometry."""

import json
import sys
import tempfile
import threading
import unittest
from unittest import mock
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from easy_embodied_deploy.calibration import CalibrationStore, CalibrationRig, sdk_intrinsics, check_sampling_window
from easy_embodied_deploy.calibration_math import (
    DEFAULT_BOARD,
    board_config,
    board_image,
    board_pose,
    detect,
    flange_transform,
    inverse,
    solve_extrinsics,
    solve_intrinsics,
    base_transform, validate_observation, compare_board_observations,
)
from easy_embodied_deploy.files import FileStore
from easy_embodied_deploy.files_console import Handler
from easy_embodied_deploy.calibration_console import Handler as CalibrationHandler


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = FileStore(self.root, references=lambda: [])

    def test_rename_trash_restore_and_permanent_delete(self):
        self.store.mkdir("models", "", "模型 A")
        self.store.mkdir("models", "模型 A", "weights")
        self.store.rename("models", "模型 A", "模型 B")
        item = self.store.trash("models", "模型 B")
        self.assertEqual(self.store.listing("models")["items"], [])
        self.assertEqual(len(self.store.trash_list("models")), 1)
        self.store.restore("models", item["id"])
        self.assertTrue((self.root / "models/模型 B/weights").is_dir())
        item = self.store.trash("models", "模型 B")
        with self.assertRaises(ValueError):
            self.store.purge("models", item["id"], "")
        self.store.purge("models", item["id"], "模型 B")
        self.assertEqual(self.store.trash_list("models"), [])

    def test_symlink_delete_never_follows_and_traversal_fails(self):
        target = self.root / "outside"
        target.mkdir()
        (target / "keep").write_text("keep")
        (self.store.root("datasets") / "link").symlink_to(
            target, target_is_directory=True
        )
        for path in ("../outside", "/etc", "link", "link/keep", ".workbench-trash"):
            with self.assertRaises(ValueError):
                self.store.listing("datasets", path)
        item = self.store.trash("datasets", "link")
        self.store.purge("datasets", item["id"], "link")
        self.assertEqual((target / "keep").read_text(), "keep")

    def test_busy_and_overwrite_are_rejected(self):
        self.store.mkdir("datasets", "", "a")
        self.store.mkdir("datasets", "", "b")
        with self.assertRaises(ValueError):
            self.store.rename("datasets", "a", "b")
        item = self.store.trash("datasets", "b")
        self.store.mkdir("datasets", "", "b")
        with self.assertRaises(ValueError):
            self.store.restore("datasets", item["id"])
        self.store.references = lambda: [(self.root / "datasets/a/file.hdf5", 123)]
        with self.assertRaises(RuntimeError):
            self.store.trash("datasets", "a")
        for method in (
            lambda: self.store.rename("datasets", "", "changed"),
            lambda: self.store.trash("datasets", ""),
        ):
            with self.assertRaises(ValueError):
                method()

    def test_http_mutations_and_origin(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.store = self.store
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join()

        self.addCleanup(cleanup)
        base = f"http://127.0.0.1:{server.server_port}"

        def post(route, body, origin=None):
            headers = {"Content-Type": "application/json"}
            if origin:
                headers["Origin"] = origin
            with urlopen(
                Request(base + route, json.dumps(body).encode(), headers=headers)
            ) as response:
                return json.load(response)

        post("/api/mkdir", dict(root="models", path="", name="from-web"))
        with urlopen(base + "/api/list?root=models") as response:
            self.assertEqual(json.load(response)["items"][0]["name"], "from-web")
        post("/api/rename", dict(root="models", path="from-web", name="renamed"))
        result = post("/api/delete", dict(root="models", path="renamed"))
        post("/api/restore", dict(root="models", id=result["result"]["id"]))
        with self.assertRaises(HTTPError) as exc:
            post(
                "/api/delete",
                dict(root="models", path="renamed"),
                "https://foreign.example",
            )
        self.assertEqual(exc.exception.code, 403)
        self.assertTrue((self.root / "models/renamed").is_dir())


K = np.array([[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]])
CAMERA = dict(
    K=K.tolist(),
    D=[0.0, 0.0, 0.0, 0.0, 0.0],
    model="brown",
    source="test",
    width=640,
    height=480,
)


def synthetic_samples(mounting, count=20):
    rng = np.random.default_rng(24)
    objects = np.zeros((24, 3))
    objects[:, :2] = np.mgrid[0:4, 0:6].T.reshape(-1, 2) * 0.0535
    solution = flange_transform([0.13, -0.08, 0.19, 0.2, -0.3, 0.1])
    constant = flange_transform([0.1, 0.2, 0.7, 0.1, 0.3, -0.2])
    samples = []
    for i in range(count):
        target = flange_transform(
            [
                *rng.uniform(-0.1, 0.05, 2),
                rng.uniform(0.7, 1.2),
                *rng.uniform(-0.5, 0.5, 3),
            ]
        )
        # wrist: B_F F_C C_board = B_board
        # fixed: inv(B_F) B_C C_board = F_board
        pose = (
            constant @ inverse(target) @ inverse(solution)
            if mounting == "wrist"
            else solution @ target @ inverse(constant)
        )
        pixels = cv2.projectPoints(
            objects, cv2.Rodrigues(target[:3, :3])[0], target[:3, 3], K, np.zeros(5)
        )[0].reshape(-1, 2)
        samples.append(
            dict(
                objects=objects.tolist(),
                pixels=pixels.tolist(),
                T_base_flange=pose.tolist(),
            )
        )
    return samples, solution


class GeometryTests(unittest.TestCase):
    def test_default_and_custom_board_detection(self):
        self.assertEqual(board_config({})["square_mm"], 53.5)
        for config in (
            {},
            {"columns": 7, "rows": 5, "square_mm": 24},
            {"kind": "charuco", "columns": 5, "rows": 7},
        ):
            board = board_config(config)
            image = cv2.copyMakeBorder(
                board_image(board, 50), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255
            )
            objects, pixels, ids, overlay = detect(
                cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), board
            )
            self.assertGreaterEqual(len(pixels), 8)
            self.assertEqual(len(objects), len(pixels))
            self.assertEqual(overlay.shape, (*image.shape, 3))
        with self.assertRaises(ValueError):
            board_config({"columns": 3.5})
        with self.assertRaises(ValueError):
            board_config({"square_mm": float("nan")})

    def test_piper_euler_and_inverse(self):
        pose = flange_transform([0.1, 0.2, 0.3, 0.2, 0.3, 0.4])
        rx = cv2.Rodrigues(np.array([0.2, 0.0, 0.0]))[0]
        ry = cv2.Rodrigues(np.array([0.0, 0.3, 0.0]))[0]
        rz = cv2.Rodrigues(np.array([0.0, 0.0, 0.4]))[0]
        np.testing.assert_allclose(pose[:3, :3], rz @ ry @ rx, atol=1e-12)
        np.testing.assert_allclose(pose @ inverse(pose), np.eye(4), atol=1e-12)

    def test_square_chessboard_supports_all_four_origin_directions(self):
        board = board_config({"columns": 4, "rows": 4})
        gray = cv2.copyMakeBorder(
            board_image(board, 50), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255
        )
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        _, baseline, _, _ = detect(rgb, board)
        for turns in (1, 2, 3):
            _, rotated, _, _ = detect(rgb, board, rotation_quarters=turns)
            np.testing.assert_allclose(
                rotated,
                np.rot90(baseline.reshape(4, 4, 2), turns).reshape(-1, 2),
                atol=0.01,
            )
        with self.assertRaises(ValueError):
            detect(rgb, board, rotation_quarters=4)

    def test_both_wrist_and_both_global_directions(self):
        for role, mounting in (
            ("left", "wrist"),
            ("right", "wrist"),
            ("left", "fixed"),
            ("right", "fixed"),
        ):
            with self.subTest(role=role, mounting=mounting):
                samples, expected = synthetic_samples(mounting)
                result = solve_extrinsics(samples, CAMERA, mounting)
                np.testing.assert_allclose(result["matrix"], expected, atol=2e-6)
                self.assertLess(result["translation_rms_mm"], 0.01)
                self.assertEqual(result["warnings"], [])

    def test_degenerate_motion_and_insufficient_samples(self):
        samples, _ = synthetic_samples("wrist")
        with self.assertRaises(ValueError):
            solve_extrinsics(samples[:3], CAMERA, "wrist")
        for s in samples:
            s["T_base_flange"] = np.eye(4).tolist()
        with self.assertRaisesRegex(ValueError, "旋转变化不足"):
            solve_extrinsics(samples, CAMERA, "wrist")

    def test_intrinsic_solver_recovers_known_camera(self):
        samples, _ = synthetic_samples("wrist")
        result = solve_intrinsics(samples, 640, 480)
        np.testing.assert_allclose(result["intrinsics"]["K"], K, atol=0.01)
        self.assertLess(result["rms_px"], 0.001)

    def test_fisheye_pnp_recovers_pose(self):
        samples, _ = synthetic_samples("wrist")
        objects = np.asarray(samples[0]["objects"])
        expected = flange_transform([-0.08, -0.09, 0.9, 0.1, -0.2, 0.05])
        d = np.array([0.01, -0.02, 0.003, -0.001])
        points = cv2.fisheye.projectPoints(
            objects.reshape(-1, 1, 3),
            cv2.Rodrigues(expected[:3, :3])[0],
            expected[:3, 3],
            K,
            d,
        )[0]
        actual, rms = board_pose(
            objects, points, {**CAMERA, "model": "fisheye", "D": d.tolist()}
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertLess(rms, 1e-5)

    def test_sdk_distortion_order_and_unsupported_model(self):
        distortion = SimpleNamespace(
            **{
                name: i / 100
                for i, name in enumerate(
                    ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
                )
            },
            model=SimpleNamespace(name="BROWN_CONRADY_K6"),
        )
        profile = SimpleNamespace(
            get_intrinsic=lambda: SimpleNamespace(
                fx=700, fy=710, cx=320, cy=240, width=640, height=480
            ),
            get_distortion=lambda: distortion,
        )
        self.assertEqual(
            sdk_intrinsics(profile)["D"],
            [0.0, 0.01, 0.06, 0.07, 0.02, 0.03, 0.04, 0.05],
        )
        distortion.model.name = "INVERSE_BROWN_CONRADY"
        bad = sdk_intrinsics(profile)
        samples, _ = synthetic_samples("wrist")
        with self.assertRaises(ValueError):
            board_pose(samples[0]["objects"], samples[0]["pixels"], bad)

    def test_session_solve_export_and_configuration_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            store = CalibrationStore(Path(temp) / "calibration")
            board = dict(DEFAULT_BOARD)
            camera = dict(
                role="left_eye_in_hand",
                serial="test",
                sensor="LEFT_COLOR_SENSOR",
                width=640,
                height=480,
                intrinsics=CAMERA,
            )
            session = store.create("left_wrist", camera, board, "left", "v1")
            board["square_mm"] = 20
            self.assertEqual(store.load(session["id"])["board"]["square_mm"], 53.5)
            session["samples"], expected = synthetic_samples("wrist")
            for index, sample in enumerate(session["samples"]):
                sample["id"] = f"{index:032x}"
            store.save(session)
            store.solve(session["id"])
            result = store.export([session["id"]])
            self.assertTrue(Path(result["path"]).is_file())
            self.assertTrue(result["bundle"]["training_integration"])
            self.assertEqual(len(result["bundle"]["missing_jobs"]), 3)
            store.remove_sample(session["id"], session["samples"][0]["id"])
            self.assertIsNone(store.load(session["id"])["result"])
            with self.assertRaises(ValueError):
                store.export([session["id"]])


class CalibrationHTTPTests(unittest.TestCase):
    def test_capture_persistence_manual_pose_and_identity_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            camera = dict(
                role="left_eye_in_hand",
                serial="fixture",
                sensor="COLOR_SENSOR",
                width=640,
                height=480,
                pose_side="left",
                intrinsics=CAMERA,
            )
            image = board_image(DEFAULT_BOARD, 50)
            image = cv2.copyMakeBorder(
                image, 65, 65, 195, 195, cv2.BORDER_CONSTANT, value=255
            )
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            pose = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]

            class Rig:
                def status(self):
                    return dict(open=True, camera=camera)

                def read(self, with_pose=False):
                    return rgb, {"pose": pose if with_pose else None}, camera

                def capture(self, with_pose=False):
                    frames = []
                    for i in range(3):
                        frame, metadata, _ = self.read(with_pose)
                        metadata.update(camera_received_monotonic_s=10+i*.2, sample_monotonic_s=10+i*.2,
                                        sample_wall_time_s=100+i*.2, image_age_s=0, pose_age_s=0,
                                        image_pose_skew_s=0, pose_sdk_timestamp_s=100+i*.2 if with_pose else None)
                        frames.append((frame, metadata))
                    return rgb, dict(frames[-1][1]), camera, frames

            server = ThreadingHTTPServer(("127.0.0.1", 0), CalibrationHandler)
            server.store, server.rig, server.operations = (
                CalibrationStore(Path(temp)),
                Rig(),
                threading.RLock(),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def post(route, body):
                with urlopen(
                    Request(
                        base + route,
                        json.dumps(body).encode(),
                        {"Content-Type": "application/json"},
                    )
                ) as response:
                    return json.load(response)

            try:
                session = post(
                    "/api/create",
                    dict(
                        job="left_wrist",
                        board=DEFAULT_BOARD,
                        name="http",
                        mount_id="v1",
                    ),
                )
                identifier = session["id"]
                detected = post("/api/detect", dict(id=identifier, reverse=True))
                self.assertEqual(detected["count"], 24)
                self.assertTrue(detected["image"].startswith("data:image/jpeg;base64,"))
                with self.assertRaises(HTTPError):
                    post("/api/capture", dict(id=identifier, still_confirmed=False))
                session = post(
                    "/api/capture", dict(id=identifier, still_confirmed=True)
                )
                sample = session["samples"][0]
                np.testing.assert_allclose(
                    sample["T_base_flange"], flange_transform(pose)
                )
                self.assertEqual(
                    sample["metadata"]["pose_source"], "pyAgxArm_follower_flange"
                )
                with urlopen(
                    base + f"/api/sample?id={identifier}&sample={sample['id']}"
                ) as response:
                    self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                manual = [0.0, 0.1, 0.2, -0.3, 0.4, -0.5]
                session = post(
                    "/api/capture",
                    dict(id=identifier, pose=manual, still_confirmed=True),
                )
                self.assertEqual(
                    session["samples"][-1]["metadata"]["pose_source"], "manual"
                )
                self.assertEqual(server.store.load(identifier)["board"], DEFAULT_BOARD)
                camera["serial"] = "wrong-camera"
                with self.assertRaises(HTTPError):
                    post("/api/capture", dict(id=identifier, still_confirmed=True))
                self.assertEqual(len(server.store.load(identifier)["samples"]), 2)
                post("/api/remove-sample", dict(id=identifier, sample=sample["id"]))
                self.assertFalse(
                    (Path(temp) / identifier / (sample["id"] + ".png")).exists()
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


class SamplingValidityTests(unittest.TestCase):
    def rig(self, pose_age=.02, image_age=.01):
        rig = CalibrationRig()
        rig.camera = SimpleNamespace(read=lambda: SimpleNamespace(rgb=np.zeros((10, 10, 3), np.uint8), timestamp=50-image_age))
        rig.arm = SimpleNamespace(get_flange_pose=lambda: SimpleNamespace(timestamp=123456-pose_age, msg=[0]*6))
        rig.metadata = dict(role='agentview')
        return rig

    def test_clock_domains_are_mapped_before_comparison(self):
        with mock.patch('easy_embodied_deploy.calibration.time.time', return_value=123456), \
             mock.patch('easy_embodied_deploy.calibration.time.monotonic', return_value=50):
            _, meta, _ = self.rig().read(True)
            self.assertAlmostEqual(meta['image_pose_skew_s'], .01)
            self.assertAlmostEqual(meta['pose_age_s'], .02)
            for pose_age, image_age, message in [(1, .01, '过期'), (-1, .01, '时间戳'), (.2, .01, '时间差')]:
                with self.subTest(pose_age=pose_age):
                    with self.assertRaisesRegex(ValueError, message):
                        self.rig(pose_age, image_age).read(True)
            with self.assertRaisesRegex(ValueError, '图像已过期'):
                self.rig(.01, .3).capture(False)

    def frames(self):
        return [(np.zeros((10, 10, 3), np.uint8), dict(
            camera_received_monotonic_s=10+i*.2, sample_monotonic_s=10+i*.2,
            pose_sdk_timestamp_s=100+i*.2, pose=[0]*6, pose_age_s=.01,
            image_age_s=.01, image_pose_skew_s=0)) for i in range(3)]

    def test_window_rejects_motion_and_frozen_feedback(self):
        pixels = np.arange(16).reshape(8, 2).astype(float)
        detection = (np.zeros((8, 3)), pixels, list(range(8)), np.zeros((10, 10, 3), np.uint8))
        with mock.patch('easy_embodied_deploy.calibration.detect', return_value=detection):
            frames = self.frames()
            result = check_sampling_window(frames, DEFAULT_BOARD)
            self.assertTrue(result['pose_verified'])
            self.assertEqual(result['timing_basis'], 'host_receive_not_exposure')
            frames[1][1]['pose'][0] = .003
            with self.assertRaisesRegex(ValueError, '法兰仍在移动'): check_sampling_window(frames, DEFAULT_BOARD)
            frames = self.frames()
            for _, m in frames: m['pose_sdk_timestamp_s'] = 100
            with self.assertRaisesRegex(ValueError, '未持续更新'): check_sampling_window(frames, DEFAULT_BOARD)
            frames = self.frames()
            for _, m in frames: m['camera_received_monotonic_s'] = 10
            with self.assertRaisesRegex(ValueError, '不足三张'): check_sampling_window(frames, DEFAULT_BOARD)
            frames = self.frames()
            for _, m in frames: m['pose'] = None
            self.assertFalse(check_sampling_window(frames, DEFAULT_BOARD)['pose_verified'])
        shifted = (detection[0], pixels+4, detection[2], detection[3])
        with mock.patch('easy_embodied_deploy.calibration.detect', side_effect=[detection, shifted, detection]):
            with self.assertRaisesRegex(ValueError, '图像仍在移动'): check_sampling_window(self.frames(), DEFAULT_BOARD)


class IndependentValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = CalibrationStore(Path(self.temp.name))

    def session(self, job, solution=None):
        mounting = 'wrist' if job.endswith('wrist') else 'fixed'
        samples, original = synthetic_samples(mounting, 24)
        if solution is not None:
            for sample in samples:
                sample['T_base_flange'] = (solution @ inverse(original) @ sample['T_base_flange']).tolist()
        role = 'agentview' if job.startswith('global') else ('left_eye_in_hand' if job == 'left_wrist' else 'right_eye_in_hand')
        camera = dict(role=role, serial='global' if role == 'agentview' else role,
                      sensor='COLOR_SENSOR', width=640, height=480, intrinsics=CAMERA)
        session = self.store.create(job, camera, DEFAULT_BOARD, job, 'v1')
        session['samples'] = samples[:20]
        for i, s in enumerate(session['samples']): s['id'] = f'{i:032x}'
        self.store.save(session)
        return self.store.solve(session['id']), samples[20:]

    def add_validation(self, session, sample, purpose='validation', scene_id=''):
        from easy_embodied_deploy.pose_actions import matrix_rpy
        matrix = np.asarray(sample['T_base_flange'])
        pose = np.r_[matrix[:3, 3], matrix_rpy(matrix[:3, :3])].tolist()
        rgb = np.zeros((480, 640, 3), np.uint8)
        detection = (np.asarray(sample['objects']), np.asarray(sample['pixels']), list(range(len(sample['pixels']))), rgb)
        with mock.patch('easy_embodied_deploy.calibration.detect', return_value=detection):
            return self.store.add_sample(session['id'], rgb, {}, session['camera'], pose,
                                         purpose=purpose, scene_id=scene_id)

    def test_held_out_wrist_and_fixed_detect_bad_pose_without_refitting(self):
        for job in ('left_wrist', 'global_left'):
            with self.subTest(job=job):
                session, held = self.session(job)
                before = json.dumps(session['result'], sort_keys=True)
                session = self.add_validation(session, held[0])
                report = session['validation_samples'][-1]['validation']
                self.assertTrue(report['within_thresholds'])
                self.assertTrue(report['novel_pose'])
                self.assertLess(report['translation_mm'], .01)
                self.assertEqual(json.dumps(session['result'], sort_keys=True), before)
                self.assertEqual(len(session['samples']), 20)
                bad = json.loads(json.dumps(held[1]))
                bad['T_base_flange'][0][3] += .04
                session = self.add_validation(session, bad)
                self.assertFalse(session['validation_samples'][-1]['validation']['within_thresholds'])
                with self.assertRaisesRegex(ValueError, '独立验证超差'): self.store.export([session['id']])
                bundle = self.store.export([session['id']], True)['bundle']
                self.assertEqual(bundle['calibrations'][0]['validation_summary']['status'], 'needs_review')
                sample = session['validation_samples'][-1]
                session = self.store.remove_sample(session['id'], sample['id'])
                self.assertEqual(json.dumps(session['result'], sort_keys=True), before)
                session = self.store.solve(session['id'])
                self.assertEqual(self.store.validation_summary(session)['status'], 'unvalidated')
                self.assertEqual(self.store.validation_summary(session)['stale_sample_count'], 1)

    def test_base_export_direction_inverse_partial_and_mount_mismatch(self):
        left = flange_transform([.3, -.1, .2, .1, .5, -.2])
        right = flange_transform([-.4, .2, .1, -.2, .1, .4])
        a, _ = self.session('global_left', left)
        b, _ = self.session('global_right', right)
        bundle = self.store.export([a['id'], b['id']])['bundle']
        relation = np.asarray(bundle['derived_transforms']['T_left_base_right_base']['matrix'])
        reverse = bundle['derived_transforms']['T_right_base_left_base']['matrix']
        np.testing.assert_allclose(relation @ right, left, atol=3e-6)
        np.testing.assert_allclose(relation @ reverse, np.eye(4), atol=1e-10)
        self.assertFalse(bundle['derived_transforms']['T_left_base_right_base']['independently_validated'])
        self.assertEqual(self.store.export([a['id']])['bundle']['derived_transforms'], {})
        b['mount_id'] = 'v2'; self.store.save(b)
        with self.assertRaisesRegex(ValueError, '安装版本'): self.store.export([a['id'], b['id']])
        b['mount_id'] = 'v1'; b['camera']['serial'] = 'different'; self.store.save(b)
        with self.assertRaisesRegex(ValueError, '同一台相机'): self.store.export([a['id'], b['id']])

    def test_legacy_solution_can_be_validated_without_resolving(self):
        session, held = self.session('left_wrist')
        for key in ('reference_transform', 'reference_frame', 'solution_id'):
            session['result'].pop(key)
        session.pop('validation_samples')
        self.store.save(session)
        updated = self.add_validation(session, held[0])
        self.assertEqual(updated['result'], session['result'])
        self.assertEqual(self.store.validation_summary(updated)['status'], 'passed')

    def test_intrinsic_validation_uses_solved_intrinsics_and_stays_held_out(self):
        samples, _ = synthetic_samples('wrist', 24)
        session = self.store.create('intrinsics', dict(
            role='agentview', serial='test', sensor='COLOR_SENSOR', width=640,
            height=480, intrinsics=CAMERA), DEFAULT_BOARD, 'intrinsics', 'v1')
        session['samples'] = samples[:20]
        self.store.save(session)
        session = self.store.solve(session['id'])
        updated = self.add_validation(session, samples[20])
        self.assertEqual(updated['result'], session['result'])
        self.assertEqual(len(updated['samples']), 20)
        report = updated['validation_samples'][-1]['validation']
        self.assertTrue(report['within_thresholds'])
        self.assertNotIn('translation_mm', report)

    def test_dual_base_check_uses_independent_observations_and_invalidates(self):
        from easy_embodied_deploy.calibration_math import board_pose
        left_global = flange_transform([.3, .1, .2, .1, -.2, .1])
        right_global = flange_transform([-.4, .2, .3, -.2, .1, .3])
        left, _ = self.session('left_wrist')
        right, _ = self.session('right_wrist')
        gl, _ = self.session('global_left', left_global)
        gr, _ = self.session('global_right', right_global)
        relation = base_transform(left_global, right_global)
        board_left = flange_transform([.4, .1, .8, .1, -.3, .2])
        observations = []
        for s, board in ((left, board_left), (right, inverse(relation) @ board_left)):
            sample = synthetic_samples('wrist', 24)[0][-1]
            target = board_pose(sample['objects'], sample['pixels'], CAMERA)[0]
            sample['T_base_flange'] = (board @ inverse(target) @ inverse(s['result']['matrix'])).tolist()
            updated = self.add_validation(s, sample, 'dual_validation', 'fixed-board-1')
            self.assertEqual(self.store.validation_summary(updated)['sample_count'], 0)
            observations.append(updated['validation_samples'][-1])
        ids = [s['id'] for s in (left, right, gl, gr)]
        report = self.store.compare_bases(ids, observations[0]['id'], observations[1]['id'], True)
        self.assertTrue(report['within_thresholds'])
        self.assertLess(report['translation_mm'], .01)
        bundle = self.store.export(ids)['bundle']
        from easy_embodied_deploy.pose_actions import calibration_for_mode
        for mode in ('camera_delta_wrist', 'camera_delta_global'):
            self.assertEqual(set(calibration_for_mode(bundle, mode, 'v1')), {'left', 'right'})
        self.assertTrue(bundle['derived_transforms']['T_left_base_right_base']['independently_validated'])
        self.assertEqual(len(bundle['dual_base_validation']), 1)
        with self.assertRaises(ValueError): self.store.compare_bases(ids, observations[0]['id'], observations[1]['id'], False)
        right = self.store.load(right['id'])
        original_rms = right['validation_samples'][0]['validation']['reprojection_rms_px']
        right['validation_samples'][0]['validation']['reprojection_rms_px'] = 3
        self.store.save(right)
        self.assertFalse(self.store.compare_bases(ids, observations[0]['id'], observations[1]['id'], True)['within_thresholds'])
        right['validation_samples'][0]['validation']['reprojection_rms_px'] = original_rms
        self.store.save(right)
        # Corrupt an independent pose, not the matrices used to derive the relation.
        right = self.store.load(right['id'])
        right['validation_samples'][0]['T_base_flange'][0][3] += .03
        self.store.save(right)
        bad = self.store.compare_bases(ids, observations[0]['id'], observations[1]['id'], True)
        self.assertFalse(bad['within_thresholds'])
        with self.assertRaisesRegex(ValueError, '双臂独立'): self.store.export(ids)
        self.store.remove_sample(right['id'], observations[1]['id'])
        self.assertEqual(self.store.export(ids)['bundle']['dual_base_validation'], [])
        self.store.save(right)  # Restore the observation to exercise solution invalidation too.
        self.store.solve(left['id'])
        with self.assertRaisesRegex(ValueError, '已失效'):
            self.store.compare_bases(ids, observations[0]['id'], observations[1]['id'], True)
        self.assertEqual(self.store.export(ids)['bundle']['dual_base_validation'], [])


if __name__ == "__main__":
    unittest.main()
