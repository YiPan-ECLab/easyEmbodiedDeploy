"""Filesystem boundaries, HTTP contracts and known-truth calibration geometry."""

import json
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from easy_embodied_deploy.calibration import CalibrationStore, sdk_intrinsics
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


if __name__ == "__main__":
    unittest.main()
