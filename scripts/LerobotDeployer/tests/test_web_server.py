import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest
from pathlib import Path

import web_server


DEFAULT_CONFIG = web_server.DEFAULT_CONFIG
bridge_command = web_server.bridge_command
deployment_command = web_server.deployment_command
discover_checkpoints = web_server.discover_checkpoints
normalize_config = web_server.normalize_config


class WebConsoleConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        checkpoint = root / "models/finetuned_models/test/checkpoint"
        checkpoint.mkdir(parents=True)
        cameras = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
        config = {
            "type": "pi05",
            "chunk_size": 50,
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [14]},
                **{
                    f"observation.images.{camera}": {
                        "type": "VISUAL",
                        "shape": [3, 224, 224],
                    }
                    for camera in cameras
                },
            },
            "output_features": {"action": {"type": "ACTION", "shape": [14]}},
        }
        (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (checkpoint / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
        (checkpoint / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
        cls._old_root = web_server.ROOT
        cls._old_checkpoint = DEFAULT_CONFIG["checkpoint"]
        web_server.ROOT = root
        DEFAULT_CONFIG["checkpoint"] = str(checkpoint)

    @classmethod
    def tearDownClass(cls) -> None:
        web_server.ROOT = cls._old_root
        DEFAULT_CONFIG["checkpoint"] = cls._old_checkpoint
        cls.temp.cleanup()

    def test_verified_defaults_build_the_two_process_commands(self) -> None:
        config = normalize_config(DEFAULT_CONFIG)
        bridge, _ = bridge_command(config["bridge_endpoint"])
        deployer, _ = deployment_command(config)

        self.assertEqual(bridge.count("--endpoint"), 1)
        self.assertNotIn("--control-hz", bridge)
        self.assertNotIn("--smoothing-time-constant", bridge)
        self.assertEqual(deployer[deployer.index("--policy-hz") + 1], "10.0")
        self.assertEqual(deployer[deployer.index("--control-hz") + 1], "30.0")
        self.assertEqual(deployer[deployer.index("--device") + 1], "cuda:0")
        self.assertEqual(deployer[deployer.index("--max-steps") + 1], "0")
        self.assertEqual(
            deployer[deployer.index("--smoothing-time-constant") + 1], "0.1"
        )
        self.assertIn("--diagnostics-jsonl", deployer)
        self.assertNotIn("--dataset-root", deployer)
        self.assertNotIn("dataset_root", config)

    def test_action_mode_is_explicit_and_unknown_fields_are_ignored(self) -> None:
        config = normalize_config(
            {
                **DEFAULT_CONFIG,
                "action_mode": "absolute_joint",
                "removed_option": "/path/that/does/not/exist",
            }
        )
        self.assertEqual(config["action_mode"], "absolute_joint")
        self.assertNotIn("removed_option", config)

    def test_single_gpu_device_is_preserved_in_deployer_command(self) -> None:
        config = normalize_config({**DEFAULT_CONFIG, "device": "cuda:2"})
        deployer, _ = deployment_command(config)
        self.assertEqual(deployer[deployer.index("--device") + 1], "cuda:2")

        with self.assertRaisesRegex(Exception, "cpu、cuda 或 cuda:N"):
            normalize_config({**DEFAULT_CONFIG, "device": "gpu:2"})

    def test_repository_options_are_discoverable(self) -> None:
        checkpoints = discover_checkpoints()
        self.assertTrue(any(item["path"] == DEFAULT_CONFIG["checkpoint"] for item in checkpoints))


if __name__ == "__main__":
    unittest.main()
