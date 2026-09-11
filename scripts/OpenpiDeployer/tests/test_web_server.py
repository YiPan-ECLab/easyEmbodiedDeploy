from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))

import web_server  # noqa: E402
from web_server import (  # noqa: E402
    DEFAULT_CONFIG,
    bridge_command,
    deployment_command,
    discover_checkpoints,
    normalize_config,
)


class WebConsoleConfigurationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.normal = self._checkpoint("normal", action_mode="delta_joint", rtc_max_delay=0)
        self.trained = self._checkpoint("trained", action_mode="delta_joint", rtc_max_delay=5)
        self.defaults = {**DEFAULT_CONFIG, "checkpoint": str(self.normal), "action_mode": "delta_joint"}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _checkpoint(self, name: str, *, action_mode: str, rtc_max_delay: int) -> Path:
        checkpoint = self.root / "models/finetuned_models" / name / "1"
        (checkpoint / "params").mkdir(parents=True)
        (checkpoint / "params/_METADATA").write_text("{}")
        (checkpoint / "assets/local__test").mkdir(parents=True)
        manifest = {
            "trainer": "OpenpiTrainer",
            "config": {
                "output_dir": str(checkpoint.parent),
                "action_mode": action_mode,
                "action_horizon": 50,
                "rtc_max_delay": rtc_max_delay,
                "finetune_mode": "lora",
            },
        }
        (checkpoint / "training_manifest.json").write_text(json.dumps(manifest))
        return checkpoint

    def test_verified_defaults_build_isolated_openpi_and_current_bridge_commands(self) -> None:
        config = normalize_config(self.defaults)
        bridge, _ = bridge_command(config["bridge_endpoint"])
        deployer, _ = deployment_command(config)
        self.assertIn("thirdParty/.venvs/piper-home/bin/python", bridge[0])
        self.assertIn("scripts/LerobotDeployer/hardware_bridge.py", bridge[1])
        self.assertIn("thirdParty/.venvs/openpi-jax/bin/python", deployer[0])
        self.assertIn("scripts/OpenpiDeployer/OpenpiDeployer.py", deployer[1])
        self.assertEqual(deployer[deployer.index("--device") + 1], "cuda:1")
        self.assertEqual(deployer[deployer.index("--sample-steps") + 1], "10")
        self.assertIn("--rtc-mask-grippers", deployer)
        self.assertIn("--warmup", deployer)

    def test_checkpoint_action_mode_is_enforced(self) -> None:
        with self.assertRaisesRegex(Exception, "checkpoint 动作模式"):
            normalize_config({**self.defaults, "action_mode": "absolute_joint"})

    def test_single_physical_gpu_is_preserved(self) -> None:
        config = normalize_config({**self.defaults, "device": "cuda:2"})
        deployer, _ = deployment_command(config)
        self.assertEqual(deployer[deployer.index("--device") + 1], "cuda:2")
        with self.assertRaisesRegex(Exception, "cpu、cuda 或 cuda:N"):
            normalize_config({**self.defaults, "device": "gpu:2"})

    def test_trained_rtc_requires_trained_checkpoint(self) -> None:
        with self.assertRaisesRegex(Exception, "没有 trained RTC"):
            normalize_config({**self.defaults, "rtc_mode": "trained"})
        config = normalize_config({**self.defaults, "checkpoint": str(self.trained), "rtc_mode": "trained"})
        self.assertEqual(config["rtc_mode"], "trained")

    def test_repository_checkpoints_are_discoverable(self) -> None:
        with mock.patch.object(web_server, "ROOT", self.root):
            checkpoints = discover_checkpoints()
        self.assertTrue(any(item["path"] == str(self.normal) for item in checkpoints))

    def test_discovered_checkpoint_exposes_action_contract(self) -> None:
        with mock.patch.object(web_server, "ROOT", self.root):
            checkpoint = next(item for item in discover_checkpoints() if item["path"] == str(self.normal))
        self.assertEqual(checkpoint["action_mode"], "delta_joint")
        self.assertEqual(checkpoint["n_action_steps"], 10)

    def test_live_diagnostics_separate_prefill_from_runtime_starvation(self) -> None:
        path = self.root / "diagnostics.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"tick": 1, "tick_s": 0.1, "queue_size": 0, "action": None}),
                    json.dumps({"tick": 2, "tick_s": 0.1, "queue_size": 3, "action": [1.0]}),
                    json.dumps({"tick": 3, "tick_s": 0.1, "queue_size": 0, "action": None}),
                ]
            )
            + "\n"
        )
        supervisor = web_server.ProcessSupervisor()
        supervisor._diagnostics_path = path
        diagnostics = supervisor.diagnostics()
        self.assertEqual(diagnostics["prefill_starved_ticks"], 1)
        self.assertEqual(diagnostics["runtime_starved_ticks"], 1)
        self.assertEqual(diagnostics["starved_ticks"], 2)


if __name__ == "__main__":
    unittest.main()
