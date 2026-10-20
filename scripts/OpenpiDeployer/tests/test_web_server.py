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
        self.defaults = {**DEFAULT_CONFIG, "checkpoint": str(self.normal), "action_mode": "delta_joint", "rtc_mode": "guided"}

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
        config = normalize_config({**DEFAULT_CONFIG, "checkpoint": str(self.trained), "action_mode": "delta_joint"})
        bridge, _ = bridge_command(config["bridge_endpoint"])
        deployer, _ = deployment_command(config)
        self.assertIn("thirdParty/.venvs/piper-home/bin/python", bridge[0])
        self.assertIn("scripts/LerobotDeployer/hardware_bridge.py", bridge[1])
        self.assertIn("thirdParty/.venvs/openpi-jax/bin/python", deployer[0])
        self.assertIn("scripts/OpenpiDeployer/OpenpiDeployer.py", deployer[1])
        self.assertEqual(deployer[deployer.index("--policy-hz") + 1], "30.0")
        self.assertEqual(deployer[deployer.index("--control-hz") + 1], "150.0")
        self.assertEqual(deployer[deployer.index("--rtc-mode") + 1], "trained")
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

    def test_migrated_manifest_falls_back_to_local_checkpoint_directory(self) -> None:
        manifest_path = self.normal / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["output_dir"] = str(self.root / "old-server-only" / "run")
        manifest_path.write_text(json.dumps(manifest))
        with mock.patch.object(web_server, "ROOT", self.root):
            checkpoints = discover_checkpoints()
        self.assertTrue(any(item["path"] == str(self.normal) for item in checkpoints))

    def test_gpu_status_reports_driver_error(self) -> None:
        supervisor = web_server.ProcessSupervisor()
        result = mock.Mock(returncode=9, stdout="", stderr="driver unavailable")
        with mock.patch.object(web_server.subprocess, "run", return_value=result):
            self.assertEqual(supervisor._gpu_status(), [])
        self.assertEqual(supervisor._gpu_error, "driver unavailable")

    def test_gpu_status_parses_valid_rows(self) -> None:
        supervisor = web_server.ProcessSupervisor()
        result = mock.Mock(
            returncode=0,
            stdout="0, NVIDIA Test, 10, 100, 20\n1, NVIDIA Test 2, 0, 200, 0\n",
            stderr="",
        )
        with mock.patch.object(web_server.subprocess, "run", return_value=result):
            rows = supervisor._gpu_status()
        self.assertEqual([row["index"] for row in rows], [0, 1])
        self.assertIsNone(supervisor._gpu_error)

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

    def test_live_cumulative_metrics_survive_tail_window_without_final_summary(self) -> None:
        from OpenpiDeployer import Diagnostics
        import numpy as np

        path = self.root / "live.jsonl"
        writer = Diagnostics(path)
        writer.record(action=None, queue_size=0, tick_s=.01)
        writer.record(action=np.zeros(14), queue_size=5, tick_s=.01)
        writer.record(action=None, queue_size=0, tick_s=.01)
        for _ in range(300):
            writer.record(action=np.zeros(14), queue_size=3, tick_s=.01,
                          inference={"inference_chunks": 7, "inference_latency_s_median": .023,
                                     "inference_latency_s_last": .025, "inference_latency_s_max": .04},
                          move_js={"achieved_control_hz": 29.8})
        supervisor = web_server.ProcessSupervisor()
        supervisor._diagnostics_path = path
        diagnostics = supervisor.diagnostics()
        self.assertIsNone(diagnostics["runtime_summary"])
        self.assertEqual(len(diagnostics["points"]), 240)
        self.assertEqual(diagnostics["prefill_starved_ticks"], 1)
        self.assertEqual(diagnostics["runtime_starved_ticks"], 1)
        self.assertEqual(diagnostics["action_ticks"], 301)
        self.assertEqual(diagnostics["minimum_queue"], 0)
        self.assertEqual(diagnostics["live_summary"]["inference_latency_s_median"], .023)
        self.assertEqual(diagnostics["live_summary"]["move_js"]["achieved_control_hz"], 29.8)
        self.assertEqual(diagnostics["points"][-1]["inference_ms"], 25)
        writer.close()
        writer.save_summary(writer.summary())
        self.assertEqual(json.loads(path.with_suffix(".summary.json").read_text())["runtime_starved_ticks"], 1)

    def test_web_blank_diagnostic_path_allocates_unique_persistent_runs(self) -> None:
        supervisor = web_server.ProcessSupervisor()
        config = {**self.defaults, "diagnostics_jsonl": "", "offline_dataset_root": str(self.root)}
        with mock.patch.object(web_server, "normalize_config", return_value=config), \
             mock.patch.object(supervisor, "_external_processes", return_value=[]), \
             mock.patch.object(supervisor, "_managed_running", return_value=False), \
             mock.patch.object(supervisor, "_spawn") as spawn:
            supervisor.start_deployment(config)
            first = supervisor._diagnostics_path
            config["diagnostics_jsonl"] = ""
            supervisor.start_deployment(config)
            self.assertNotEqual(first, supervisor._diagnostics_path)
            self.assertIn("outputs/openpi_deployer", str(first))
            self.assertIn("--diagnostics-jsonl", spawn.call_args.args[1])

    def test_observation_carries_existing_metrics_without_extra_requests(self) -> None:
        import sys
        sys.path.insert(0, str(MODULE.parent / "LerobotDeployer"))
        from bridge_client import PiperBridge

        client = object.__new__(PiperBridge)
        observation = {"observation.state": [0.0] * 14}
        client._request = mock.Mock(return_value={
            "observation": observation,
            "command_loop": {"achieved_control_hz": 30.0},
        })
        self.assertIs(client.observation(), observation)
        self.assertEqual(client.command_loop_stats["achieved_control_hz"], 30)
        client._request.assert_called_once_with("observe")
        # Old bridges remain readable without fabricating a control frequency.
        client._request.return_value = {"observation": observation}
        self.assertIs(client.observation(), observation)
        self.assertEqual(client.command_loop_stats, {})


if __name__ == "__main__":
    unittest.main()
