import importlib.util
import io
import json
import sys
import threading
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from easy_embodied_deploy.http import read_json, json_safe
from easy_embodied_deploy.resources import hardware_lease


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DEPLOYERS = [
    load(f"review_{name}", ROOT / "scripts" / name / "web_server.py")
    for name in ("LerobotDeployer", "OpenpiDeployer")
]
TRAINERS = []
for name in ("LerobotTrainer", "OpenpiTrainer"):
    load("trainer", ROOT / "scripts" / name / "trainer.py")
    TRAINERS.append(load(f"review_{name}", ROOT / "scripts" / name / "web_server.py"))


class HttpBoundaryTests(unittest.TestCase):
    def test_rejects_negative_and_oversized_bodies_without_reading(self):
        for length in ("-1", "1000001", "invalid"):
            headers = Message()
            headers["Content-Length"] = length
            stream = mock.Mock()
            with self.assertRaises(ValueError):
                read_json(SimpleNamespace(headers=headers, rfile=stream))
            stream.read.assert_not_called()

    def test_rejects_cross_origin_and_duplicate_lengths(self):
        for duplicate in (False, True):
            headers = Message()
            headers["Content-Length"] = "2"
            headers["Host"] = "127.0.0.1:8088"
            if duplicate:
                headers["Content-Length"] = "2"
            else:
                headers["Origin"] = "https://outside.example"
            with self.assertRaises(ValueError):
                read_json(SimpleNamespace(headers=headers, rfile=io.BytesIO(b"{}")))

    def test_nonfinite_metrics_remain_valid_json(self):
        self.assertEqual(
            json.loads(json.dumps(json_safe({"loss": float("nan")}), allow_nan=False)),
            {"loss": None},
        )

    def test_hardware_lease_rejects_second_owner(self):
        with hardware_lease("unit-test"):
            with self.assertRaisesRegex(RuntimeError, "占用"):
                with hardware_lease("unit-test"):
                    pass
        with hardware_lease("unit-test"):
            pass


class DeploymentLifecycleTests(unittest.TestCase):
    def test_stop_cancels_waiting_start_without_late_model_launch(self):
        for module in DEPLOYERS:
            with self.subTest(module=module.__name__):
                supervisor = module.ProcessSupervisor()
                bridge = SimpleNamespace(
                    running=True,
                    state="starting",
                    ready=threading.Event(),
                    command=["tcp://127.0.0.1:6007"],
                )
                entered = threading.Event()

                def start_bridge(endpoint):
                    supervisor._processes["bridge"] = bridge
                    entered.set()
                    return bridge

                with (
                    mock.patch.object(
                        module,
                        "normalize_config",
                        return_value={"bridge_endpoint": "tcp://127.0.0.1:6007"},
                    ),
                    mock.patch.object(
                        supervisor, "_external_processes", return_value=[]
                    ),
                    mock.patch.object(
                        supervisor, "start_bridge", side_effect=start_bridge
                    ),
                    mock.patch.object(supervisor, "_signal_process") as stop,
                    mock.patch.object(supervisor, "start_deployment") as deploy,
                ):
                    supervisor.start_stack({})
                    self.assertTrue(entered.wait(2))
                    supervisor.stop_all(asynchronous=False)
                    bridge.ready.set()
                    supervisor._orchestrator.join(2)
                    deploy.assert_not_called()
                    self.assertEqual(supervisor._operation, "idle")
                    stop.assert_any_call("bridge", timeout=90)

    def test_stop_blocks_new_process_launch(self):
        for module in DEPLOYERS:
            supervisor = module.ProcessSupervisor()
            supervisor._operation = "stopping"
            with mock.patch.object(module.subprocess, "Popen") as popen:
                with self.assertRaises(module.ConsoleError):
                    supervisor._spawn("bridge", [], {})
                popen.assert_not_called()

    def test_external_bridge_is_not_taken_over(self):
        for module in DEPLOYERS:
            supervisor = module.ProcessSupervisor()
            with (
                mock.patch.object(
                    module,
                    "normalize_config",
                    return_value={"bridge_endpoint": "tcp://127.0.0.1:6007"},
                ),
                mock.patch.object(supervisor, "_external_processes", return_value=[]),
                mock.patch.object(module, "port_is_open", return_value=True),
            ):
                with self.assertRaisesRegex(module.ConsoleError, "外部桥"):
                    supervisor.start_deployment({})

    def test_failed_stack_homes_the_bridge_it_created(self):
        for module in DEPLOYERS:
            supervisor = module.ProcessSupervisor()
            bridge = SimpleNamespace(
                running=True, state="ready", command=["tcp://127.0.0.1:6007"]
            )

            def create(endpoint):
                supervisor._processes["bridge"] = bridge
                return bridge

            with (
                mock.patch.object(
                    module,
                    "normalize_config",
                    return_value={"bridge_endpoint": "tcp://127.0.0.1:6007"},
                ),
                mock.patch.object(supervisor, "_external_processes", return_value=[]),
                mock.patch.object(supervisor, "start_bridge", side_effect=create),
                mock.patch.object(
                    supervisor,
                    "start_deployment",
                    side_effect=module.ConsoleError("bad model"),
                ),
                mock.patch.object(supervisor, "_signal_process") as stop,
            ):
                supervisor.start_stack({})
                supervisor._orchestrator.join(2)
                stop.assert_called_once_with("bridge", timeout=90)
                self.assertEqual(supervisor._operation, "failed")

    def test_failed_model_load_cleans_up_owned_bridge(self):
        for module in DEPLOYERS:
            supervisor = module.ProcessSupervisor()
            process = mock.Mock(stdout=io.StringIO("model load failed\n"))
            process.wait.return_value = 1
            process.poll.return_value = 1
            managed = module.ManagedProcess("deployer", process, ["model"], 0)
            supervisor._processes["deployer"] = managed
            supervisor._processes["bridge"] = SimpleNamespace(running=True)
            with mock.patch.object(supervisor, "_signal_process") as stop:
                supervisor._read_process(managed)
                supervisor._stop_thread.join(2)
                stop.assert_any_call("bridge", timeout=90)
                self.assertEqual(managed.state, "failed")

    def test_direct_bridge_start_cannot_finish_after_stop(self):
        for module in DEPLOYERS:
            supervisor = module.ProcessSupervisor()
            entered = threading.Event()
            release = threading.Event()
            errors = []

            def port_check(endpoint):
                entered.set()
                release.wait(2)
                return False

            def launch():
                try:
                    supervisor.start_bridge("tcp://127.0.0.1:6007")
                except module.ConsoleError as exc:
                    errors.append(str(exc))

            with (
                mock.patch.object(module, "port_is_open", side_effect=port_check),
                mock.patch.object(module, "bridge_command", return_value=([], {})),
                mock.patch.object(supervisor, "_signal_process"),
                mock.patch.object(supervisor, "_spawn") as spawn,
            ):
                thread = threading.Thread(target=launch)
                thread.start()
                self.assertTrue(entered.wait(2))
                supervisor.stop_all(asynchronous=False)
                release.set()
                thread.join(2)
                spawn.assert_not_called()
                self.assertIn("启动已取消", errors)

    def test_stop_waits_for_existing_home_without_interrupting_it(self):
        for module in DEPLOYERS:
            supervisor = module.ProcessSupervisor()
            process = mock.Mock()
            managed = SimpleNamespace(running=True, state="homing", process=process)
            supervisor._processes["bridge"] = managed
            with mock.patch.object(module.os, "killpg") as signal:
                supervisor._signal_process("bridge", timeout=90)
                process.wait.assert_called_once_with(timeout=90)
                signal.assert_not_called()


class TrainingLifecycleTests(unittest.TestCase):
    def test_stop_during_validation_cancels_spawn(self):
        for module in TRAINERS:
            with self.subTest(module=module.__name__):
                supervisor = module.TrainingSupervisor()
                entered = threading.Event()
                release = threading.Event()
                errors = []

                def validate(*args, **kwargs):
                    entered.set()
                    release.wait(2)
                    return {}

                def run():
                    try:
                        supervisor.start({})
                    except module.ConsoleError as exc:
                        errors.append(str(exc))

                with (
                    mock.patch.object(
                        module, "normalize_web_config", return_value=SimpleNamespace()
                    ),
                    mock.patch.object(
                        module,
                        "validation_contract",
                        return_value=(Path("/tmp"), "quantiles", "delta_joint"),
                    ),
                    mock.patch.object(module, "validate_dataset", side_effect=validate),
                    mock.patch.object(supervisor, "_assert_gpu_available"),
                    mock.patch.object(
                        module, "wrapper_command", return_value=["unused"]
                    ),
                    mock.patch.object(module, "build_environment", return_value={}),
                    mock.patch.object(module.subprocess, "Popen") as popen,
                ):
                    thread = threading.Thread(target=run)
                    thread.start()
                    self.assertTrue(entered.wait(2))
                    supervisor.stop()
                    release.set()
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
                    popen.assert_not_called()
                    self.assertIn("启动已取消", errors)

    def test_stopped_supervisor_does_not_restart_after_shutdown(self):
        for module in TRAINERS:
            supervisor = module.TrainingSupervisor()
            supervisor.shutdown()
            with self.assertRaises(module.ConsoleError):
                supervisor.start({})

    def test_openpi_discovery_uses_local_tree_for_migrated_manifest(self):
        import tempfile

        module = TRAINERS[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "models/finetuned_models/run"
            (run / "12/params").mkdir(parents=True)
            (run / "training_manifest.json").write_text(
                json.dumps(
                    {
                        "trainer": "OpenpiTrainer",
                        "config": {"output_dir": "/root/inaccessible/old-host/run"},
                    }
                )
            )
            with mock.patch.object(module, "PROJECT_ROOT", root):
                found = module.discover_resume_configs()
            self.assertEqual(found[0]["step"], 12)
            self.assertEqual(found[0]["output_dir"], str(run))


class ConfigurationBoundaryTests(unittest.TestCase):
    def test_string_false_never_enables_motion_or_dry_run(self):
        from easy_embodied_deploy.contracts import boolean

        self.assertFalse(boolean("false", "flag"))
        self.assertTrue(boolean("true", "flag"))
        for value in (None, [], 1, "invalid"):
            with self.assertRaises(ValueError):
                boolean(value, "flag")

    def test_gpu_missing_telemetry_does_not_break_status(self):
        from easy_embodied_deploy import system

        gpu = SimpleNamespace(
            stdout="0, uuid, Test GPU, 100, 200, [N/A], [Not Supported]"
        )
        processes = SimpleNamespace(stdout="uuid, 123, python, [N/A]")
        with mock.patch.object(system.subprocess, "run", side_effect=[gpu, processes]):
            result = system.gpu_status()
        self.assertIsNone(result[0]["temperature"])
        self.assertIsNone(result[0]["processes"][0]["memory_mb"])

    def test_checkpoint_manifest_mismatch_cannot_change_action_semantics(self):
        import tempfile
        from easy_embodied_deploy.contracts import checkpoint_action_mode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "checkpoints/1/pretrained_model"
            model.mkdir(parents=True)
            self.assertIsNone(checkpoint_action_mode(model, "delta_joint"))
            (root / "training_manifest.json").write_text(
                json.dumps({"config": {"action_mode": "absolute_joint"}})
            )
            with self.assertRaisesRegex(ValueError, "动作模式"):
                checkpoint_action_mode(model, "delta_joint")
            self.assertEqual(
                checkpoint_action_mode(model, "absolute_joint"), "absolute_joint"
            )

    def test_openpi_offline_launch_never_starts_hardware(self):
        module = DEPLOYERS[1]
        supervisor = module.ProcessSupervisor()
        config = dict(
            offline_dataset_root="/dataset",
            bridge_endpoint="tcp://127.0.0.1:6007",
            diagnostics_jsonl="",
        )
        with (
            mock.patch.object(module, "normalize_config", return_value=config),
            mock.patch.object(supervisor, "_external_processes", return_value=[]),
            mock.patch.object(
                module, "deployment_command", return_value=(["infer"], {})
            ),
            mock.patch.object(supervisor, "_spawn") as spawn,
            mock.patch.object(supervisor, "start_bridge") as bridge,
        ):
            supervisor.start_stack(config)
            spawn.assert_called_once_with("deployer", ["infer"], {})
            bridge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
