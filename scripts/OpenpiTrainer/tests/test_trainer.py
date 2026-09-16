from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))

import trainer  # noqa: E402


def make_dataset(root: Path, action_mode: str = "delta_joint") -> Path:
    dataset = root / "sample" / "lerobot"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    vector = {"dtype": "float32", "shape": [14], "names": {"axes": list(trainer.AXES)}}
    image = {"dtype": "video", "shape": [224, 224, 3]}
    info = {
        "fps": 10,
        "total_episodes": 2,
        "total_frames": 20,
        "features": {
            "observation.state": vector,
            "action": vector,
            **{key: image for key in trainer.IMAGE_KEYS},
        },
    }
    stats = {
        key: {
            "q01": [0.0] * 14, "q99": [1.0] * 14,
            "min": [0.0] * 14, "max": [1.0] * 14,
            "mean": [0.0] * 14, "std": [1.0] * 14,
        }
        for key in ("observation.state", "action")
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "stats.json").write_text(json.dumps(stats))
    (dataset.parent / "processing.json").write_text(json.dumps({"settings": {"action_mode": action_mode}}))
    return dataset


def make_model(root: Path, *, adapter: bool = False, chunk_size: int = 50) -> Path:
    model = root / ("adapter" if adapter else "base")
    (model / "params").mkdir(parents=True)
    (model / "training_manifest.json").write_text(json.dumps({
        "config": {"finetune_mode": "lora" if adapter else "full", "action_horizon": chunk_size}
    }))
    return model


def raw_config(dataset: Path, model: Path, output: Path, **overrides):
    values = {
        "dataset_root": str(dataset), "repo_id": "local/test", "pretrained": str(model),
        "output_dir": str(output), "job_name": "test", "action_mode": "delta_joint",
        "normalization_mode": "quantiles", "finetune_mode": "lora",
        "training_scope": "expert_only", "rtc_mode": "normal", "rtc_max_delay": 5,
        "lora_rank": 16, "lora_alpha": 32, "batch_size": 1, "steps": 2,
        "save_freq": 1, "log_freq": 1, "num_workers": 0, "seed": 1,
        "n_action_steps": 10, "dtype": "bfloat16", "gradient_checkpointing": True,
        "save_checkpoint": False, "device": "cpu", "gpu_ids": "0",
        "wandb_enabled": False, "wandb_project": "", "wandb_entity": "",
        "wandb_mode": "disabled", "resume_config": "",
    }
    values.update(overrides)
    return values


class TrainerCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = make_dataset(self.root)
        self.model = make_model(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, **overrides):
        return trainer.config_from_mapping(raw_config(self.dataset, self.model, self.root / "out", **overrides))

    def test_validates_exact_piper_contract(self):
        report = trainer.validate_dataset(self.dataset, normalization_mode="quantiles", action_mode="delta_joint")
        self.assertEqual(report["frames"], 20)
        self.assertEqual(tuple(report["axes"]), trainer.AXES)
        info_path = self.dataset / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["features"]["action"]["names"]["axes"][0] = "wrong"
        info_path.write_text(json.dumps(info))
        with self.assertRaisesRegex(trainer.TrainerConfigError, "轴顺序"):
            trainer.validate_dataset(self.dataset, normalization_mode="quantiles", action_mode="delta_joint")

    def test_rejects_infinite_stats_and_action_mode_mismatch(self):
        stats_path = self.dataset / "meta/stats.json"
        stats = json.loads(stats_path.read_text())
        stats["action"]["q99"][3] = math.inf
        stats_path.write_text(json.dumps(stats))
        with self.assertRaisesRegex(trainer.TrainerConfigError, "NaN 或无穷"):
            trainer.validate_dataset(self.dataset, normalization_mode="quantiles", action_mode="delta_joint")
        stats["action"]["q99"][3] = 1
        stats_path.write_text(json.dumps(stats))
        with self.assertRaisesRegex(trainer.TrainerConfigError, "页面选择"):
            trainer.validate_dataset(self.dataset, normalization_mode="quantiles", action_mode="absolute_joint")

    def test_lora_command_and_wandb_off_are_explicit(self):
        command = trainer.build_command(self.config())
        self.assertIn("--finetune-mode", command)
        self.assertEqual(command[command.index("--finetune-mode") + 1], "lora")
        self.assertEqual(command[command.index("--lora-rank") + 1], "16")
        self.assertEqual(command[command.index("--lora-alpha") + 1], "32")
        self.assertIn("--no-wandb-enabled", command)

    def test_running_manifest_is_available_for_intermediate_checkpoint(self):
        config = self.config()
        (config.output_dir / "1/params").mkdir(parents=True)
        report = trainer.validate_dataset(
            self.dataset,
            normalization_mode="quantiles",
            action_mode="delta_joint",
            decode_sample=False,
        )
        trainer.write_training_manifest(config, report, ["train"], status="running")
        root_manifest = json.loads((config.output_dir / "training_manifest.json").read_text())
        step_manifest = json.loads((config.output_dir / "1/training_manifest.json").read_text())
        self.assertEqual(root_manifest["status"], "running")
        self.assertNotIn("completed_at", root_manifest)
        self.assertEqual(step_manifest["config"]["action_mode"], "delta_joint")

    def test_direct_full_model_unfreezes_all_and_has_no_peft(self):
        config = self.config(finetune_mode="full", training_scope="full_model", wandb_mode="online")
        command = trainer.build_command(config)
        self.assertEqual(command[command.index("--training-scope") + 1], "full_model")
        self.assertEqual(command[command.index("--finetune-mode") + 1], "full")

    def test_adapter_metadata_is_detected(self):
        adapter = make_model(self.root / "other", adapter=True)
        config = trainer.config_from_mapping(raw_config(self.dataset, adapter, self.root / "out"))
        self.assertTrue(trainer.local_model_metadata(config.pretrained)["is_adapter"])

    def test_model_chunk_size_controls_action_and_rtc_limits(self):
        small = make_model(self.root / "small", chunk_size=12)
        with self.assertRaisesRegex(trainer.TrainerConfigError, "chunk size 12"):
            trainer.config_from_mapping(raw_config(self.dataset, small, self.root / "out", n_action_steps=13))
        with self.assertRaisesRegex(trainer.TrainerConfigError, "chunk size 12"):
            trainer.config_from_mapping(raw_config(self.dataset, small, self.root / "out", rtc_mode="trained", rtc_max_delay=12))

    def test_trained_rtc_delay_reaches_native_pi05_config(self):
        config = self.config(rtc_mode="trained", rtc_max_delay=7)
        command = trainer.build_command(config)
        self.assertEqual(command[command.index("--rtc-max-delay") + 1], "7")

    def test_resume_does_not_override_saved_training_configuration(self):
        resume = self.root / "train_config.json"; resume.write_text("{}")
        config = self.config(resume_config=str(resume))
        command = trainer.build_command(config)
        self.assertEqual(command[-2:], ["--resume-manifest", str(resume)])

    def test_resume_validation_uses_checkpoint_dataset_and_normalization(self):
        absolute = make_dataset(self.root / "resume_data", action_mode="absolute")
        resume = self.root / "checkpoint" / "pretrained_model" / "train_config.json"
        resume.parent.mkdir(parents=True)
        resume.write_text(json.dumps({"config": {
            "dataset_root": str(absolute),
            "normalization_mode": "mean_std",
            "action_mode": "absolute_joint",
        }}))
        config = self.config(resume_config=str(resume), action_mode="delta_joint")
        root, normalization, action = trainer.validation_contract(config)
        self.assertEqual(root, absolute)
        self.assertEqual(normalization, "mean_std")
        self.assertEqual(action, "absolute_joint")

    @mock.patch.object(trainer, "available_gpu_indices", return_value=(0, 1, 2))
    def test_multi_gpu_uses_jax_fsdp_and_exact_visibility(self, _available):
        config = trainer.config_from_mapping(raw_config(self.dataset, self.model, self.root / "out", device="cuda", gpu_ids="0,2"))
        command = trainer.build_command(config)
        self.assertEqual(command[command.index("--fsdp-devices") + 1], "2")
        self.assertEqual(trainer.build_environment(config)["CUDA_VISIBLE_DEVICES"], "0,2")

    @mock.patch.object(trainer, "available_gpu_indices", return_value=(0,))
    def test_auto_gpu_selects_gpu_zero_on_single_gpu_host(self, _available):
        config = trainer.config_from_mapping(
            raw_config(self.dataset, self.model, self.root / "out", device="cuda", gpu_ids="auto")
        )
        self.assertEqual(config.gpu_ids, (0,))
        self.assertEqual(trainer.build_environment(config)["CUDA_VISIBLE_DEVICES"], "0")


if __name__ == "__main__":
    unittest.main()
