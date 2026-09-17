"""Piper-specific OpenPI configuration kept outside the vendor repository."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import einops
import flax.nnx as nnx
import numpy as np

from openpi import transforms
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.shared import normalize
from openpi.training import config as training_config
from openpi.training import weight_loaders


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from easy_embodied_deploy.action_contract import POSE_MODES

CAMERAS = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
IMAGE_KEYS = tuple(f"observation.images.{name}" for name in CAMERAS)
AXES = tuple(
    f"{side}_{axis}"
    for side in ("left", "right")
    for axis in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
)
ACTION_HORIZON = 50
MODEL_ACTION_DIM = 32


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Map the local 14-D Piper contract into OpenPI's model schema."""

    expected_cameras: ClassVar[tuple[str, ...]] = CAMERAS

    @staticmethod
    def _image(value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.shape == (3, 224, 224):
            image = einops.rearrange(image, "c h w -> h w c")
        if image.shape != (224, 224, 3):
            raise ValueError(f"Piper image must be HWC/CHW 224x224 RGB, got {image.shape}")
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != 14:
            raise ValueError(f"Piper state must have 14 dimensions, got {state.shape}")
        incoming = data["images"]
        if set(incoming) != set(self.expected_cameras):
            raise ValueError(f"Piper images must be exactly {self.expected_cameras}, got {tuple(incoming)}")
        image_names = {
            "base_0_rgb": "agentview",
            "left_wrist_0_rgb": "left_eye_in_hand",
            "right_wrist_0_rgb": "right_eye_in_hand",
        }
        result = {
            "image": {dest: self._image(incoming[source]) for dest, source in image_names.items()},
            "image_mask": {dest: np.True_ for dest in image_names},
            "state": state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 14:
                raise ValueError(f"Piper actions must have 14 dimensions, got {actions.shape}")
            result["actions"] = actions
        if "action_valid" in data:
            result["action_valid"] = np.asarray(data["action_valid"], dtype=bool)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {**data, "actions": actions[..., :14]}


@dataclasses.dataclass(frozen=True)
class PiperDataConfig(training_config.DataConfigFactory):
    default_prompt: str | None = None
    normalization_mode: str = "quantiles"
    pose_actions: bool = False

    def create(self, assets_dirs: Path, model_config) -> training_config.DataConfig:
        if self.normalization_mode not in {"quantiles", "min_max", "mean_std"}:
            raise ValueError(f"unsupported normalization mode: {self.normalization_mode}")
        base = self.create_base_config(assets_dirs, model_config)
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "agentview": "observation.images.agentview",
                            "left_eye_in_hand": "observation.images.left_eye_in_hand",
                            "right_eye_in_hand": "observation.images.right_eye_in_hand",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                        **({"action_valid": "action_valid"} if self.pose_actions else {}),
                    }
                )
            ]
        )
        return dataclasses.replace(
            base,
            repack_transforms=repack,
            data_transforms=transforms.Group(inputs=[PiperInputs()], outputs=[PiperOutputs()]),
            model_transforms=training_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
            use_quantile_norm=self.normalization_mode in {"quantiles", "min_max"},
            action_sequence_keys=("action",),
            prompt_from_task=False,
        )


@dataclasses.dataclass(frozen=True)
class PiperTrainConfig(training_config.TrainConfig):
    output_dir: str = dataclasses.field(default="", repr=False)
    save_checkpoint: bool = True

    @property
    def checkpoint_dir(self) -> Path:
        if not self.output_dir:
            raise ValueError("output_dir must be set")
        return Path(self.output_dir).expanduser().resolve()


def asset_id(repo_id: str, normalization_mode: str) -> str:
    """Return a checkpoint-portable norm-stat identifier.

    OpenPI stores normalization assets under this identifier inside every
    checkpoint.  An absolute dataset path would make the same checkpoint use a
    different identifier after it is copied to another machine.
    """
    digest = hashlib.sha256(f"{repo_id}\0{normalization_mode}".encode()).hexdigest()[:12]
    safe = repo_id.replace("/", "__").replace(" ", "_")
    return f"{safe}-{digest}"


def write_norm_stats(dataset_root: str | Path, output_parent: str | Path, repo_id: str, mode: str) -> tuple[Path, str]:
    root = Path(dataset_root).resolve()
    source = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    from easy_embodied_deploy.pose_dataset import training_stats
    source = training_stats(root, source)
    selected: dict[str, normalize.NormStats] = {}
    for target, source_key in (("state", "observation.state"), ("actions", "action")):
        raw = source[source_key]
        mean = np.asarray(raw["mean"], dtype=np.float32)
        std = np.asarray(raw["std"], dtype=np.float32)
        if mode == "quantiles":
            q01 = np.asarray(raw["q01"], dtype=np.float32)
            q99 = np.asarray(raw["q99"], dtype=np.float32)
        elif mode == "min_max":
            q01 = np.asarray(raw["min"], dtype=np.float32)
            q99 = np.asarray(raw["max"], dtype=np.float32)
        else:
            q01 = np.asarray(raw["q01"], dtype=np.float32)
            q99 = np.asarray(raw["q99"], dtype=np.float32)
        selected[target] = normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)
    identifier = asset_id(repo_id, mode)
    # Separate concurrent runs whose datasets share a repo_id. Checkpoints
    # still use the portable identifier; only the source cache is content addressed.
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:20]
    parent = Path(output_parent).resolve() / ".openpi_assets" / digest
    normalize.save(parent / identifier, selected)
    return parent, identifier


def _freeze_filter(model: pi0_config.Pi0Config, finetune_mode: str, training_scope: str):
    if finetune_mode == "lora":
        trainable = nnx.Any(
            nnx_utils.PathRegex(".*lora.*"),
            nnx_utils.PathRegex(".*action_(in|out)_proj.*"),
            nnx_utils.PathRegex(".*time_mlp.*"),
        )
        return nnx.Not(trainable)
    if training_scope == "full_model":
        return nnx.Nothing
    if training_scope == "vision_frozen":
        return nnx_utils.PathRegex(".*PaliGemma/img.*")
    if training_scope == "expert_only":
        trainable = nnx.Any(
            nnx_utils.PathRegex(".*PaliGemma/llm.*_1.*"),
            nnx_utils.PathRegex(".*action_(in|out)_proj.*"),
            nnx_utils.PathRegex(".*time_mlp.*"),
        )
        return nnx.Not(trainable)
    raise ValueError(f"unsupported training scope: {training_scope}")


def _model_config(settings: dict[str, Any]) -> pi0_config.Pi0Config:
    finetune_mode = str(settings["finetune_mode"])
    is_lora = finetune_mode == "lora"
    config_class = pi0_config.Pi0Config
    if settings["action_mode"] in (*POSE_MODES, "delta_joint"):
        from easy_embodied_deploy.pose_training_jax import PosePi0Config
        config_class = PosePi0Config
    return config_class(
        dtype=str(settings["dtype"]),
        paligemma_variant="gemma_2b_lora" if is_lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if is_lora else "gemma_300m",
        action_dim=MODEL_ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        pi05=True,
        train_time_rtc_max_delay=int(settings["rtc_max_delay"]),
        lora_rank=int(settings["lora_rank"]) if is_lora else None,
        lora_alpha=float(settings["lora_alpha"]) if is_lora else None,
        gradient_checkpointing=bool(settings["gradient_checkpointing"]),
    )


def build_train_config(settings: dict[str, Any], *, resume: bool = False) -> PiperTrainConfig:
    finetune_mode = str(settings["finetune_mode"])
    is_lora = finetune_mode == "lora"
    model = _model_config(settings)
    output_dir = Path(settings["output_dir"]).resolve()
    assets_parent, identifier = write_norm_stats(
        settings["dataset_root"], output_dir.parent, settings["repo_id"], settings["normalization_mode"]
    )
    pretrained = str(settings["pretrained"])
    params_path = str(Path(pretrained) / "params") if not pretrained.startswith(("gs://", "hf://")) and (Path(pretrained) / "params").is_dir() else pretrained
    data = PiperDataConfig(
        pose_actions=settings["action_mode"] in (*POSE_MODES, "delta_joint"),
        repo_id=str(settings["repo_id"]),
        assets=training_config.AssetsConfig(assets_dir=str(assets_parent), asset_id=identifier),
        default_prompt=str(settings.get("task", "") or "") or None,
        normalization_mode=str(settings["normalization_mode"]),
    )
    return PiperTrainConfig(
        name="piper_pi05",
        exp_name=str(settings["job_name"]),
        project_name=str(settings["wandb_project"] or "openpi-piper"),
        model=model,
        weight_loader=weight_loaders.NoOpWeightLoader() if resume else weight_loaders.CheckpointWeightLoader(params_path),
        freeze_filter=_freeze_filter(model, finetune_mode, str(settings["training_scope"])),
        data=data,
        seed=int(settings["seed"]),
        batch_size=int(settings["batch_size"]),
        num_workers=int(settings["num_workers"]),
        num_train_steps=int(settings["steps"]),
        log_interval=int(settings["log_freq"]),
        save_interval=int(settings["save_freq"]),
        keep_period=int(settings["save_freq"]),
        overwrite=False,
        resume=resume,
        wandb_enabled=bool(settings["wandb_enabled"]),
        policy_metadata={
            "trainer": "OpenpiTrainer",
            "robot_type": "piper_dual",
            "axes": AXES,
            "cameras": CAMERAS,
            "action_mode": settings["action_mode"],
            "normalization_mode": settings["normalization_mode"],
            "rtc_training_max_delay": int(settings["rtc_max_delay"]),
            "n_action_steps": int(settings["n_action_steps"]),
        },
        fsdp_devices=int(settings.get("fsdp_devices", 1)),
        ema_decay=None if is_lora else 0.99,
        output_dir=str(output_dir),
        save_checkpoint=bool(settings["save_checkpoint"]),
    )


def build_inference_config(settings: dict[str, Any], checkpoint_dir: str | Path) -> PiperTrainConfig:
    """Recreate only the model and transform contract needed for inference."""
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    model = _model_config(settings)
    identifier = asset_id(str(settings["repo_id"]), str(settings["normalization_mode"]))
    data = PiperDataConfig(
        pose_actions=settings["action_mode"] in (*POSE_MODES, "delta_joint"),
        repo_id=str(settings["repo_id"]),
        assets=training_config.AssetsConfig(
            assets_dir=str(checkpoint / "assets"),
            asset_id=identifier,
        ),
        default_prompt=None,
        normalization_mode=str(settings["normalization_mode"]),
    )
    return PiperTrainConfig(
        name="piper_pi05",
        exp_name=str(settings.get("job_name", "piper_pi05")),
        model=model,
        weight_loader=weight_loaders.NoOpWeightLoader(),
        freeze_filter=_freeze_filter(
            model,
            str(settings["finetune_mode"]),
            str(settings["training_scope"]),
        ),
        data=data,
        policy_metadata={
            "trainer": "OpenpiTrainer",
            "robot_type": "piper_dual",
            "axes": AXES,
            "cameras": CAMERAS,
            "action_mode": settings["action_mode"],
            "normalization_mode": settings["normalization_mode"],
            "rtc_training_max_delay": int(settings["rtc_max_delay"]),
            "n_action_steps": int(settings["n_action_steps"]),
        },
        output_dir=str(checkpoint.parent),
    )
