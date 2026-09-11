#!/usr/bin/env python3
"""Validate Piper data and launch LeRobot PI05 training without modifying LeRobot."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
VENV_PYTHON = PROJECT_ROOT / "thirdParty/.venvs/lerobot-pi05/bin/python"
LEROBOT_TRAIN = PROJECT_ROOT / "thirdParty/.venvs/lerobot-pi05/bin/lerobot-train"
DEFAULT_PRETRAINED = PROJECT_ROOT / "models/base_models/pi05_base"
DEFAULT_DATASET = PROJECT_ROOT / (
    "datasets/lerobot/"
    "Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "models/finetuned_models/pi05_trainer_run"

CAMERAS = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
IMAGE_KEYS = tuple(f"observation.images.{name}" for name in CAMERAS)
AXES = tuple(
    f"{side}_{axis}"
    for side in ("left", "right")
    for axis in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
)
NORMALIZATION_MODES = {
    "quantiles": "QUANTILES",
    "min_max": "MIN_MAX",
    "mean_std": "MEAN_STD",
}
TRAINING_SCOPES = {
    "expert_only": (True, True),
    "vision_frozen": (True, False),
    "full_model": (False, False),
}


class TrainerConfigError(ValueError):
    """Configuration error safe to show in the CLI or Web UI."""


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: Path
    repo_id: str
    pretrained: str
    output_dir: Path
    job_name: str
    action_mode: str
    normalization_mode: str
    finetune_mode: str
    training_scope: str
    rtc_mode: str
    rtc_max_delay: int
    lora_rank: int
    lora_alpha: int
    batch_size: int
    steps: int
    save_freq: int
    log_freq: int
    num_workers: int
    seed: int
    n_action_steps: int
    dtype: str
    gradient_checkpointing: bool
    save_checkpoint: bool
    device: str
    gpu_ids: tuple[int, ...]
    wandb_enabled: bool
    wandb_project: str
    wandb_entity: str
    wandb_mode: str
    resume_config: Path | None


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def cli_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_dataset_root(value: str | Path) -> Path:
    root = resolve_path(value)
    if not (root / "meta/info.json").is_file() and (root / "lerobot").is_dir():
        root = root / "lerobot"
    return root


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise TrainerConfigError(f"{name} 必须是布尔值")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainerConfigError(f"{name} 必须是整数") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TrainerConfigError(f"{name} 必须是整数")
    if number < minimum:
        raise TrainerConfigError(f"{name} 不能小于 {minimum}")
    return number


def parse_gpu_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
    else:
        parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(not re.fullmatch(r"\d+", part) for part in parts):
        raise TrainerConfigError("GPU 必须使用逗号分隔的物理编号，例如 0 或 0,1")
    result = tuple(int(part) for part in parts)
    if len(result) != len(set(result)):
        raise TrainerConfigError("GPU 编号不能重复")
    return result


def available_gpu_indices() -> tuple[int, ...]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ()
    return tuple(int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit())


def _model_source(value: Any) -> str:
    source = str(value).strip()
    if not source:
        raise TrainerConfigError("预训练模型不能为空")
    if source.startswith(("lerobot/", "hf://")):
        return source
    path = resolve_path(source)
    if not path.is_dir():
        raise TrainerConfigError(f"预训练模型目录不存在：{path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise TrainerConfigError(f"预训练模型缺少 config.json：{path}")
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainerConfigError(f"无法读取预训练模型 config.json：{exc}") from exc
    if model_config.get("type") != "pi05":
        raise TrainerConfigError(f"仅支持 PI05，检测到：{model_config.get('type', 'unknown')}")
    return str(path)


def local_model_metadata(source: str) -> dict[str, Any]:
    """Return the small amount of local checkpoint metadata needed before loading torch."""
    path = Path(source)
    if not path.is_absolute() or not path.is_dir():
        return {"is_adapter": False, "chunk_size": 50}
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    chunk_size = config.get("chunk_size", 50)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise TrainerConfigError(f"模型 chunk_size 无效：{chunk_size}")
    return {"is_adapter": (path / "adapter_config.json").is_file(), "chunk_size": chunk_size}


def _default_repo_id(dataset_root: Path) -> str:
    name = dataset_root.parent.name if dataset_root.name == "lerobot" else dataset_root.name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_.-") or "piper_dataset"
    return f"local/{safe}"


def config_from_mapping(raw: dict[str, Any], *, check_output: bool = True) -> TrainingConfig:
    dataset_root = resolve_dataset_root(raw.get("dataset_root", ""))
    pretrained = _model_source(raw.get("pretrained", ""))
    output_dir = resolve_path(raw.get("output_dir", ""))
    resume_value = str(raw.get("resume_config", "") or "").strip()
    resume_config = resolve_path(resume_value) if resume_value else None
    if resume_config is not None and not resume_config.is_file():
        raise TrainerConfigError(f"恢复配置不存在：{resume_config}")

    device = str(raw.get("device", "cuda")).strip()
    if device not in {"cuda", "cpu"}:
        raise TrainerConfigError("设备必须是 cuda 或 cpu")
    gpu_ids = parse_gpu_ids(raw.get("gpu_ids", "0")) if device == "cuda" else ()
    detected = available_gpu_indices()
    if device == "cuda":
        if not detected:
            raise TrainerConfigError("未检测到 NVIDIA GPU")
        missing = sorted(set(gpu_ids).difference(detected))
        if missing:
            raise TrainerConfigError(f"GPU 不存在：{missing}；可用编号：{list(detected)}")

    action_mode = str(raw.get("action_mode", "delta_joint"))
    normalization_mode = str(raw.get("normalization_mode", "quantiles"))
    finetune_mode = str(raw.get("finetune_mode", "lora"))
    training_scope = str(raw.get("training_scope", "expert_only"))
    rtc_mode = str(raw.get("rtc_mode", "normal"))
    dtype = str(raw.get("dtype", "bfloat16"))
    wandb_mode = str(raw.get("wandb_mode", "online"))
    if action_mode not in {"delta_joint", "absolute_joint"}:
        raise TrainerConfigError("动作表示无效")
    if normalization_mode not in NORMALIZATION_MODES:
        raise TrainerConfigError("归一化模式无效")
    if finetune_mode not in {"lora", "full"}:
        raise TrainerConfigError("微调模式无效")
    if training_scope not in TRAINING_SCOPES:
        raise TrainerConfigError("训练范围无效")
    if rtc_mode not in {"normal", "trained"}:
        raise TrainerConfigError("RTC 训练模式无效")
    if dtype not in {"bfloat16", "float32"}:
        raise TrainerConfigError("dtype 必须是 bfloat16 或 float32")
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise TrainerConfigError("W&B mode 无效")

    rtc_max_delay = _integer(raw.get("rtc_max_delay", 5), "RTC 最大延迟", minimum=0)
    if rtc_mode == "trained" and rtc_max_delay < 1:
        raise TrainerConfigError("trained RTC 的最大延迟必须大于 0")
    if rtc_mode == "normal":
        rtc_max_delay = 0

    wandb_enabled = _bool(raw.get("wandb_enabled", True), "W&B")
    if wandb_mode == "disabled":
        wandb_enabled = False

    config = TrainingConfig(
        dataset_root=dataset_root,
        repo_id=str(raw.get("repo_id", "")).strip() or _default_repo_id(dataset_root),
        pretrained=pretrained,
        output_dir=output_dir,
        job_name=str(raw.get("job_name", "pi05_piper")).strip(),
        action_mode=action_mode,
        normalization_mode=normalization_mode,
        finetune_mode=finetune_mode,
        training_scope=training_scope,
        rtc_mode=rtc_mode,
        rtc_max_delay=rtc_max_delay,
        lora_rank=_integer(raw.get("lora_rank", 16), "LoRA rank", minimum=1),
        lora_alpha=_integer(raw.get("lora_alpha", 16), "LoRA alpha", minimum=1),
        batch_size=_integer(raw.get("batch_size", 8), "Batch size", minimum=1),
        steps=_integer(raw.get("steps", 30_000), "训练步数", minimum=1),
        save_freq=_integer(raw.get("save_freq", 5_000), "保存频率", minimum=1),
        log_freq=_integer(raw.get("log_freq", 200), "日志频率", minimum=1),
        num_workers=_integer(raw.get("num_workers", 4), "Workers", minimum=0),
        seed=_integer(raw.get("seed", 1000), "随机种子", minimum=0),
        n_action_steps=_integer(raw.get("n_action_steps", 10), "Action steps", minimum=1),
        dtype=dtype,
        gradient_checkpointing=_bool(
            raw.get("gradient_checkpointing", True), "Gradient checkpointing"
        ),
        save_checkpoint=_bool(raw.get("save_checkpoint", True), "保存 checkpoint"),
        device=device,
        gpu_ids=gpu_ids,
        wandb_enabled=wandb_enabled,
        wandb_project=str(raw.get("wandb_project", "piper-pi05")).strip(),
        wandb_entity=str(raw.get("wandb_entity", "") or "").strip(),
        wandb_mode=wandb_mode,
        resume_config=resume_config,
    )
    if not config.job_name:
        raise TrainerConfigError("Job name 不能为空")
    if any(ord(char) < 32 for char in config.job_name):
        raise TrainerConfigError("Job name 不能包含控制字符")
    if not config.repo_id.startswith("local/"):
        raise TrainerConfigError("本地数据集 repo ID 必须以 local/ 开头")
    if config.wandb_enabled and not config.wandb_project:
        raise TrainerConfigError("启用 W&B 时 project 不能为空")
    model_metadata = local_model_metadata(config.pretrained)
    chunk_size = model_metadata["chunk_size"]
    if model_metadata["is_adapter"] and config.finetune_mode != "lora":
        raise TrainerConfigError("LoRA adapter 不能作为直接权重微调的输入；请选择 base/全量 checkpoint")
    if config.n_action_steps > chunk_size:
        raise TrainerConfigError(f"PI05 action steps 不能大于模型 chunk size {chunk_size}")
    if config.rtc_max_delay >= chunk_size:
        raise TrainerConfigError(f"RTC 最大延迟必须小于模型 chunk size {chunk_size}")
    if check_output and config.resume_config is None and config.output_dir.exists():
        raise TrainerConfigError(f"输出目录已存在，请更换目录：{config.output_dir}")
    return config


def _feature_axes(feature: dict[str, Any]) -> tuple[str, ...]:
    names = feature.get("names")
    if isinstance(names, dict):
        names = names.get("axes")
    if not isinstance(names, list):
        return ()
    return tuple(str(name) for name in names)


def _finite_vector(stats: dict[str, Any], key: str, stat: str) -> list[float]:
    values = stats.get(key, {}).get(stat)
    if not isinstance(values, list) or len(values) != 14:
        raise TrainerConfigError(f"{key}/{stat} 必须是 14 维统计量")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise TrainerConfigError(f"{key}/{stat} 包含非数值")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TrainerConfigError(f"{key}/{stat} 包含非数值") from exc
        if not math.isfinite(number):
            raise TrainerConfigError(f"{key}/{stat} 包含 NaN 或无穷值")
        converted.append(number)
    return converted


def validate_dataset(
    root: Path,
    *,
    normalization_mode: str,
    action_mode: str,
    decode_sample: bool = False,
) -> dict[str, Any]:
    info_path = root / "meta/info.json"
    stats_path = root / "meta/stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        raise TrainerConfigError(f"不是完整的 LeRobot 数据集：{root}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainerConfigError(f"数据集元数据无法读取：{exc}") from exc

    features = info.get("features")
    if not isinstance(features, dict):
        raise TrainerConfigError("数据集 info.json 缺少 features")
    required = {"observation.state", "action", *IMAGE_KEYS}
    missing = required.difference(features)
    if missing:
        raise TrainerConfigError(f"数据集缺少 PI05 输入：{sorted(missing)}")
    for key in ("observation.state", "action"):
        feature = features[key]
        if feature.get("shape") != [14] or feature.get("dtype") != "float32":
            raise TrainerConfigError(f"{key} 必须是 float32[14]，实际为 {feature}")
        axes = _feature_axes(feature)
        if axes != AXES:
            raise TrainerConfigError(f"{key} 轴顺序与双臂 Piper 不一致：{axes}")
    for key in IMAGE_KEYS:
        feature = features[key]
        if feature.get("shape") != [224, 224, 3]:
            raise TrainerConfigError(f"{key} 必须是 HWC [224,224,3]")
        if feature.get("dtype") not in {"video", "image"}:
            raise TrainerConfigError(f"{key} 必须是 video/image 特征")

    selected_stats = {
        "quantiles": ("q01", "q99"),
        "min_max": ("min", "max"),
        "mean_std": ("mean", "std"),
    }[normalization_mode]
    checked_stats: dict[str, dict[str, list[float]]] = {}
    for key in ("observation.state", "action"):
        checked_stats[key] = {
            stat: _finite_vector(stats, key, stat) for stat in selected_stats
        }
        lower, upper = checked_stats[key].values()
        if normalization_mode in {"quantiles", "min_max"} and any(
            low > high for low, high in zip(lower, upper, strict=True)
        ):
            raise TrainerConfigError(f"{key} 的归一化上下界顺序错误")
        if normalization_mode == "mean_std" and any(value < 0 for value in upper):
            raise TrainerConfigError(f"{key}/std 不能为负数")

    episodes = int(info.get("total_episodes", 0) or 0)
    frames = int(info.get("total_frames", 0) or 0)
    fps = float(info.get("fps", 0) or 0)
    if episodes < 1 or frames < 1 or not math.isfinite(fps) or fps <= 0:
        raise TrainerConfigError("数据集 episodes、frames 或 fps 无效")

    declared_action_mode = None
    processing_path = root.parent / "processing.json"
    if processing_path.is_file():
        try:
            processing = json.loads(processing_path.read_text(encoding="utf-8"))
            declared_action_mode = processing["settings"]["action_mode"]
            if declared_action_mode == "absolute":
                declared_action_mode = "absolute_joint"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TrainerConfigError(f"processing.json 无法读取：{exc}") from exc
        if declared_action_mode != action_mode:
            raise TrainerConfigError(
                f"数据集声明 {declared_action_mode}，页面选择的是 {action_mode}"
            )

    if decode_sample:
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset("local/trainer_validation", root=root, download_videos=False)
        try:
            sample = dataset[0]
            for key in ("observation.state", "action"):
                tensor = sample[key]
                if tuple(tensor.shape) != (14,) or not tensor.isfinite().all().item():
                    raise TrainerConfigError(f"首帧 {key} 形状或数值异常")
            for key in IMAGE_KEYS:
                tensor = sample[key]
                if tuple(tensor.shape) != (3, 224, 224) or not tensor.isfinite().all().item():
                    raise TrainerConfigError(f"首帧 {key} 不是有限的 CHW [3,224,224]")
        finally:
            finalize = getattr(dataset, "finalize", None)
            if callable(finalize):
                finalize()

    return {
        "root": str(root),
        "repo_id": _default_repo_id(root),
        "fps": fps,
        "episodes": episodes,
        "frames": frames,
        "action_mode": declared_action_mode,
        "normalization_mode": normalization_mode,
        "sample_checked": decode_sample,
        "cameras": list(CAMERAS),
        "axes": list(AXES),
    }


def validation_contract(config: TrainingConfig) -> tuple[Path, str, str]:
    """Resolve the dataset contract that the launched command will actually consume."""
    if config.resume_config is None:
        return config.dataset_root, config.normalization_mode, config.action_mode
    try:
        saved = json.loads(config.resume_config.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainerConfigError(f"恢复配置无法读取：{exc}") from exc
    try:
        saved_root = saved["dataset"]["root"]
    except (KeyError, TypeError) as exc:
        raise TrainerConfigError("恢复配置缺少 dataset.root") from exc
    root = resolve_dataset_root(saved_root)
    mapping = saved.get("policy", {}).get("normalization_mapping", {})
    action_normalization = str(mapping.get("ACTION", "")).upper()
    reverse_modes = {value: key for key, value in NORMALIZATION_MODES.items()}
    normalization_mode = reverse_modes.get(action_normalization, config.normalization_mode)

    action_mode = config.action_mode
    manifest_path = config.resume_config.parent / "training_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            action_mode = manifest.get("config", {}).get("action_mode", action_mode)
        except (OSError, ValueError, TypeError):
            pass
    processing_path = root.parent / "processing.json"
    if processing_path.is_file():
        try:
            processing = json.loads(processing_path.read_text(encoding="utf-8"))
            action_mode = processing.get("settings", {}).get("action_mode", action_mode)
        except (OSError, ValueError, TypeError) as exc:
            raise TrainerConfigError(f"恢复数据集 processing.json 无法读取：{exc}") from exc
        if action_mode == "absolute":
            action_mode = "absolute_joint"
    return root, normalization_mode, action_mode


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def build_command(config: TrainingConfig) -> list[str]:
    if config.resume_config is not None:
        command = [
            str(LEROBOT_TRAIN),
            f"--config_path={config.resume_config}",
            "--resume=true",
            f"--policy.device={config.device}",
        ]
    else:
        model_metadata = local_model_metadata(config.pretrained)
        mode = NORMALIZATION_MODES[config.normalization_mode]
        mapping = json.dumps(
            {"ACTION": mode, "STATE": mode, "VISUAL": "IDENTITY"},
            separators=(",", ":"),
        )
        freeze_vision, train_expert_only = TRAINING_SCOPES[config.training_scope]
        if config.finetune_mode == "lora":
            freeze_vision, train_expert_only = True, True
        command = [
            str(LEROBOT_TRAIN),
            f"--dataset.repo_id={config.repo_id}",
            f"--dataset.root={cli_path(config.dataset_root)}",
            "--policy.type=pi05",
            f"--policy.pretrained_path={cli_path(Path(config.pretrained)) if Path(config.pretrained).is_absolute() else config.pretrained}",
            f"--policy.normalization_mapping={mapping}",
            f"--policy.n_action_steps={config.n_action_steps}",
            "--policy.empty_cameras=0",
            f"--policy.freeze_vision_encoder={_bool_arg(freeze_vision)}",
            f"--policy.train_expert_only={_bool_arg(train_expert_only)}",
            f"--policy.gradient_checkpointing={_bool_arg(config.gradient_checkpointing)}",
            f"--policy.dtype={config.dtype}",
            f"--policy.device={config.device}",
            f"--policy.rtc_training_max_delay={config.rtc_max_delay}",
            f"--policy.use_peft={_bool_arg(model_metadata['is_adapter'])}",
            f"--output_dir={cli_path(config.output_dir)}",
            f"--job_name={config.job_name}",
            f"--batch_size={config.batch_size}",
            f"--steps={config.steps}",
            f"--save_checkpoint={_bool_arg(config.save_checkpoint)}",
            f"--save_freq={config.save_freq}",
            f"--log_freq={config.log_freq}",
            f"--num_workers={config.num_workers}",
            f"--seed={config.seed}",
            "--policy.push_to_hub=false",
            f"--wandb.enable={_bool_arg(config.wandb_enabled)}",
            "--wandb.add_tags=false",
        ]
        if config.wandb_enabled:
            command.extend(
                [
                    f"--wandb.project={config.wandb_project}",
                    f"--wandb.mode={config.wandb_mode}",
                ]
            )
            if config.wandb_entity:
                command.append(f"--wandb.entity={config.wandb_entity}")
        if config.finetune_mode == "lora":
            command.extend(
                [
                    "--peft.method_type=LORA",
                    f"--peft.r={config.lora_rank}",
                    f"--peft.lora_alpha={config.lora_alpha}",
                ]
            )

    if config.device == "cuda" and len(config.gpu_ids) > 1:
        command = [
            str(VENV_PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(config.gpu_ids)}",
            *command,
        ]
    return command


def build_environment(config: TrainingConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["CUDA_VISIBLE_DEVICES"] = (
        ",".join(str(index) for index in config.gpu_ids)
        if config.device == "cuda"
        else ""
    )
    return environment


def wrapper_command(
    config: TrainingConfig, *, dry_run: bool = False, validate_only: bool = False
) -> list[str]:
    """Build the isolated wrapper CLI used by the Web supervisor."""
    command = [
        str(VENV_PYTHON),
        str(MODULE_DIR / "trainer.py"),
        "--dataset-root",
        str(config.dataset_root),
        "--repo-id",
        config.repo_id,
        "--pretrained",
        config.pretrained,
        "--output-dir",
        str(config.output_dir),
        "--job-name",
        config.job_name,
        "--action-mode",
        config.action_mode,
        "--normalization-mode",
        config.normalization_mode,
        "--finetune-mode",
        config.finetune_mode,
        "--training-scope",
        config.training_scope,
        "--rtc-mode",
        config.rtc_mode,
        "--rtc-max-delay",
        str(config.rtc_max_delay),
        "--lora-rank",
        str(config.lora_rank),
        "--lora-alpha",
        str(config.lora_alpha),
        "--batch-size",
        str(config.batch_size),
        "--steps",
        str(config.steps),
        "--save-freq",
        str(config.save_freq),
        "--log-freq",
        str(config.log_freq),
        "--num-workers",
        str(config.num_workers),
        "--seed",
        str(config.seed),
        "--n-action-steps",
        str(config.n_action_steps),
        "--dtype",
        config.dtype,
        "--gradient-checkpointing" if config.gradient_checkpointing else "--no-gradient-checkpointing",
        "--save-checkpoint" if config.save_checkpoint else "--no-save-checkpoint",
        "--device",
        config.device,
        "--gpu-ids",
        ",".join(str(index) for index in config.gpu_ids) or "0",
        "--wandb-enabled" if config.wandb_enabled else "--no-wandb-enabled",
        "--wandb-project",
        config.wandb_project,
        "--wandb-entity",
        config.wandb_entity,
        "--wandb-mode",
        config.wandb_mode,
    ]
    if config.resume_config is not None:
        command.extend(["--resume-config", str(config.resume_config)])
    if validate_only:
        command.append("--validate-only")
    if dry_run:
        command.append("--dry-run")
    return command


def config_for_json(config: TrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["dataset_root"] = str(config.dataset_root)
    payload["output_dir"] = str(config.output_dir)
    payload["resume_config"] = None if config.resume_config is None else str(config.resume_config)
    payload["gpu_ids"] = list(config.gpu_ids)
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_training_manifest(
    config: TrainingConfig, report: dict[str, Any], command: list[str]
) -> None:
    if config.resume_config is not None:
        return
    manifest = {
        "schema_version": 2,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "trainer": "LerobotTrainer",
        "config": config_for_json(config),
        "dataset": report,
        "command": command,
        "normalization_source": "checkpoint policy processor state",
    }
    if config.output_dir.is_dir():
        _write_json_atomic(config.output_dir / "training_manifest.json", manifest)
        for pretrained_dir in config.output_dir.glob("checkpoints/*/pretrained_model"):
            if pretrained_dir.is_dir():
                _write_json_atomic(pretrained_dir / "training_manifest.json", manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--job-name", default="pi05_piper")
    parser.add_argument("--action-mode", choices=("delta_joint", "absolute_joint"), default="delta_joint")
    parser.add_argument("--normalization-mode", choices=tuple(NORMALIZATION_MODES), default="quantiles")
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--training-scope", choices=tuple(TRAINING_SCOPES), default="expert_only")
    parser.add_argument("--rtc-mode", choices=("normal", "trained"), default="normal")
    parser.add_argument("--rtc-max-delay", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="piper-pi05")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--resume-config", default="")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    namespace = parser.parse_args()
    raw = vars(namespace)
    try:
        config = config_from_mapping(raw, check_output=not namespace.validate_only)
        dataset_root, normalization_mode, action_mode = validation_contract(config)
        report = validate_dataset(
            dataset_root,
            normalization_mode=normalization_mode,
            action_mode=action_mode,
            decode_sample=True,
        )
    except TrainerConfigError as exc:
        parser.error(str(exc))
    if namespace.validate_only:
        print("LEROBOT_TRAINER_VALIDATION=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0

    command = build_command(config)
    plan = {
        "config": config_for_json(config),
        "dataset": report,
        "command": command,
        "cuda_visible_devices": build_environment(config)["CUDA_VISIBLE_DEVICES"],
    }
    print("LEROBOT_TRAINER_PLAN=" + json.dumps(plan, ensure_ascii=False), flush=True)
    if namespace.dry_run:
        return 0

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=build_environment(config),
            check=False,
        )
        return_code = result.returncode
    except KeyboardInterrupt:
        return_code = 130
    if return_code == 0:
        write_training_manifest(config, report, command)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
