#!/usr/bin/env python3
"""JAX training entrypoint for the OpenPI Piper adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
OPENPI_ROOT = PROJECT_ROOT / "thirdParty/openpi"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from easy_embodied_deploy.action_contract import ACTION_MODES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root")
    parser.add_argument("--repo-id")
    parser.add_argument("--pretrained")
    parser.add_argument("--output-dir")
    parser.add_argument("--job-name")
    parser.add_argument("--action-mode", choices=ACTION_MODES)
    parser.add_argument("--normalization-mode", choices=("quantiles", "min_max", "mean_std"))
    parser.add_argument("--finetune-mode", choices=("lora", "full"))
    parser.add_argument("--training-scope", choices=("expert_only", "vision_frozen", "full_model"))
    parser.add_argument("--rtc-max-delay", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--log-freq", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"))
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="openpi-piper")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--resume-manifest")
    return parser


def _settings(namespace: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[dict[str, Any], bool]:
    if namespace.resume_manifest:
        path = Path(namespace.resume_manifest).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        settings = dict(payload.get("config", payload))
        settings["fsdp_devices"] = int(os.environ.get("OPENPI_FSDP_DEVICES", "1"))
        return settings, True
    required = (
        "dataset_root", "repo_id", "pretrained", "output_dir", "job_name", "action_mode",
        "normalization_mode", "finetune_mode", "training_scope", "rtc_max_delay", "lora_rank",
        "lora_alpha", "batch_size", "steps", "save_freq", "log_freq", "num_workers", "seed",
        "n_action_steps", "dtype",
    )
    missing = [name for name in required if getattr(namespace, name) is None]
    if missing:
        parser.error(f"missing required settings: {', '.join(missing)}")
    return vars(namespace), False


def _vendor_train_module():
    path = OPENPI_ROOT / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("openpi_vendor_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import OpenPI train script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = _parser()
    namespace = parser.parse_args()
    settings, resume = _settings(namespace, parser)
    os.environ["WANDB_MODE"] = str(settings.get("wandb_mode", "online"))
    if settings.get("wandb_entity"):
        os.environ["WANDB_ENTITY"] = str(settings["wandb_entity"])

    from dataset_v3 import install_openpi_loader
    from openpi_config import build_train_config

    install_openpi_loader(settings["dataset_root"])
    config = build_train_config(settings, resume=resume)
    print(
        "OPENPI_JAX_CONFIG="
        + json.dumps(
            {
                "output_dir": str(config.checkpoint_dir),
                "action_horizon": config.model.action_horizon,
                "action_dim": config.model.action_dim,
                "train_time_rtc_max_delay": config.model.train_time_rtc_max_delay,
                "trainable_filter": str(config.trainable_filter),
                "devices": config.fsdp_devices,
                "resume": resume,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    from functools import partial
    from easy_embodied_deploy.checkpoint_runtime import initialize_for_resume
    vendor = _vendor_train_module()
    initialize = vendor._checkpoints.initialize_checkpoint_dir
    vendor._checkpoints.initialize_checkpoint_dir = partial(initialize_for_resume, initialize)
    try:
        vendor.main(config)
    finally:
        vendor._checkpoints.initialize_checkpoint_dir = initialize
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
