#!/usr/bin/env python3
"""Validate a local LeRobot dataset and launch the original lerobot-train command."""
from __future__ import annotations
import argparse
import json
import subprocess
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# No dataset is selected implicitly.  The caller chooses the exact dataset to train.
DEFAULT_DATASET_ROOT = None
DEFAULT_DATASET_REPO_ID = "local/piper_dataset"
DEFAULT_PRETRAINED = Path("models/base_models/pi05_base")
DEFAULT_OUTPUT_DIR = Path("models/pi05_piper_10hz_delta")
DEFAULT_JOB_NAME = "pi05_piper"
DEFAULT_BATCH_SIZE = 8
DEFAULT_STEPS = 30_000
DEFAULT_SAVE_FREQ = 5_000
DEFAULT_NUM_WORKERS = 4
NORMALIZATION_MODES = {
    "quantiles": "QUANTILES",
    "min_max": "MIN_MAX",
    "mean_std": "MEAN_STD",
}

def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()

def cli_path(path: Path) -> str:
    """Use a project-relative path in commands whenever possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

def validate_dataset(root: Path) -> dict:
    # Accept either .../<dataset>/lerobot or .../<dataset>.
    if not (root / "meta/info.json").is_file() and (root / "lerobot").is_dir():
        root = root / "lerobot"
    info_path, stats_path = root / "meta/info.json", root / "meta/stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        raise ValueError(f"不是完整的 LeRobot v3 数据集：{root}")
    info, stats = json.loads(info_path.read_text()), json.loads(stats_path.read_text())
    required = {"observation.state", "action", "observation.images.agentview",
                "observation.images.left_eye_in_hand", "observation.images.right_eye_in_hand"}
    missing = required.difference(info.get("features", {}))
    if missing:
        raise ValueError(f"数据集缺少 PI05 输入：{', '.join(sorted(missing))}")
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if feature.get("shape") != [14] or feature.get("dtype") != "float32":
            raise ValueError(f"{key} 必须是 float32[14]，实际为 {feature.get('dtype')} {feature.get('shape')}")
        absent = {"q01", "q99", "mean", "std", "min", "max"}.difference(stats.get(key, {}))
        if absent:
            raise ValueError(f"{key} 缺少统计量：{', '.join(sorted(absent))}")
        for stat_name in ("q01", "q99", "mean", "std", "min", "max"):
            values = stats[key][stat_name]
            if len(values) != 14 or not all(isinstance(x, (int, float)) and x == x for x in values):
                raise ValueError(f"{key}/{stat_name} 不是有限的 14 维数值统计量")
    if int(info.get("total_episodes", 0)) < 1 or int(info.get("total_frames", 0)) < 1:
        raise ValueError("数据集没有可训练帧")
    for key in required - {"observation.state", "action"}:
        if info["features"][key].get("dtype") != "video":
            raise ValueError(f"{key} 必须是 video 特征")
    # Exercise the same dataset reader used by lerobot-train before launching a
    # long run. This checks Parquet schema, episode indexing, and video decoding.
    from lerobot.datasets import LeRobotDataset
    dataset = LeRobotDataset("local/audit", root=root, download_videos=False)
    sample = dataset[0]
    for key in ("observation.state", "action"):
        if tuple(sample[key].shape) != (14,) or not sample[key].isfinite().all():
            raise ValueError(f"读取样本的 {key} 形状或数值异常")
    for key in sorted(required - {"observation.state", "action"}):
        if tuple(sample[key].shape) != (3, 224, 224) or not sample[key].isfinite().all():
            raise ValueError(f"读取样本的 {key} 不是有限的 (3,224,224) 图像")
    dataset.finalize()
    return {"root": str(root), "fps": info["fps"], "episodes": info["total_episodes"],
            "frames": info["total_frames"], "features": sorted(required),
            "sample_checked": True}

def build_command(args: argparse.Namespace) -> list[str]:
    mode = NORMALIZATION_MODES[args.normalization_mode]
    mapping = json.dumps({"ACTION": mode, "STATE": mode, "VISUAL": "IDENTITY"}, separators=(",", ":"))
    cmd = [args.lerobot_train, f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={cli_path(args.dataset_root)}", "--policy.type=pi05",
        f"--policy.pretrained_path={cli_path(Path(args.pretrained)) if not str(args.pretrained).startswith(('lerobot/', 'local/')) else args.pretrained}", f"--policy.normalization_mapping={mapping}",
        "--policy.n_action_steps=10", "--policy.empty_cameras=0",
        "--policy.freeze_vision_encoder=true", "--policy.train_expert_only=true",
        "--policy.gradient_checkpointing=true", "--policy.dtype=bfloat16",
        f"--policy.device={args.device}",
        f"--output_dir={cli_path(args.output_dir)}", f"--job_name={args.job_name}",
        f"--batch_size={args.batch_size}", f"--steps={args.steps}",
        f"--save_freq={args.save_freq}", f"--num_workers={args.num_workers}",
        "--policy.push_to_hub=false", f"--wandb.enable={'true' if args.wandb else 'false'}",
        f"--wandb.project={args.wandb_project}", f"--wandb.mode={args.wandb_mode}",
        "--wandb.add_tags=false"]
    if args.wandb_entity:
        cmd.append(f"--wandb.entity={args.wandb_entity}")
    if args.finetune_mode == "lora":
        # Top-level peft config makes lerobot-train wrap the base policy; do not
        # set policy.use_peft here, as that flag means loading an existing adapter.
        cmd += ["--peft.method_type=LORA", f"--peft.r={args.lora_rank}", "--policy.use_peft=false"]
    if args.rtc_mode == "trained":
        cmd += [f"--policy.rtc_training_max_delay={args.rtc_max_delay}"]
    if args.gpu_ids:
        args.num_gpus = len([x for x in args.gpu_ids.split(',') if x.strip()])
    if args.num_gpus > 1:
        cmd = [str(PROJECT_ROOT / "thirdParty/.venvs/lerobot-pi05/bin/accelerate"), "launch",
               "--multi_gpu", f"--num_processes={args.num_gpus}"] + cmd
    return cmd

def write_adapter_record(args: argparse.Namespace, report: dict) -> None:
    """Persist the exact normalization contract for PiperDeployer.

    LeRobot also stores this in the checkpoint preprocessor/postprocessor; this
    small sidecar makes the contract discoverable before/without loading torch.
    """
    stats = json.loads((args.dataset_root / "meta/stats.json").read_text())
    mode = NORMALIZATION_MODES[args.normalization_mode]
    record = {
        "schema_version": 1,
        "normalization_mode": args.normalization_mode,
        "normalization_mapping": {"ACTION": mode, "STATE": mode, "VISUAL": "IDENTITY"},
        "finetune_mode": args.finetune_mode,
        "rtc_mode": args.rtc_mode,
        "stats_source": "dataset/meta/stats.json",
        "stats": {k: stats[k] for k in ("action", "observation.state")},
        "dataset": report,
        "note": "Use LeRobot NormalizeProcessor/UnnormalizeProcessor; do not normalize twice.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_adapter.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        required=DEFAULT_DATASET_ROOT is None,
                        help="要训练的具体 LeRobot 数据集目录（必填；可传其父目录）")
    parser.add_argument("--repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1, help="多卡进程数；1 为单卡")
    parser.add_argument("--gpu-ids", default=None, help="显式 GPU 编号，如 0,1 或 0,2；优先于 num-gpus")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="piper-pi05")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--normalization-mode", choices=sorted(NORMALIZATION_MODES), default="quantiles",
                        help="LeRobot 原生归一化：quantiles(q01/q99)、min_max(min/max) 或 mean_std(mean/std)")
    parser.add_argument("--finetune-mode", choices=["full", "lora"], default="full")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--rtc-mode", choices=["normal", "trained"], default="normal",
                        help="PI05 训练时是否启用 trained RTC；部署 RTC guided 不属于训练选项")
    parser.add_argument("--rtc-max-delay", type=int, default=5)
    parser.add_argument("--lerobot-train", default="lerobot-train")
    parser.add_argument("--resume-config", type=Path, help="checkpoint train_config.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rtc_mode == "trained" and args.rtc_max_delay <= 0:
        parser.error("--rtc-max-delay 必须大于 0")
    args.dataset_root, args.output_dir = resolve(args.dataset_root), resolve(args.output_dir)
    if not (args.dataset_root / "meta/info.json").is_file() and (args.dataset_root / "lerobot").is_dir():
        args.dataset_root = args.dataset_root / "lerobot"
    args.pretrained = str(resolve(args.pretrained)) if not str(args.pretrained).startswith(("lerobot/", "local/")) else args.pretrained
    if args.lerobot_train == "lerobot-train":
        candidate = PROJECT_ROOT / "thirdParty/.venvs/lerobot-pi05/bin/lerobot-train"
        if candidate.is_file():
            args.lerobot_train = str(candidate)
    if not str(args.pretrained).startswith(("lerobot/", "local/")) and not Path(args.pretrained).is_dir():
        raise ValueError(f"预训练模型目录不存在：{args.pretrained}")
    if args.repo_id == DEFAULT_DATASET_REPO_ID:
        args.repo_id = "local/" + args.dataset_root.parent.name.replace(" ", "_")
    report = validate_dataset(args.dataset_root)
    command = build_command(args)
    if args.resume_config:
        command.extend(["--resume=true", f"--config_path={resolve(args.resume_config)}"])
    print(json.dumps({"dataset": report, "checkpoint_dir": str(args.output_dir / "checkpoints"),
                      "command": command}, ensure_ascii=False, indent=2))
    if args.dry_run:
        write_adapter_record(args, report)
        return 0
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if args.gpu_ids:
        run_env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids.replace(" ", "")
    result = subprocess.call(command, cwd=PROJECT_ROOT, env=run_env)
    if result == 0:
        write_adapter_record(args, report)
    return result

if __name__ == "__main__":
    raise SystemExit(main())
