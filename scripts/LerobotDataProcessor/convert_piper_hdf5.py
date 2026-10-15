#!/usr/bin/env python3
"""DataProcessor CLI; defaults shared by the web UI. All paths are project-relative."""
from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm
from processor import ACTION_MODES, GAP_POLICIES, Settings, run, scan, validate_lerobot_bundle

# -------- Editable defaults (CLI flags and web form can override these) --------
DEFAULT_INPUT_DIR = "datasets/hdf5/撕下快递单并贴到桌子上的快递上"
DEFAULT_OUTPUT_ROOT = None  # Auto: datasets/lerobot/<task>_<fps>hz_<mode>
DEFAULT_REPO_ID = None     # Local identifier; does not upload anything
TASK_DESCRIPTION = ""  # Default: preserve each HDF5 demo's language attribute.
DEFAULT_SOURCE_FPS = None  # Infer from each HDF5's timestamps
DEFAULT_FPS = 10           # Must divide every detected source FPS
DEFAULT_ACTION_MODE = "delta_joint"  # or "absolute_joint" (all 14 action dimensions)
DEFAULT_LIMIT = 0          # 0 = all episodes; positive values useful for a small trial
DEFAULT_GAP_POLICY = "reject"  # reject timestamp gaps; compact is an explicit opt-in


def defaults() -> Settings:
    return Settings(input_dir=DEFAULT_INPUT_DIR, output_root=DEFAULT_OUTPUT_ROOT,
                    task=TASK_DESCRIPTION, target_fps=DEFAULT_FPS,
                    source_fps=DEFAULT_SOURCE_FPS, action_mode=DEFAULT_ACTION_MODE,
                    repo_id=DEFAULT_REPO_ID, limit=DEFAULT_LIMIT,
                    gap_policy=DEFAULT_GAP_POLICY, input_dirs=[DEFAULT_INPUT_DIR])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-timing", nargs=2, action="append", default=[], metavar=("PATH", "REVISION"),
                        help="人工允许指定文件版本的物理时序异常（由检查页面生成）")
    parser.add_argument("--input-dir", action="append", dest="input_dirs", default=None,
                        help="HDF5 目录；可重复传入多个目录，按传入顺序合并")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="New output bundle directory")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default=TASK_DESCRIPTION)
    parser.add_argument("--task-source", choices=("hdf5", "override"), default="hdf5",
                        help="hdf5: 逐条继承 language；override: 统一使用 --task")
    parser.add_argument("--source-fps", type=int, default=DEFAULT_SOURCE_FPS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--action-mode", choices=(*ACTION_MODES, "absolute"), default=DEFAULT_ACTION_MODE,
                        help="absolute_joint 或 delta_joint；absolute 是旧兼容别名")
    parser.add_argument("--gap-policy", choices=GAP_POLICIES, default=DEFAULT_GAP_POLICY,
                        help="compact 压缩时间空档，reject 遇到空档直接失败")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--calibration-bundle", help="camera delta 使用的标定包 JSON")
    parser.add_argument("--calibration-mount-id", default="", help="确认源数据采集时的相机安装版本")
    parser.add_argument("--inspect", action="store_true", help="Check input and preview output counts only")
    parser.add_argument("--validate-output", metavar="PATH",
                        help="只验收已有 LeRobot bundle，不读取 HDF5 或加载模型")
    parser.add_argument("--validate-action-mode", choices=(*ACTION_MODES, "absolute"),
                        help="验收没有 processing.json 时指定 action 表示")
    args = parser.parse_args()
    if args.validate_output:
        import json
        expected = ({"action_mode": args.validate_action_mode}
                    if args.validate_action_mode else None)
        result = validate_lerobot_bundle(args.validate_output, expected=expected)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["valid"] else 1)
    input_dirs = args.input_dirs or [DEFAULT_INPUT_DIR]
    settings = Settings(input_dir=input_dirs[0], input_dirs=input_dirs, output_root=args.output_root,
                        task=args.task, task_source=args.task_source, target_fps=args.fps, source_fps=args.source_fps,
                        action_mode=args.action_mode, repo_id=args.repo_id, limit=args.limit,
                        gap_policy=args.gap_policy, calibration_bundle=args.calibration_bundle,
                        calibration_mount_id=args.calibration_mount_id,
                        timing_overrides={str(Path(path).expanduser().resolve()): rev for path, rev in args.allow_timing})
    if args.inspect:
        import json
        print(json.dumps(scan(settings), ensure_ascii=False, indent=2))
        return
    with tqdm(desc="DataProcessor", unit="episode") as bar:
        def progress(event):
            bar.total = event.get("episodes", bar.total)
            bar.n = event.get("episode", bar.n)
            bar.set_description(event["stage"])
            bar.set_postfix_str(event.get("message", "") + (
                f" {event['frame']}/{event['frames']}" if "frame" in event else ""))
        result = run(settings, progress)
    print("处理完成：", result["plan"]["output_root"])


if __name__ == "__main__":
    main()
