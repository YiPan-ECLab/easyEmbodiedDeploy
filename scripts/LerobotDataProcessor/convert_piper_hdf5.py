#!/usr/bin/env python3
"""DataProcessor CLI; defaults shared by the web UI. All paths are project-relative."""
from __future__ import annotations

import argparse
from tqdm import tqdm
from processor import Settings, run, scan

# -------- Editable defaults (CLI flags and web form can override these) --------
DEFAULT_INPUT_DIR = "datasets/hdf5/撕下快递单并贴到桌子上的快递上"
DEFAULT_OUTPUT_ROOT = None  # Auto: datasets/processed/<task>_<fps>hz_<mode>
DEFAULT_REPO_ID = None     # Local identifier; does not upload anything
TASK_DESCRIPTION = "Remove the shipping label and place it on the package on the table."
DEFAULT_SOURCE_FPS = None  # Infer from each HDF5's timestamps; or explicitly set 20
DEFAULT_FPS = 20           # Output FPS, e.g. 10 for 20 -> 10 Hz
DEFAULT_ACTION_MODE = "absolute"  # "absolute" or "delta_joint" (all 14 action dimensions)
DEFAULT_LIMIT = 0          # 0 = all episodes; positive values useful for a small trial


def defaults() -> Settings:
    return Settings(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_ROOT, TASK_DESCRIPTION, DEFAULT_FPS,
                    DEFAULT_SOURCE_FPS, DEFAULT_ACTION_MODE, DEFAULT_REPO_ID, DEFAULT_LIMIT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="New output bundle directory")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default=TASK_DESCRIPTION)
    parser.add_argument("--source-fps", type=int, default=DEFAULT_SOURCE_FPS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--action-mode", choices=("absolute", "delta_joint"), default=DEFAULT_ACTION_MODE)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--inspect", action="store_true", help="Check input and preview output counts only")
    args = parser.parse_args()
    settings = Settings(args.input_dir, args.output_root, args.task, args.fps,
                        args.source_fps, args.action_mode, args.repo_id, args.limit)
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
