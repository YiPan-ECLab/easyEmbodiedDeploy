#!/usr/bin/env python3
"""Run the CPU regression matrix in each module's pinned environment."""

import argparse
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = [
    ("piper-home", "scripts/PiperRecorder/tests", "test*.py"),
    ("lerobot-pi05", "scripts/LerobotTrainer/tests", "test*.py"),
    ("lerobot-pi05", "scripts/LerobotDeployer/tests", "test*.py"),
    ("openpi-jax", "scripts/OpenpiTrainer/tests", "test*.py"),
    ("openpi-jax", "scripts/OpenpiDeployer/tests", "test*.py"),
    ("piper-home", "tests", "test_runtime_boundaries.py"),
    ("piper-home", "tests", "test_resources_calibration.py"),
    ("lerobot-pi05", "tests", "test_data_pipeline.py"),
    ("piper-home", "tests", "test_pose_actions.py"),
    ("piper-home", "tests", "test_pose_runtime.py"),
    ("lerobot-pi05", "tests", "test_pose_training_torch.py"),
    ("openpi-jax", "tests", "test_pose_training_jax.py"),
    ("lerobot-pi05", "tests", "test_pose_pipeline.py"),
    ("piper-home", "tests", "test_pose_hardware.py"),
    ("piper-home", "tests", "test_workflow.py"),
    ("piper-home", "tests", "test_workflow_library.py"),
    ("piper-home", "tests", "test_workflow_continuous.py"),
    ("piper-home", "tests", "test_workflow_switches.py"),
    ("piper-home", "tests", "test_workflow_rtc_timing.py"),
    ("piper-home", "tests", "test_workflow_triggers.py"),
    ("piper-home", "tests", "test_workflow_recording.py"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    reports = args.report_dir or Path(tempfile.mkdtemp(prefix="easy-embodied-checks-"))
    reports.mkdir(parents=True, exist_ok=True)
    failures = []
    for index, (environment, directory, pattern) in enumerate(SUITES):
        python = ROOT / "thirdParty/.venvs" / environment / "bin/python"
        log = reports / f"{index}-{Path(directory).parent.name}-{pattern}.log"
        print(f"RUN {environment}: {directory}/{pattern}", flush=True)
        with log.open("w") as output:
            result = subprocess.run(
                [
                    str(python),
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    directory,
                    "-p",
                    pattern,
                    "-v",
                ],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        print("\n".join(log.read_text().splitlines()[-4:]), flush=True)
        if result.returncode:
            failures.append(str(log))
    print(f"Reports: {reports}")
    if failures:
        print("FAILED:\n" + "\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
