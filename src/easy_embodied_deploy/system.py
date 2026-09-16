"""Read-only device inventory, independent of either ML environment."""

import csv
import subprocess
from pathlib import Path
from typing import Any


def _optional_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def gpu_status() -> list[dict[str, Any]]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            query, check=True, capture_output=True, text=True, timeout=3
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    gpus: list[dict[str, Any]] = []
    by_uuid: dict[str, dict[str, Any]] = {}
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 7:
            continue
        item = {
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "memory_used_mb": _optional_int(row[3]),
            "memory_total_mb": _optional_int(row[4]),
            "utilization": _optional_int(row[5]),
            "temperature": _optional_int(row[6]),
            "processes": [],
        }
        gpus.append(item)
        by_uuid[item["uuid"]] = item
    try:
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        for row in csv.reader(processes.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 4 or row[0] not in by_uuid:
                continue
            by_uuid[row[0]]["processes"].append(
                {
                    "pid": int(row[1]),
                    "name": Path(row[2]).name,
                    "memory_mb": _optional_int(row[3]),
                }
            )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return gpus
