"""Preserve a valid OpenPI step-zero checkpoint during explicit resume."""

import logging
import json
from pathlib import Path
from datetime import datetime, timezone


def initialize_for_resume(initialize, *args, **kwargs):
    manager, resuming = initialize(*args, **kwargs)
    if kwargs.get("resume") and not kwargs.get("overwrite") and not resuming:
        # The vendor treats both () and (0,) as empty. Its step 0 checkpoint
        # actually stores the state after the first optimizer update.
        steps = tuple(manager.all_steps())
        if steps == (0,):
            logging.info("Restoring the completed OpenPI step-zero checkpoint")
            return manager, True
        manager.close()
        raise ValueError("恢复目录没有完整 checkpoint；请选择已保存的训练，或另建训练目录")
    return manager, resuming


def copy_resume_manifest(source, output, checkpoint_directories, *, status, command, pose_contract=None, steps=None):
    """Make newly resumed checkpoints portable without rewriting old ones."""
    source, output = Path(source), Path(output)
    if not source.is_file():
        if pose_contract is not None:
            raise ValueError("位姿恢复训练缺少原始 training_manifest.json")
        return
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest.update(status=status, command=command,
                    updated_at=datetime.now(timezone.utc).isoformat())
    if steps is not None:
        manifest["config"]["steps"] = steps
    if status == "completed":
        manifest["completed_at"] = manifest["updated_at"]
    destinations = [output / "training_manifest.json", *(Path(p) / "training_manifest.json" for p in checkpoint_directories)]
    for destination in destinations:
        if destination != output / "training_manifest.json" and destination.exists():
            continue
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
