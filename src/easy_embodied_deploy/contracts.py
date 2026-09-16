"""Small configuration contracts shared by native model consoles."""

import json
from pathlib import Path


def boolean(value, name, error=ValueError):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise error(f"{name} 必须是布尔值")


def checkpoint_action_mode(checkpoint, selected):
    """Honor explicit training metadata; legacy checkpoints remain selectable."""
    checkpoint = Path(checkpoint)
    for folder in (checkpoint, *list(checkpoint.parents)[:3]):
        path = folder / "training_manifest.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        mode = manifest.get("config", {}).get("action_mode")
        if mode and mode != selected:
            raise ValueError(f"checkpoint 动作模式为 {mode}，所选模式为 {selected}")
        return mode
    return None
