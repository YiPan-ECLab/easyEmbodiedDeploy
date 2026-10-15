"""Recorder start-pose documents; no hardware is opened on import."""

import json
import math
import time
from contextlib import ExitStack

from .calibration import atomic_json
from .module_http import ROOT

START_PATH = ROOT / "datasets/recorder/start_point.json"


def normalize(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError("起点配置须为对象")
    if value is None or value.get("mode", "zero") == "zero":
        return dict(mode="zero", name="默认零位", arms={s: [0.0]*7 for s in ("left", "right")})
    if value.get("mode") != "custom":
        raise ValueError("起点模式须为 zero 或 custom")
    arms = value.get("arms", {})
    if not isinstance(arms, dict) or set(arms) != {"left", "right"}:
        raise ValueError("起点必须包含左右两侧，可分别从单臂关键点导入")
    result = dict(mode="custom", name=str(value.get("name") or "自定义起点"), arms={})
    for side, coordinates in arms.items():
        if not isinstance(coordinates, (list, tuple)):
            raise ValueError("起点须为六个关节角及夹爪宽度")
        q = [float(v) for v in coordinates]
        if len(q) != 7 or not all(math.isfinite(v) for v in q) or not 0 <= q[6] <= .1:
            raise ValueError("起点须为六个有限关节角（rad）及 0–0.1 m 夹爪宽度")
        result["arms"][side] = q
    return result


def load():
    return normalize(json.loads(START_PATH.read_text()) if START_PATH.exists() else None)


def save(value):
    value = normalize(value)
    START_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(START_PATH, value)
    return value


def workflow_points():
    """Read snapshots, without importing the workflow runtime or owning its files."""
    rows = []
    for path in sorted((ROOT / "datasets/workflows").glob("*.json")):
        try:
            document = json.loads(path.read_text())
            points = document if isinstance(document, list) else document.get("points", [])
            title = "共享点库" if isinstance(document, list) else document.get("name", path.stem)
            for point in points:
                arms = {}
                for side, q in point.get("arms", {}).items():
                    if point.get("space") == "joint":
                        arms[side] = q
                    else:
                        joints = point.get("source", {}).get("feedback", {}).get(side, {}).get("joint_angles")
                        if joints is not None:
                            arms[side] = [*joints, q[6]]
                if arms:
                    rows.append(dict(name=f"{title} / {point['name']}", arms=arms))
        except (ValueError, KeyError, TypeError):
            continue
    return rows


def capture(side="both"):
    """Read physical followers only. No enable, disable, role or motion writes."""
    from .resources import hardware_lease
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
    import record
    sides = ("left", "right") if side == "both" else (side,)
    if any(s not in ("left", "right") for s in sides):
        raise ValueError("请选择左侧、右侧或双侧")
    values = {}
    with ExitStack() as resources:
        resources.enter_context(hardware_lease())
        for s in sides:
            arm = AgxArmFactory.create_arm(create_agx_arm_config(robot=ArmModel.PIPER,
                firmeware_version=record.FIRMWARE, channel=record.PIPERS[s]["follower_name"]))
            resources.callback(arm.disconnect)
            gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            arm.connect()
            deadline = time.monotonic()+3
            while True:
                names = ("joint_12", "joint_34", "joint_56")
                groups = [getattr(arm._parser, name, None) for name in names]
                grip = gripper.get_gripper_status()
                now = time.time()
                if all(m is not None for m in groups) and grip is not None:
                    stamps = [float(m.timestamp) for m in groups]
                    if (all(-.02 <= now-t <= .15 for t in stamps) and max(stamps)-min(stamps) <= .05
                            and -.02 <= now-grip.timestamp <= .5 and grip.msg.mode == "width"):
                        q = [float(getattr(groups[i//2].msg, f"joint_{i+1}")) for i in range(6)]
                        if stamps == [float(getattr(arm._parser, n).timestamp) for n in names]:
                            values[s] = [*q, float(grip.msg.value)]
                            break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{s} 从臂没有完整新鲜反馈，未保存起点")
                time.sleep(.01)
    normalize(dict(mode="custom", arms={s: values.get(s, [0]*7) for s in ("left", "right")}))
    return values
