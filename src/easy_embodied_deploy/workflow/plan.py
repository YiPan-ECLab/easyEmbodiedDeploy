"""Portable workflow documents, interpolation and camera-point geometry."""

import copy
import json
import math
import uuid
from pathlib import Path

import cv2
import numpy as np

from ..action_contract import ACTION_MODES
from ..calibration import atomic_json
from ..calibration_math import intrinsics
from ..pose_actions import (finite, flange_matrix, validate_transform, rpy_matrix,
                            matrix_rpy, rotvec_matrix, matrix_rotvec)
from .devices import normalize_device

SCHEMA = "easy_embodied_workflow"
SIDES = ("left", "right")


def number(value, label, low=0, high=float("inf")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} 须为有效数字") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} 必须在 {low:g}–{high:g} 之间")
    return value


def new_plan():
    return dict(schema=SCHEMA, version=1, name="我的工作流", points=[], models=[], steps=[],
                mode="sequential", continuous_vla=dict(model="", task="", takeovers=[], switches=[]),
                settings=dict(joint_command="move_js", control_hz=50, joint_speed=3.0, joint_acceleration=3.0,
                              linear_speed=1.0, angular_speed=.5, gripper_speed=1.0,
                              transition_s=.6, prefetch_s=1.0))


def unique_name(name, rows):
    names = {str(row.get('name') or '').strip() for row in rows}
    candidate, index = name, 2
    while candidate in names:
        candidate = f'{name} {index}'
        index += 1
    return candidate


def validate_names(rows, label):
    names = set()
    for row in rows:
        name = row.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'请为{label}命名')
        name = row['name'] = name.strip()
        if name in names:
            raise ValueError(f'{label}名称「{name}」已存在，请使用不同名称')
        names.add(name)


def repair_vla_names(plan):
    """Upgrade old saved cards without changing model references or execution."""
    used = []
    models = {row['id']: row.get('name', row['id']) for row in plan.get('models', [])}
    for step in plan.get('steps', []):
        if step.get('type') == 'vla':
            base = (step.get('name') or models.get(step.get('model')) or 'VLA').strip() or 'VLA'
            step['name'] = unique_name(base, used)
            used.append(step)
    return plan


def validate(plan):
    plan = copy.deepcopy(plan)
    if plan.get("schema") != SCHEMA or plan.get("version") != 1:
        raise ValueError("不是受支持的工作流文件")
    if not isinstance(plan.get("name"), str) or not plan["name"].strip():
        raise ValueError("请为工作流命名")
    settings = {**new_plan()["settings"], **plan.get("settings", {})}
    if settings["joint_command"] not in ("move_js", "move_j"):
        raise ValueError("关节下发接口须为 move_js 或 move_j")
    for key, value in settings.items():
        if key == "joint_command":
            continue
        settings[key] = number(value, key, .001)
    number(settings["control_hz"], "控制频率", 10, 100)
    plan["settings"] = settings
    plan.setdefault('mode', 'sequential')
    plan.setdefault('continuous_vla', dict(model='', task='', takeovers=[], switches=[]))
    if plan['mode'] not in ('sequential', 'continuous_vla'):
        raise ValueError('执行模式无效')
    tables = {}
    for kind in ("points", "models", "steps"):
        rows = plan.get(kind, [])
        if not isinstance(rows, list):
            raise ValueError(f"{kind} 必须为列表")
        ids = [row.get("id") for row in rows]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{kind} 的标识不能为空或重复")
        tables[kind] = {row["id"]: row for row in rows}
    validate_names(plan['points'], '共享关键点')
    validate_names(plan['models'], 'VLA 模型')
    # Older API callers may omit card labels. Explicit names must be unique.
    vla_steps = [s for s in plan['steps'] if s.get('type') == 'vla']
    for step in vla_steps:
        if 'name' not in step:
            model = tables['models'].get(step.get('model'), {})
            step['name'] = unique_name(model.get('name', 'VLA'), vla_steps)
    validate_names(vla_steps, 'VLA 步骤')
    for point in plan["points"]:
        if not isinstance(point.get("name"), str) or not point["name"].strip():
            raise ValueError("请为关键点命名")
        if point.get("space") not in ("joint", "pose"):
            raise ValueError("关键点必须使用关节或末端坐标")
        arms = point.get("arms", {})
        if not arms or set(arms) - set(SIDES):
            raise ValueError("关键点须包含左臂或右臂")
        for value in arms.values():
            value = finite(value, (7,), "关键点：6 轴 + 夹爪")
            number(value[6], "夹爪宽度/m", 0, .1)
            if point["space"] == "pose":
                if abs(value[4]) > math.pi / 2 or max(abs(value[3]), abs(value[5])) > math.pi:
                    raise ValueError("末端姿态使用主值 RPY 弧度角")
    for model in plan["models"]:
        model['device'] = normalize_device(model.get('device', 'cuda:0'))
        if model.get("backend") not in ("lerobot", "openpi", "mock"):
            raise ValueError("模型后端须为 LeRobot 或 OpenPI")
        if model.get("action_mode") not in ACTION_MODES:
            raise ValueError("模型动作模式无效")
        # Drafts may contain unfinished models. Check required checkpoints only
        # for models being loaded/executed, not unrelated library entries.
        if not isinstance(model.get("checkpoint", ""), str):
            raise ValueError(f"模型「{model.get('name', model['id'])}」的 checkpoint 须为路径文本")
        model["checkpoint"] = model.get("checkpoint", "").strip()
        model["policy_hz"] = number(model.get("policy_hz", 10), "policy Hz", .001)
        model["control_hz"] = number(model.get("control_hz", settings["control_hz"]), "VLA control Hz", .001)
        model["smoothing_time_constant"] = number(model.get("smoothing_time_constant", .1), "时间平滑常数", 0)
        model.setdefault("inference_mode", "rtc" if model.get("rtc", True) else "sync")
        model.setdefault("rtc_mode", "guided")
        model.setdefault("rtc_schedule", "exp")
        model.setdefault("rtc_mask_grippers", True)
        if model["inference_mode"] not in ("rtc", "sync") or model["rtc_mode"] not in ("guided", "trained"):
            raise ValueError("请选择有效的推理模式与 RTC 模式")
        if model["inference_mode"] == "sync" and model["rtc_mode"] == "trained":
            raise ValueError("trained RTC 需要 RTC 推理模式")
        if model["rtc_schedule"] not in ("exp", "linear", "ones", "zeros"):
            raise ValueError("RTC schedule 无效")
        for key, default, low in (("rtc_horizon",20,1), ("rtc_queue_threshold",40,0),
                                  ("sample_steps",10,1), ("n_action_steps",0,0)):
            value = number(model.get(key, default), key, low)
            if not value.is_integer():
                raise ValueError(f"{key} 须为整数")
            model[key] = int(value)
        model["rtc_max_guidance_weight"] = number(model.get("rtc_max_guidance_weight",5), "RTC 引导权重", 0)
    for step in plan["steps"]:
        kind = step.get("type")
        if kind == "point":
            if step.get("point") not in tables["points"]:
                raise ValueError("运动步骤引用的关键点不存在")
            step["duration"] = number(step.get("duration", 1), "运动时长", .05)
        elif kind == "vla":
            if step.get("side", "both") not in (*SIDES, "both"):
                raise ValueError("VLA 控制侧别无效")
            if step.get("model") not in tables["models"]:
                raise ValueError("VLA 步骤引用的模型不存在")
            if step.get("until", "duration") not in ("duration", "actions", "manual"):
                raise ValueError("VLA 结束方式无效")
            if step.get("until", "duration") != "manual":
                step["value"] = number(step.get("value", 10), "VLA 时长/动作数", 1)
        elif kind == "wait":
            step["duration"] = number(step.get("duration", 1), "等待时长", 0)
        elif kind == "gripper":
            if step.get("side", "both") not in (*SIDES, "both"):
                raise ValueError("夹爪侧别无效")
            step["width"] = number(step.get("width", .04), "夹爪宽度/m", 0, .1)
            step["duration"] = number(step.get("duration", 1), "夹爪时长", .05)
        else:
            raise ValueError(f"未知步骤：{kind}")
    if plan['mode'] == 'continuous_vla':
        continuous = plan['continuous_vla']
        if not isinstance(continuous, dict) or continuous.get('model') not in tables['models']:
            raise ValueError('请选择持续运行的 VLA 模型')
        if not isinstance(continuous.get('task', ''), str):
            raise ValueError('持续 VLA 任务指令须为文本')
        if continuous.get('side', 'both') not in (*SIDES, 'both'):
            raise ValueError('持续 VLA 控制侧别无效')
        events = continuous.setdefault('takeovers', [])
        if not isinstance(events, list):
            raise ValueError('定时接管须为列表')
        identifiers, times = set(), set()
        for event in events:
            if not isinstance(event, dict) or not isinstance(event.get('id'), str) or not event['id'] or event['id'] in identifiers:
                raise ValueError('定时接管标识不能为空或重复')
            identifiers.add(event['id'])
            event['at_s'] = number(event.get('at_s', 5), '接管时间/秒', 0)
            trigger = event.setdefault('trigger', dict(type='time'))
            # User replaced the J5/J6 AND rule with the signed J4+J6 sum.
            # Upgrade old saved plans and browser drafts without carrying a
            # J5 angle, direction or threshold into the new controller.
            if isinstance(trigger, dict) and trigger.get('type') == 'right_joints':
                trigger = event['trigger'] = {k: v for k, v in trigger.items()
                    if k in ('end_s', 'hold_s', 'reference', 'j6_reference_deg')}
                trigger.update(type='right_joint_sum', sum_deg=30, j4_reference_deg=0)
            if not isinstance(trigger, dict) or trigger.get('type') not in ('time', 'right_joint_sum', 'right_j6_pose', 'right_joint_sum_pose'):
                raise ValueError('接管触发方式须为定时或右臂关节条件')
            if trigger['type'] == 'time':
                if event['at_s'] in times:
                    raise ValueError('同一时间的关键点请放在同一个接管动作段中')
                times.add(event['at_s'])
            else:
                trigger['end_s'] = number(trigger.get('end_s', float('nan')), '触发区间结束/秒', 0)
                if trigger['end_s'] <= event['at_s']:
                    raise ValueError('触发区间结束时间必须晚于开始时间')
                trigger['hold_s'] = number(trigger.get('hold_s', .2), '关节条件持续时间/秒', .05, 5)
                if trigger['end_s']-event['at_s'] < trigger['hold_s']:
                    raise ValueError('触发时间区间不能短于条件持续时间')
                if trigger['type'] == 'right_joint_sum_pose':
                    high_z_defined = 'high_z_mm' in trigger
                    high_z = trigger.get('high_z_mm')
                    trigger = event['trigger'] = dict(type='right_joint_sum_pose',
                        end_s=trigger['end_s'], hold_s=trigger['hold_s'],
                        sum_deg=number(trigger.get('sum_deg', 30), 'J4＋J6 实际角度合计下限/度', -220, 220),
                        z_mm=number(trigger.get('z_mm', 250), '右臂基座系法兰 Z 下限/mm', -700, 750))
                    if high_z is not None:
                        high_z = number(high_z, '独立触发法兰 Z 下限/mm', -700, 750)
                        if high_z <= trigger['z_mm']:
                            raise ValueError('独立触发高度必须高于角度联合条件的高度下限')
                        trigger['high_z_mm'] = high_z
                    elif high_z_defined:
                        trigger['high_z_mm'] = None
                elif trigger['type'] == 'right_j6_pose':
                    # Absolute signed joint feedback and base-frame flange Z.
                    trigger = event['trigger'] = dict(type='right_j6_pose',
                        end_s=trigger['end_s'], hold_s=trigger['hold_s'],
                        j6_deg=number(trigger.get('j6_deg', -10), 'J6 实际角度下限/度', -120, 120),
                        z_mm=number(trigger.get('z_mm', 500), '右臂基座系法兰 Z 下限/mm', -700, 750))
                else:
                    trigger.setdefault('reference', 'window')
                    if trigger['reference'] not in ('window', 'manual'):
                        raise ValueError('关节触发基准须为窗口首次反馈或指定角度')
                    trigger['sum_deg'] = number(trigger.get('sum_deg', 30), 'J4＋J6 合计转角阈值/度', .1, 720)
                    for joint in ('j4', 'j6'):
                        if trigger['reference'] == 'manual':
                            trigger[f'{joint}_reference_deg'] = number(trigger.get(f'{joint}_reference_deg', 0), f'{joint} 基准/度', -180, 180)
                    for key in ('j5_deg', 'j5_direction', 'j5_reference_deg', 'j4_deg', 'j4_direction', 'j6_deg', 'j6_direction'):
                        trigger.pop(key, None)
            rows = event.get('steps')
            if not isinstance(rows, list) or not rows:
                raise ValueError('每个接管动作段至少添加一个关键点')
            ids = set()
            for row in rows:
                if not isinstance(row, dict) or row.get('type') != 'point' or row.get('point') not in tables['points']:
                    raise ValueError('定时接管只支持引用已有关键点')
                if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids:
                    raise ValueError('接管关键点标识不能为空或重复')
                ids.add(row['id'])
                row['duration'] = number(row.get('duration', 1), '接管运动时长', .05)
        events.sort(key=lambda event: event['at_s'])
        switches = continuous.setdefault('switches', [])
        if not isinstance(switches, list):
            raise ValueError('VLA 切换时间表须为列表')
        switch_ids, switch_times = set(), set()
        for switch in switches:
            if not isinstance(switch, dict) or not isinstance(switch.get('id'), str) or not switch['id'] or switch['id'] in switch_ids:
                raise ValueError('VLA 切换标识不能为空或重复')
            switch_ids.add(switch['id'])
            switch['at_s'] = number(switch.get('at_s', 5), 'VLA 切换时间/秒', 0)
            if switch['at_s'] in switch_times:
                raise ValueError('同一时间只能切换到一个 VLA 模型')
            switch_times.add(switch['at_s'])
            if switch.get('model') not in tables['models']:
                raise ValueError('VLA 切换引用的模型不存在')
            if not isinstance(switch.get('task', ''), str):
                raise ValueError('VLA 切换任务指令须为文本')
        switches.sort(key=lambda switch: switch['at_s'])
    return plan


def execution_steps(plan):
    if plan.get('mode') == 'continuous_vla':
        return [step for event in plan['continuous_vla']['takeovers'] for step in event['steps']]
    return plan['steps']


def used_model_ids(plan, steps=None):
    if plan.get('mode') == 'continuous_vla':
        config = plan['continuous_vla']
        return {config['model']} | {switch['model'] for switch in config.get('switches', [])}
    return {step['model'] for step in (plan['steps'] if steps is None else steps) if step['type'] == 'vla'}


def require_models(models, identifiers):
    table = {model['id']: model for model in models}
    for identifier in identifiers:
        if identifier not in table:
            raise ValueError(f"模型不存在：{identifier}")
        model = table[identifier]
        if model['backend'] != 'mock' and not model.get('checkpoint', '').strip():
            raise ValueError(f"模型「{model.get('name', identifier)}」未填写 checkpoint；请编辑该模型")


class PlanStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, identifier):
        if not isinstance(identifier, str) or not identifier or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("工作流 ID 无效")
        return self.directory / f"{identifier}.json"

    def save(self, plan, identifier=None):
        plan = validate(plan)
        identifier = identifier or uuid.uuid4().hex
        path = self.path(identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, plan)
        return identifier

    def load(self, identifier):
        return validate(json.loads(self.path(identifier).read_text()))

    def listing(self):
        result = []
        for path in sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                plan = self.load(path.stem)
                result.append(dict(id=path.stem, name=plan["name"], steps=len(plan["steps"])))
            except (ValueError, KeyError):
                continue
        return result


def interpolate(start, end, fraction, space="joint", v0=None, v1=None, duration=1):
    """Quintic joint path (shared knot velocity, zero knot acceleration).

    Cartesian orientation follows the shortest SO(3) arc rather than Euler
    subtraction, including crossing +/-pi. Positions/grippers use quintics.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    u = float(np.clip(fraction, 0, 1))
    a = np.zeros_like(start) if v0 is None else np.asarray(v0) * duration
    b = np.zeros_like(start) if v1 is None else np.asarray(v1) * duration
    d = end - start
    out = start + a*u + (10*d-6*a-4*b)*u**3 + (-15*d+8*a+7*b)*u**4 + (6*d-3*a-3*b)*u**5
    if space == "pose":
        blend = 10*u**3 - 15*u**4 + 6*u**5
        for offset in (0, 7):
            rotation = rpy_matrix(start[offset+3:offset+6])
            delta = matrix_rotvec(rpy_matrix(end[offset+3:offset+6]) @ rotation.T)
            out[offset+3:offset+6] = matrix_rpy(rotvec_matrix(blend*delta) @ rotation)
    return out


def tangent(previous, current, following, before_s, after_s):
    a, b = (current-previous)/before_s, (following-current)/after_s
    same = a*b > 0
    return np.where(same, np.sign(a)*np.minimum((np.abs(a)+np.abs(b))/2,
                                               1.5*np.minimum(np.abs(a), np.abs(b))), 0)


def camera_point(snapshot, pixel, bundle, side, tcp_offset, offset):
    """A frozen RGB/depth/pose snapshot is the sole anchor for a picked pixel."""
    if side not in SIDES or bundle.get("schema") != "easy_embodied_camera_calibration":
        raise ValueError("请选择机械臂与有效标定包")
    camera = snapshot["camera"]
    role = camera["role"]
    job = f"global_{side}" if role == "agentview" else f"{side}_wrist"
    if role not in ("agentview", f"{side}_eye_in_hand"):
        raise ValueError("腕部相机只能使用对应机械臂的标定")
    matches = [item for item in bundle.get("calibrations", []) if item.get("job") == job]
    if len(matches) != 1:
        raise ValueError(f"标定包须包含 {job}")
    item = matches[0]
    result = item["result"]
    expected = f"T_{side}_base_global_camera" if role == "agentview" else f"T_{side}_flange_{side}_camera"
    if result.get("transform") != expected or not result.get("accepted"):
        raise ValueError("外参方向错误或尚未确认导出")
    if any(item["camera"].get(k) != camera.get(k) for k in ("serial", "sensor", "width", "height")):
        raise ValueError("当前相机身份/视频配置与标定不一致")
    x, y = finite(pixel, (2,), "像素")
    width, height = camera["width"], camera["height"]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("选点在图像之外")
    depth = snapshot.get("depth")
    if depth is None or depth.shape != (height, width):
        raise ValueError("该相机没有与当前 RGB 对齐的深度；可使用全局深度相机或手动坐标")
    ix, iy = int(x), int(y)
    # Do not replace an invalid clicked pixel with foreground/background nearby.
    z = float(depth[iy, ix])
    if not math.isfinite(z) or z <= 0:
        raise ValueError("选中像素没有有效深度，请选择物体表面其他位置")
    # Use the same intrinsics used to solve these extrinsics, including an
    # operator-calibrated lens, rather than silently reverting to SDK defaults.
    intr = item["camera"].get("intrinsics", camera["intrinsics"])
    uv = np.array([[[x, y]]], dtype=float)
    K, D, model = intrinsics(intr)
    if model == "fisheye":
        ray = cv2.fisheye.undistortPoints(uv, K, D).reshape(2)
    elif model == "brown":
        ray = cv2.undistortPoints(uv, K, D).reshape(2)
    else:
        raise ValueError("当前相机畸变模型不支持选点")
    T = validate_transform(result["matrix"])
    pose = finite(snapshot["poses"][side], (6,), "快照法兰位姿")
    if role != "agentview":
        T = flange_matrix(pose) @ T
    xyz = (T @ np.r_[ray*z, z, 1])[:3] + finite(offset, (3,), "base 偏移")
    # The chosen surface is the TCP target; SDK move_p controls the flange.
    flange_xyz = xyz - rpy_matrix(pose[3:]) @ finite(tcp_offset, (3,), "TCP 偏移")
    target = np.r_[flange_xyz, pose[3:], snapshot["joints"][6 if side == "left" else 13]]
    return dict(space="pose", arms={side: target.tolist()},
                source=dict(kind="camera", pixel=[float(x), float(y)], depth_m=z,
                            camera=role, serial=camera["serial"], mount_id=item["mount_id"],
                            tcp_offset=list(tcp_offset), base_offset=list(offset), target_tcp=xyz.tolist()))
