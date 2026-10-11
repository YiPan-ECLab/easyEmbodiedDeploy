#!/home/py/main/easyEmbodiedDeploy/thirdParty/.venvs/piper-home/bin/python
"""Inspect or recover the known Piper leader J2 zero-offset failure.

This utility deliberately supports only the failure signature observed on this
rig.  It never searches for a mechanical stop and never moves a joint.  The
operator must first verify that J2 is resting at its physical lower-zero mark.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from easy_embodied_deploy.resources import hardware_lease
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

from piper_recorder.robot import verify_can_interface

ARMS = {"left": "piper-lt", "right": "piper-rt"}
JOINT_INDEX = 2
KNOWN_FAULT_CODE = 0xA4
MIN_OFFSET_DEG = 2.0
MAX_OFFSET_DEG = 10.0
OTHER_JOINT_TOLERANCE_DEG = 2.0
MAX_STATIONARY_VELOCITY_RAD_S = 0.05
ZERO_VERIFY_TOLERANCE_DEG = 0.5
RESULT_PREFIX = "PIPER_ZERO_RESULT="


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _wait_regular_feedback(arm, timeout=5.0):
    previous = arm.get_joint_angles()
    previous_timestamp = 0.0 if previous is None else previous.timestamp
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        arm.set_follower_mode()
        message = arm.get_joint_angles()
        if message is not None and message.timestamp > previous_timestamp:
            values = [_finite(value) for value in message.msg]
            if len(values) == 6 and all(value is not None for value in values):
                return values
        time.sleep(0.05)
    raise RuntimeError("切换受控模式后没有收到新的六关节反馈")


def _snapshot(arm, joints=None):
    if joints is None:
        message = arm.get_joint_angles()
        if message is None or len(message.msg) != 6:
            raise RuntimeError("没有完整的六关节反馈")
        joints = [_finite(value) for value in message.msg]
    if any(value is None for value in joints):
        raise RuntimeError("关节反馈包含 NaN/Inf")

    drivers = []
    motors = []
    for index in range(1, 7):
        driver_message = arm.get_driver_states(index)
        motor_message = arm.get_motor_states(index)
        if driver_message is None or motor_message is None:
            raise RuntimeError(f"J{index} 驱动器反馈不完整")
        driver = driver_message.msg
        foc = driver.foc_status
        motor = motor_message.msg
        drivers.append(
            {
                "joint": index,
                "code": int(driver.foc_status_code),
                "enabled": bool(foc.driver_enable_status),
                "voltage_too_low": bool(foc.voltage_too_low),
                "motor_overheating": bool(foc.motor_overheating),
                "driver_overcurrent": bool(foc.driver_overcurrent),
                "driver_overheating": bool(foc.driver_overheating),
                "collision": bool(foc.collision_status),
                "driver_error": bool(foc.driver_error_status),
                "stall": bool(foc.stall_status),
                "voltage_v": _finite(driver.vol),
                "driver_temp_c": _finite(driver.foc_temp),
                "motor_temp_c": _finite(driver.motor_temp),
                "bus_current_a": _finite(driver.bus_current),
            }
        )
        motors.append(
            {
                "joint": index,
                "position_rad": _finite(motor.position),
                "velocity_rad_s": _finite(motor.velocity),
                "current_a": _finite(motor.current),
                "torque_nm": _finite(motor.torque),
            }
        )
    return {
        "joints_rad": list(joints),
        "joints_deg": [math.degrees(value) for value in joints],
        "drivers": drivers,
        "motors": motors,
    }


def _wait_snapshot(arm, joints=None, timeout=3.0):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return _snapshot(arm, joints)
        except RuntimeError as exc:
            last_error = exc
            joints = None
            time.sleep(0.05)
    raise RuntimeError(f"等待完整驱动器反馈超时：{last_error}")


def assess_snapshot(snapshot):
    """Return fail-closed eligibility for the field-confirmed J2 fault."""
    reasons = []
    joints = snapshot.get("joints_deg", [])
    drivers = snapshot.get("drivers", [])
    motors = snapshot.get("motors", [])
    if len(joints) != 6 or len(drivers) != 6 or len(motors) != 6:
        return False, ["反馈维度不完整"]

    j2 = drivers[1]
    motor = motors[1]
    offset = float(joints[1])
    other_max = max(abs(float(value)) for index, value in enumerate(joints) if index != 1)
    if not MIN_OFFSET_DEG <= offset <= MAX_OFFSET_DEG:
        reasons.append(
            f"J2 当前 {offset:+.3f}°，不在已验证的 {MIN_OFFSET_DEG:g}–{MAX_OFFSET_DEG:g}° 偏移范围"
        )
    if other_max > OTHER_JOINT_TOLERANCE_DEG:
        reasons.append(
            f"其余关节最大偏离 {other_max:.3f}°，尚未处于零位附近"
        )
    if int(j2.get("code", -1)) != KNOWN_FAULT_CODE:
        reasons.append(
            f"J2 状态码为 0x{int(j2.get('code', -1)) & 0xff:02X}，不是已验证的 0x{KNOWN_FAULT_CODE:02X}"
        )
    if bool(j2.get("enabled")):
        reasons.append("J2 仍处于使能状态")
    velocity = motor.get("velocity_rad_s")
    if velocity is None or abs(float(velocity)) > MAX_STATIONARY_VELOCITY_RAD_S:
        reasons.append("J2 未确认静止")
    return not reasons, reasons


def _wait_joint_enabled(arm, enabled, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(arm.get_joint_enable_status(JOINT_INDEX)) is enabled:
            return
        if enabled:
            arm.enable(JOINT_INDEX)
        else:
            arm.disable(JOINT_INDEX)
        time.sleep(0.05)
    state = "使能" if enabled else "失能"
    raise RuntimeError(f"J2 {state}反馈超时")


def _wait_zero_feedback(arm, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = arm.get_joint_angles()
        if message is not None and len(message.msg) == 6:
            value = _finite(message.msg[1])
            if value is not None and abs(math.degrees(value)) <= ZERO_VERIFY_TOLERANCE_DEG:
                return
        time.sleep(0.05)
    raise RuntimeError("标定已应答，但 J2 零位读回验证失败")


def run(side, repair=False, confirmed=False):
    if side not in ARMS:
        raise ValueError("side 必须是 left 或 right")
    if repair and not confirmed:
        raise ValueError("必须人工确认 J2 已位于物理下限零位")

    channel = ARMS[side]
    verify_can_interface(channel)
    config = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.V189,
        channel=channel,
    )
    arm = AgxArmFactory.create_arm(config)
    arm.connect()
    restored = False
    try:
        joints = _wait_regular_feedback(arm)
        before = _wait_snapshot(arm, joints)
        eligible, reasons = assess_snapshot(before)
        result = {
            "ok": True,
            "operation": "repair" if repair else "inspect",
            "side": side,
            "arm": channel,
            "joint": JOINT_INDEX,
            "eligible": eligible,
            "reasons": reasons,
            "before": before,
        }
        if not repair:
            return result
        if not eligible:
            raise RuntimeError("拒绝自动标零：" + "；".join(reasons))

        # No position command is sent.  The operator-confirmed physical pose is
        # written as J2 zero only after the fault signature and stillness gates.
        arm.disable(JOINT_INDEX)
        _wait_joint_enabled(arm, False)
        if not arm.clear_joint_error(JOINT_INDEX, timeout=2.0):
            raise RuntimeError("J2 清除错误未收到 ACK")
        time.sleep(0.2)
        arm.disable(JOINT_INDEX)
        _wait_joint_enabled(arm, False)
        if not arm.calibrate_joint(JOINT_INDEX, timeout=2.0):
            raise RuntimeError("J2 设置当前位置为零未收到成功 ACK")
        _wait_zero_feedback(arm)
        if not arm.clear_joint_error(JOINT_INDEX, timeout=2.0):
            raise RuntimeError("J2 标零后清错未收到 ACK")
        arm.enable(JOINT_INDEX)
        _wait_joint_enabled(arm, True)
        time.sleep(0.2)
        after = _wait_snapshot(arm)
        driver = after["drivers"][1]
        if (
            abs(after["joints_deg"][1]) > ZERO_VERIFY_TOLERANCE_DEG
            or driver["code"] != 0x40
            or not driver["enabled"]
        ):
            raise RuntimeError(
                "J2 标零后状态异常："
                f"位置={after['joints_deg'][1]:+.3f}°，状态码=0x{driver['code']:02X}"
            )
        result.update(repaired=True, after=after)
        return result
    finally:
        try:
            arm.set_leader_mode()
            time.sleep(0.1)
            arm.restore_leader_drag_mode()
            restored = True
        finally:
            arm.disconnect()
        if not restored:
            raise RuntimeError(f"{channel} 未能恢复 leader 拖动模式")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", required=True, choices=tuple(ARMS))
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--confirm-physical-zero", action="store_true")
    args = parser.parse_args()
    try:
        with hardware_lease():
            result = run(
                args.side,
                repair=args.repair,
                confirmed=args.confirm_physical_zero,
            )
    except Exception as exc:  # noqa: BLE001 - JSON is the maintenance API boundary
        result = {
            "ok": False,
            "operation": "repair" if args.repair else "inspect",
            "side": args.side,
            "arm": ARMS[args.side],
            "error": str(exc),
        }
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
