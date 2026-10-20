"""Follower diagnosis and bounded recovery; never calibrates encoder zeroes."""

import time

import numpy as np

from ..pose_actions import finite

FAULTS = {
    'voltage_too_low': '欠压', 'motor_overheating': '电机过温',
    'driver_overcurrent': '过流', 'driver_overheating': '驱动过温',
    'collision_status': '碰撞', 'driver_error_status': '驱动错误',
    'stall_status': '堵转',
}
MAX_CORRECTION = np.deg2rad(3)


def fresh(message, label):
    if message is None or not -.02 <= time.time()-float(message.timestamp) <= .5:
        raise RuntimeError(f'{label}反馈缺失或过期；请检查供电、CAN 和从臂模式，不能据此判断零位异常')
    return message.msg


def inspect_arm(arm, limits):
    status = fresh(arm.get_arm_status(), '控制器')
    joints = finite(fresh(arm.get_joint_angles(), '关节'), (6,), '关节反馈')
    limits = np.asarray(limits)
    target = np.clip(joints, limits[:, 0], limits[:, 1])
    correction = target-joints
    drivers, reasons = [], []
    for index in range(1, 7):
        driver = fresh(arm.get_driver_states(index), f'J{index} 驱动器')
        motor = fresh(arm.get_motor_states(index), f'J{index} 电机')
        flags = [name for key, name in FAULTS.items() if bool(getattr(driver.foc_status, key, False))]
        speed = float(motor.velocity)
        if not np.isfinite(speed) or abs(speed) > .05:
            reasons.append(f'J{index} 尚未静止')
        if flags:
            reasons.append(f'J{index}：{"、".join(flags)}，请先排查故障，不能直接使能或标零')
        drivers.append(dict(joint=index, code=int(driver.foc_status_code),
                            enabled=bool(driver.foc_status.driver_enable_status),
                            faults=flags, velocity_rad_s=speed))
    arm_status, ctrl = int(status.arm_status), int(status.ctrl_mode)
    teach = int(status.teach_status)
    manual = teach == 1 or arm_status == 0x0B or (ctrl in (2, 6) and not (ctrl == 2 and teach == 2 and arm_status == 0))
    if manual:
        reasons.append('请先用机身按钮退出示教')
    if arm_status not in (0, 1):
        reasons.append(f'控制器故障 0x{arm_status:02X}，不属于普通失能/急停待复位')
    for index in range(1, 7):
        if getattr(status.err_status, f'communication_status_joint_{index}', False):
            reasons.append(f'J{index} 电机通信异常')
    if np.max(np.abs(correction)) > MAX_CORRECTION:
        reasons.append('关节超出限位超过 3°，需要核对实际姿态与零位；不自动改写零位')
    summary = []
    if arm_status == 1:
        summary.append('控制器处于急停，需要显式复位')
    disabled = [str(d['joint']) for d in drivers if not d['enabled']]
    if disabled:
        summary.append('J'+ '/J'.join(disabled)+' 失能')
    if np.max(np.abs(correction)) > 1e-6:
        summary.append(f'反馈超出软件限位，最大 {np.degrees(np.max(np.abs(correction))):.2f}°')
    known = drivers[1]['code'] == 0xA4
    summary.append('J2 出现主臂曾见的 0xA4 故障，但仍需现场核对零位' if known else
                   '未出现主臂已知的 J2 0xA4 故障签名；当前反馈不能证明零位丢失')
    return dict(eligible=not reasons, reasons=reasons, summary='；'.join(summary),
                arm_status=arm_status, ctrl_mode=ctrl, teach_status=teach, drivers=drivers,
                joints_deg=np.degrees(joints).tolist(), target_deg=np.degrees(target).tolist(),
                correction_deg=np.degrees(correction).tolist(), known_leader_fault=known)


def restore_arm(arm, limits, check=lambda: None, timeout=6, *, support_confirmed=False):
    """Recheck, prime a nearby planned target, reset if needed, enable and verify.

    No homing, gripper motion, role switch, joint disable, error clearing, or
    zero calibration. Out-of-range feedback is corrected at most 3 degrees.
    """
    check()
    before = inspect_arm(arm, limits)
    if not before['eligible']:
        raise RuntimeError('不能自动恢复：'+'；'.join(before['reasons']))
    if before['arm_status'] == 1 and not support_confirmed:
        raise RuntimeError('急停复位时固件会短暂失能；请先托稳从臂并确认，再执行恢复')
    start = np.radians(before['joints_deg'])
    target = np.radians(before['target_deg'])
    previous_speed = getattr(getattr(arm, '_msg_mode', None), 'move_spd_rate_ctrl', 100)
    started = time.time()
    successful = False
    try:
        check()
        arm.set_speed_percent(5)
        # Replace any stale target before releasing controller emergency stop.
        arm.move_j(target.tolist())
        if before['arm_status'] == 1:
            check()
            reset_at = time.time()
            arm.reset()
            # This firmware reboots motor communication and drops enable on
            # reset. Wait through transient 0x05 instead of treating it as a
            # persistent fault, or trusting pre-reset enable feedback.
            reset_deadline, stable = time.monotonic()+3, 0
            while time.monotonic() < reset_deadline:
                check()
                message = arm.get_arm_status()
                ready = (message is not None and message.timestamp > reset_at
                         and -.02 <= time.time()-message.timestamp <= .5
                         and int(message.msg.arm_status) == 0)
                stable = stable+1 if ready else 0
                if stable >= 3:
                    after_reset = inspect_arm(arm, limits)
                    if not after_reset['eligible']:
                        raise RuntimeError('复位后仍有异常：'+'；'.join(after_reset['reasons']))
                    if np.max(np.abs(np.radians(after_reset['joints_deg'])-start)) > MAX_CORRECTION:
                        raise RuntimeError('复位期间姿态变化超过 3°，请托稳从臂后重新诊断')
                    break
                time.sleep(.05)
            else:
                raise RuntimeError('急停复位后 3 秒内未恢复正常反馈；从臂可能仍失能，请托稳并检查设备')
            check()
            arm.set_speed_percent(5)
            arm.move_j(target.tolist())
            started = time.time()
        deadline, settled = time.monotonic()+timeout, 0
        while time.monotonic() < deadline:
            check()
            actual = finite(fresh(arm.get_joint_angles(), '恢复关节'), (6,), '恢复关节')
            if np.max(np.abs(actual-start)) > MAX_CORRECTION+np.deg2rad(.5):
                arm.electronic_emergency_stop()
                raise RuntimeError('恢复运动超出 3.5° 验收范围，已发送该从臂急停')
            status = fresh(arm.get_arm_status(), '恢复控制器')
            if int(status.arm_status) not in (0, 1):
                raise RuntimeError(f'恢复中控制器报告故障 0x{int(status.arm_status):02X}')
            drivers = [arm.get_driver_states(i) for i in range(1, 7)]
            enabled = True
            for index, message in enumerate(drivers, 1):
                driver = fresh(message, f'恢复 J{index}')
                if any(getattr(driver.foc_status, key, False) for key in FAULTS):
                    raise RuntimeError(f'恢复中 J{index} 出现驱动故障，请检查设备')
                enabled &= bool(driver.foc_status.driver_enable_status) and message.timestamp > started
            ready = enabled and int(status.arm_status) == 0 and np.max(np.abs(actual-target)) < np.deg2rad(.5)
            settled = settled+1 if ready else 0
            if settled >= 5:
                after = inspect_arm(arm, limits)
                if after['eligible']:
                    successful = True
                    return dict(before=before, after=after, recovered=True,
                                detail='从臂已恢复使能并保持当前位置；未重新标零、未回零、未操作夹爪')
            if not enabled:
                check()
                arm.enable()
            elif int(status.arm_status) == 0 and not ready:
                # Some firmware discards position commands received while
                # disabled. Reissue the same bounded target after enable ACK.
                check()
                arm.move_j(target.tolist())
            time.sleep(.05)
        raise RuntimeError('恢复超时：使能、正常状态或到位反馈未满足，请查看诊断，不要反复标零')
    finally:
        # Leave the low speed in place on failure/cancellation.
        if successful:
            check()
            arm.set_speed_percent(previous_speed)
