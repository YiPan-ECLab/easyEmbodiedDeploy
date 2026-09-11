"""Check a Piper leader gripper teaching pendant without moving the robot."""

import argparse
import time

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="piper-rt", choices=("piper-lt", "piper-rt"))
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=10.0,
        help="等待真人操作示教器并显示 0x159 控制帧的秒数",
    )
    args = parser.parse_args()

    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.V189,
        channel=args.channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    robot.connect()
    try:
        robot.set_leader_mode()
        robot.restore_leader_drag_mode()
        param = gripper.get_gripper_teaching_pendant_param(
            timeout=2.0, min_interval=0.0
        )
        if param is None:
            raise RuntimeError(f"{args.channel} 夹爪示教器参数查询超时")
        print(
            f"{args.channel} 夹爪示教器连接正常: "
            f"range={param.msg.teaching_range_per}%, "
            f"max={param.msg.max_range_config} m, "
            f"friction={param.msg.teaching_friction}"
        )

        print(f"请在 {args.watch_seconds:g} 秒内操作夹爪示教器以检查 0x159...")
        deadline = time.monotonic() + args.watch_seconds
        last_timestamp = 0.0
        received = 0
        while time.monotonic() < deadline:
            control = gripper.get_gripper_ctrl_states()
            if control is not None and control.timestamp > last_timestamp:
                last_timestamp = control.timestamp
                received += 1
                print(
                    f"value={control.msg.value:.6f} m, "
                    f"force={control.msg.force:.3f} N, "
                    f"status_code={control.msg.status_code}"
                )
            time.sleep(0.01)
        if received:
            print(f"收到 {received} 帧新的 0x159 示教控制数据")
        else:
            print("未检测到操作；参数查询已确认示教器连接正常")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
