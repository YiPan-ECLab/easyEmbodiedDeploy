#!/home/py/main/easyEmbodiedDeploy/thirdParty/.venvs/piper-home/bin/python
"""Record dual-Piper demonstrations with pyAgxArm, without normalization."""

import argparse
import select
import sys
import termios
import time
from contextlib import ExitStack
from pathlib import Path

from pyAgxArm import PiperFW

from piper_recorder.camera import CAMERA_NAMES, OrbbecCamera, open_cameras
from piper_recorder.dataset import HOME_PHASE, TASK_PHASE, EpisodeData, write_episode
from piper_recorder.robot import DualPiper, ticks

RECORD_HZ = 10
IMAGE_SIZE = (224, 224)
OUTPUT_DIR = Path("/home/py/main/easyEmbodiedDeploy/datasets/hdf5/pyagxarm_recordings")

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
#   640x480   @ 5/10/15/30/60 FPS
#   848x480   @ 5/10/15/30/60 FPS
#   1280x720  @ 5/10/15/30 FPS
#   1280x800  @ 5/10/15/30 FPS


CAMERA_MAX_AGE_S = 0.5
CAMERA_SERIALS = {
    "agentview": "CP2AB530005P",
    "left_eye_in_hand": "CV2L360000KH",
    "right_eye_in_hand": "CV2L3600003K",
}

PIPERS = {
    "left": {
        "leader_name": "piper-lt",
        "follower_name": "piper-lo",
        "joint6_deg": -45,
    },
    "right": {
        "leader_name": "piper-rt",
        "follower_name": "piper-ro",
        "joint6_deg": 45,
    },
}
FIRMWARE = PiperFW.V189
FIRMWARE_VERSION = "S-V1.9-1"
HOME_SPEED = 30
HOME_TIMEOUT_S = 60.0
HOME_TOLERANCE_DEG = 1.0
HOME_LEADER_WAIT_S = 5.0
PREPOSITION_SPEED = 20
PREPOSITION_TIMEOUT_S = 15.0
ZERO_TOLERANCE_DEG = 1.0
FOLLOW_HZ = 200.0
STATUS_PERIOD_S = 0.5
GRIPPER_FORCE_N = 1.0
GRIPPER_MAX_WIDTH_M = 0.1
GRIPPER_HOME_TIMEOUT_S = 5.0
GRIPPER_HOME_TOLERANCE_M = 0.002


def terminal_bell() -> None:
    print("\a", end="", flush=True)
    time.sleep(0.15)
    print("\a", end="", flush=True)


def clear_pending_input() -> None:
    if sys.stdin.isatty():
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except (OSError, termios.error):
            pass


def prompt_once(message: str) -> str:
    clear_pending_input()
    terminal_bell()
    answer = input(message).strip()
    clear_pending_input()
    return answer


def print_summary(output_dir: Path) -> None:
    print(f"pyAgxArm 固件配置: {FIRMWARE}")
    print("机械臂映射:")
    for side, pair in PIPERS.items():
        print(
            f"  {side}: {pair['leader_name']} -> {pair['follower_name']}，"
            f"J6={pair['joint6_deg']:+g} deg"
        )
    print("相机映射:")
    for name, serial in CAMERA_SERIALS.items():
        print(f"  {name}: {serial}")
    print(
        f"采集 {RECORD_HZ:g} Hz，RGB {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}，输出 {output_dir}"
    )
    print("actions: 关节 rad、夹爪宽度 m，未归一化")


def record_episode(
    robot: DualPiper, cameras: dict[str, OrbbecCamera], hz: float
) -> EpisodeData:
    robot.preposition_and_start()
    episode = EpisodeData()

    def capture(phase: int) -> None:
        frames = {name: cameras[name].read() for name in CAMERA_NAMES}
        episode.append(frames, robot.sample(), phase)

    clear_pending_input()
    print("录制中，按回车停止...")
    terminal_bell()
    for _ in ticks(hz):
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            clear_pending_input()
            break
        capture(TASK_PHASE)
        print(f"\r已录制 {len(episode)} 帧", end="", flush=True)

    print("\n四台机械臂回 ZERO，四个夹爪闭合到 0 m，并继续录制...")
    robot.home(on_sample=lambda: capture(HOME_PHASE), sample_hz=hz)
    return episode


def check_installation() -> None:
    import cv2
    import h5py
    import numpy
    import pyAgxArm
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
    from pyorbbecsdk import Context

    config = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=FIRMWARE,
        channel="piper-format-check",
    )
    arm = AgxArmFactory.create_arm(config)
    gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
    assert callable(arm.move_j) and callable(arm.get_joint_angles)
    assert callable(gripper.move_gripper_m) and callable(gripper.get_gripper_status)
    assert Context is not None
    print(f"pyAgxArm {pyAgxArm.__version__}: move_j/get_joint_angles = 6 x rad")
    print("AgxGripper: move_gripper_m/get_gripper_status(width) = m; force = N")
    print(
        f"numpy {numpy.__version__}, h5py {h5py.__version__}, "
        f"opencv {cv2.__version__}, pyorbbecsdk 已加载"
    )
    print("静态检查通过；未连接相机或机械臂")


def run(output_dir: Path) -> None:
    aliases = [
        pair[key]
        for pair in PIPERS.values()
        for key in ("leader_name", "follower_name")
    ]
    if len(set(aliases)) != 4:
        raise ValueError("四台机械臂的 CAN 名称必须互不相同")
    if len(set(CAMERA_SERIALS.values())) != 3:
        raise ValueError("三个相机序列号必须互不相同")

    import pyAgxArm

    print_summary(output_dir)
    with ExitStack() as resources:
        cameras = open_cameras(
            CAMERA_SERIALS,
            resources,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            image_size=IMAGE_SIZE,
            max_age_s=CAMERA_MAX_AGE_S,
        )
        robot = DualPiper(
            PIPERS,
            firmware=FIRMWARE,
            home_speed=HOME_SPEED,
            home_timeout_s=HOME_TIMEOUT_S,
            home_tolerance_deg=HOME_TOLERANCE_DEG,
            home_leader_wait_s=HOME_LEADER_WAIT_S,
            preposition_speed=PREPOSITION_SPEED,
            preposition_timeout_s=PREPOSITION_TIMEOUT_S,
            zero_tolerance_deg=ZERO_TOLERANCE_DEG,
            follow_hz=FOLLOW_HZ,
            status_period_s=STATUS_PERIOD_S,
            gripper_force=GRIPPER_FORCE_N,
            gripper_max_width=GRIPPER_MAX_WIDTH_M,
            gripper_home_timeout_s=GRIPPER_HOME_TIMEOUT_S,
            gripper_home_tolerance_m=GRIPPER_HOME_TOLERANCE_M,
        )
        resources.callback(robot.close)
        robot.connect()
        print("四台机械臂回 ZERO，四个夹爪闭合到 0 m...")
        robot.home()

        instruction = ""
        while True:
            entered = prompt_once("\n任务指令（可留空，回车沿用，:q 退出）: ")
            if entered == ":q":
                return
            instruction = entered or instruction
            print(f"任务: {instruction or '（空）'}")
            episode = record_episode(robot, cameras, RECORD_HZ)
            choice = prompt_once("保存本次录制？[回车/Y=保存，n=丢弃]: ").lower()
            if choice in ("", "y", "yes"):
                path = write_episode(
                    episode,
                    instruction,
                    output_dir,
                    pyagxarm_version=pyAgxArm.__version__,
                    firmware_profile=FIRMWARE,
                    firmware_version=FIRMWARE_VERSION,
                    gripper_force=GRIPPER_FORCE_N,
                )
                print(f"已保存: {path} ({len(episode)} 帧)")
            else:
                print(f"已丢弃 {len(episode)} 帧")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="HDF5 输出目录")
    parser.add_argument(
        "--check", action="store_true", help="只检查依赖和 pyAgxArm API，不连接硬件"
    )
    args = parser.parse_args()
    try:
        if args.check:
            check_installation()
        else:
            run(args.output.expanduser().resolve())
    except KeyboardInterrupt:
        print("\n已停止")
    except Exception as exc:  # noqa: BLE001 - CLI boundary prints hardware errors
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
