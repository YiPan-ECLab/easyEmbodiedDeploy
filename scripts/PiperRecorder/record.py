#!/home/py/main/easyEmbodiedDeploy/thirdParty/.venvs/piper-home/bin/python
"""Record dual-Piper demonstrations with pyAgxArm, without normalization."""

import argparse
import math
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

RECORD_HZ = 30
MAX_TASK_GAP_S = 0.35
START_JOINT_DELTA_RAD = 0.01
START_GRIPPER_DELTA_M = 0.002
IMAGE_SIZE = (224, 224)
OUTPUT_DIR = Path("/home/py/main/easyEmbodiedDeploy/datasets/hdf5/撕下快递单并贴到桌子上的快递上_test")

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


def print_summary(
    output_dir: Path,
    record_hz: float = RECORD_HZ,
    max_task_gap_s: float = MAX_TASK_GAP_S,
    start_joint_delta_rad: float = START_JOINT_DELTA_RAD,
    start_gripper_delta_m: float = START_GRIPPER_DELTA_M,
) -> None:
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
        f"采集 {record_hz:g} Hz，RGB {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}，输出 {output_dir}"
    )
    print(f"任务阶段最大允许采样间隔: {max_task_gap_s:g} s（超出则本次录制不保存）")
    print(
        f"开始动作阈值: 任一关节 {start_joint_delta_rad:g} rad，"
        f"夹爪 {start_gripper_delta_m:g} m（阈值前静止帧丢弃）"
    )
    print("actions: 关节 rad、夹爪宽度 m，未归一化")


def _operator_started(
    samples: dict,
    baseline: dict,
    joint_delta_rad: float,
    gripper_delta_m: float,
) -> bool:
    for side, sample in samples.items():
        reference_joints, reference_width = baseline[side]
        if max(
            abs(current - reference)
            for current, reference in zip(sample.action.joints, reference_joints)
        ) >= joint_delta_rad:
            return True
        if abs(sample.action.gripper_width - reference_width) >= gripper_delta_m:
            return True
    return False


def _validate_options(
    hz: float,
    max_task_gap_s: float,
    start_joint_delta_rad: float,
    start_gripper_delta_m: float,
) -> None:
    values = (
        (hz, "采样频率"),
        (max_task_gap_s, "空档阈值"),
        (start_joint_delta_rad, "开始关节动作阈值"),
        (start_gripper_delta_m, "开始夹爪动作阈值"),
    )
    if any(not math.isfinite(value) or value <= 0 for value, _ in values):
        raise ValueError("采样频率、空档阈值和开始动作阈值必须是有限的正数")


def record_episode(
    robot: DualPiper,
    cameras: dict[str, OrbbecCamera],
    hz: float,
    *,
    max_task_gap_s: float = MAX_TASK_GAP_S,
    start_joint_delta_rad: float = START_JOINT_DELTA_RAD,
    start_gripper_delta_m: float = START_GRIPPER_DELTA_M,
) -> EpisodeData:
    _validate_options(
        hz, max_task_gap_s, start_joint_delta_rad, start_gripper_delta_m
    )
    robot.preposition_and_start()
    episode = EpisodeData()
    baseline_samples = robot.sample()
    baseline = {
        side: (sample.action.joints, sample.action.gripper_width)
        for side, sample in baseline_samples.items()
    }
    recording_started = False
    discarded_idle_frames = 0
    previous_timestamp: float | None = None
    previous_scheduled_at: float | None = None

    def capture(phase: int, scheduled_at: float) -> bool:
        nonlocal previous_timestamp, previous_scheduled_at
        nonlocal recording_started, discarded_idle_frames
        started_at = time.monotonic()
        camera_started_at = started_at
        frames = {name: cameras[name].read() for name in CAMERA_NAMES}
        camera_duration = time.monotonic() - camera_started_at
        robot_started_at = time.monotonic()
        samples = robot.sample()
        robot_duration = time.monotonic() - robot_started_at
        sample_diagnostics = getattr(robot, "sample_diagnostics", lambda: {})()
        captured_at = time.monotonic()
        if phase == TASK_PHASE and not recording_started:
            if _operator_started(
                samples, baseline, start_joint_delta_rad, start_gripper_delta_m
            ):
                recording_started = True
                print("\n检测到真人开始操作，正式记录 task 帧")
            else:
                discarded_idle_frames += 1
                return False
        if phase == HOME_PHASE and not recording_started:
            return False
        if phase == TASK_PHASE and previous_timestamp is not None:
            interval = captured_at - previous_timestamp
            if interval > max_task_gap_s:
                scheduled_interval = (
                    None
                    if previous_scheduled_at is None
                    else scheduled_at - previous_scheduled_at
                )
                print(
                    f"\n采样空档 {interval:.3f}s 超过 {max_task_gap_s:g}s，"
                    "本次录制已停止且不会保存。"
                )
                raise RuntimeError(
                    f"任务阶段采样空档过大: {interval:.3f}s > {max_task_gap_s:g}s; "
                    f"scheduled_gap={scheduled_interval!r}s; "
                    f"camera={camera_duration:.3f}s robot={robot_duration:.3f}s; "
                    f"robot_components={sample_diagnostics}"
                )
        episode.append(
            frames,
            samples,
            phase,
            timestamp=captured_at,
            scheduled_timestamp=scheduled_at,
            capture_duration_s=captured_at - started_at,
            camera_duration_s=camera_duration,
            robot_duration_s=robot_duration,
        )
        if phase == TASK_PHASE:
            previous_timestamp = captured_at
            previous_scheduled_at = scheduled_at
        return True

    clear_pending_input()
    print("录制中；请在 J6 预定位后开始操作，按回车停止...")
    terminal_bell()
    for scheduled_at in ticks(hz):
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            clear_pending_input()
            break
        captured = capture(TASK_PHASE, scheduled_at)
        if captured:
            print(f"\r已录制 {len(episode)} 帧", end="", flush=True)
        else:
            print(
                f"\r等待真人开始操作，已丢弃静止帧 {discarded_idle_frames} 帧",
                end="",
                flush=True,
            )

    print("\n四台机械臂回 ZERO，四个夹爪闭合到 0 m，并继续录制...")
    robot.home(
        on_sample=lambda: capture(HOME_PHASE, time.monotonic()), sample_hz=hz
    )
    if not any(phase == TASK_PHASE for phase in episode.phases):
        raise RuntimeError("没有录到任务阶段帧，不能保存")
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


def run(
    output_dir: Path,
    *,
    record_hz: float = RECORD_HZ,
    max_task_gap_s: float = MAX_TASK_GAP_S,
    start_joint_delta_rad: float = START_JOINT_DELTA_RAD,
    start_gripper_delta_m: float = START_GRIPPER_DELTA_M,
) -> None:
    _validate_options(
        record_hz, max_task_gap_s, start_joint_delta_rad, start_gripper_delta_m
    )
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

    print_summary(
        output_dir,
        record_hz,
        max_task_gap_s,
        start_joint_delta_rad,
        start_gripper_delta_m,
    )
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
            try:
                episode = record_episode(
                    robot,
                    cameras,
                    record_hz,
                    max_task_gap_s=max_task_gap_s,
                    start_joint_delta_rad=start_joint_delta_rad,
                    start_gripper_delta_m=start_gripper_delta_m,
                )
            except Exception as exc:  # noqa: BLE001 - keep the recording session alive
                print(f"本次录制失败: {type(exc).__name__}: {exc}", file=sys.stderr)
                print("机械臂重新回 ZERO 后继续下一条...")
                robot.home()
                continue

            while True:
                choice = prompt_once("保存本次录制？[回车/Y=保存，n=丢弃]: ").lower()
                if choice not in ("", "y", "yes"):
                    print(f"已丢弃 {len(episode)} 帧")
                    break
                try:
                    path = write_episode(
                        episode,
                        instruction,
                        output_dir,
                        pyagxarm_version=pyAgxArm.__version__,
                        firmware_profile=FIRMWARE,
                        firmware_version=FIRMWARE_VERSION,
                        gripper_force=GRIPPER_FORCE_N,
                        record_hz=record_hz,
                    )
                except Exception as exc:  # noqa: BLE001 - allow retry or discard
                    print(f"保存失败: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                print(f"已保存: {path} ({len(episode)} 帧)")
                break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="HDF5 输出目录")
    parser.add_argument(
        "--check", action="store_true", help="只检查依赖和 pyAgxArm API，不连接硬件"
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=RECORD_HZ,
        help=f"任务采样频率，默认 {RECORD_HZ:g} Hz",
    )
    parser.add_argument(
        "--max-task-gap-s", type=float, default=MAX_TASK_GAP_S,
        help="任务阶段允许的最大相邻采样间隔，超出则停止且不保存",
    )
    parser.add_argument(
        "--start-joint-delta-rad", type=float, default=START_JOINT_DELTA_RAD,
        help="开始正式记录所需的任一关节动作幅度，默认 0.01 rad",
    )
    parser.add_argument(
        "--start-gripper-delta-m", type=float, default=START_GRIPPER_DELTA_M,
        help="开始正式记录所需的夹爪动作幅度，默认 0.002 m",
    )
    args = parser.parse_args()
    try:
        if args.check:
            check_installation()
        else:
            run(
                args.output.expanduser().resolve(),
                record_hz=args.hz,
                max_task_gap_s=args.max_task_gap_s,
                start_joint_delta_rad=args.start_joint_delta_rad,
                start_gripper_delta_m=args.start_gripper_delta_m,
            )
    except KeyboardInterrupt:
        print("\n已停止")
    except Exception as exc:  # noqa: BLE001 - CLI boundary prints hardware errors
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
