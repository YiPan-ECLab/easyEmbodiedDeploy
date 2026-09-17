"""Bridge HDF5 pose recordings and portable, independently checkable sidecars."""

import numpy as np

from .pose_actions import PoseCodec, flange_matrix
from .pose_dataset import PoseDataset, training_stats, write_episode


def read_raw_poses(demo):
    values = []
    count = len(demo["states"])
    for side in ("left", "right"):
        key = f"obs/{side}_ee_states"
        if key not in demo:
            raise ValueError(f"位姿转换缺少 {key}")
        field = demo[key]
        units = field.attrs.get("units", "")
        if isinstance(units, bytes):
            units = units.decode()
        if units != "m,m,m,rad,rad,rad":
            raise ValueError(f"{key} 须明确声明 m,m,m,rad,rad,rad，不能猜测位姿单位")
        array = np.asarray(field, dtype=np.float64)
        if array.shape != (count, 6) or not np.isfinite(array).all():
            raise ValueError(f"{key} 必须为有限 [T,6] xyz/RPY")
        values.append(array)
    return np.concatenate(values, axis=1)


def convert_episode(root, contract, demo, selected, source):
    poses = read_raw_poses(demo)[selected]
    grips = np.asarray(demo["actions"], dtype=np.float64)[selected][:, [6, 13]]
    times = np.asarray(demo["timestamps"], dtype=np.float64)[selected]
    if not np.all(np.asarray(demo["trajectory_phase"])[selected] == 0):
        raise ValueError("位姿未来目标不能包含 home 帧")
    write_episode(root, contract, poses, grips, selected, times, source)
    codec = PoseCodec(contract["action_mode"], contract["calibration"])
    return np.asarray([codec.encode(poses[i], poses[i+1], grips[i+1]) for i in range(len(poses)-1)], dtype=np.float32)


def validate_pose_bundle(root, columns):
    errors, metrics = [], {}
    try:
        dataset = PoseDataset(root)
        if dataset.rows != len(columns["action"]):
            raise ValueError("位姿 sidecar 与 LeRobot 行数不一致")
        training_stats(root, {})
        largest = 0.0
        for index, episode in enumerate(dataset.episodes):
            arrays = dataset.episode(index)
            row_start, rows = episode["row_start"], episode["rows"]
            ids = slice(row_start, row_start+rows)
            if not np.all(np.asarray(columns["episode_index"])[ids] == index):
                raise ValueError("位姿 sidecar 的 episode 与 parquet 不一致")
            dt = np.diff(arrays["timestamps"])
            period = 1/dataset.contract["fps"]
            if np.any((dt < period/1.5) | (dt > period*1.5)):
                raise ValueError("位姿未来目标跨越异常实际采样间隔")
            codec = dataset.codec
            for local in range(rows):
                expected = codec.encode(arrays["poses"][local], arrays["poses"][local+1], arrays["grippers"][local+1])
                actual = np.asarray(columns["action"][row_start+local])
                largest = max(largest, float(np.max(np.abs(expected-actual))))
                decoded, grip = codec.decode(arrays["poses"][local], actual)
                for offset in (0, 6):
                    if not np.allclose(flange_matrix(decoded[offset:offset+6]), flange_matrix(arrays["poses"][local+1, offset:offset+6]), atol=2e-6, rtol=0):
                        raise ValueError(f"episode {index} 第 {local} 帧位姿标签不能还原未来目标")
                if not np.allclose(grip, arrays["grippers"][local+1], atol=1e-7, rtol=0):
                    raise ValueError("位姿夹爪标签不是未来帧绝对宽度")
        if largest > 2e-6:
            errors.append("位姿 action 与 sidecar 几何语义不一致")
        metrics.update(action_mode=dataset.contract["action_mode"], target_offset=1,
                       horizon=50, max_first_target_error=largest,
                       calibration_sha256=dataset.contract["calibration_sha256"])
    except (ValueError, KeyError, OSError, IndexError) as exc:
        errors.append(str(exc))
    return dict(errors=errors, metrics=metrics)
