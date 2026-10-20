"""Optional, offline singularity filtering; never bridge a removed interval."""

from functools import lru_cache
from pathlib import Path
import runpy

import numpy as np

DEFAULT_SINGULARITY_RATIO = .01
CHARACTERISTIC_LENGTH_M = .3


@lru_cache(maxsize=1)
def piper_mdh():
    # Reuse the SDK's table without importing CAN/camera packages into conversion.
    path = Path(__file__).resolve().parents[2] / "thirdParty/pyAgxArm/pyAgxArm/api/constants.py"
    return np.asarray(runpy.run_path(str(path))["ROBOT_MDH_PRESET"]["piper"], dtype=float)


def geometric_jacobian(joints):
    q = np.asarray(joints, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or not np.isfinite(q).all():
        raise ValueError("奇异检测需要有限的 [T,6] 关节角")
    transform = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    axes, origins = [], []
    for i, (d, a, alpha, offset) in enumerate(piper_mdh()):
        ca, sa = np.cos(alpha), np.sin(alpha)
        fixed = np.array([[1, 0, 0, a], [0, ca, -sa, 0], [0, sa, ca, 0], [0, 0, 0, 1]])
        base = transform @ fixed
        axes.append(base[:, :3, 2].copy())
        origins.append(base[:, :3, 3].copy())
        moving = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        c, s = np.cos(q[:, i]+offset), np.sin(q[:, i]+offset)
        moving[:, 0, 0] = moving[:, 1, 1] = c
        moving[:, 1, 0], moving[:, 0, 1] = s, -s
        moving[:, 2, 3] = d
        transform = base @ moving
    axes, origins = np.stack(axes, axis=1), np.stack(origins, axis=1)
    translation = np.cross(axes, transform[:, None, :3, 3]-origins)
    return np.concatenate((translation.transpose(0, 2, 1), axes.transpose(0, 2, 1)), axis=1)


def singularity_ratio(joints):
    jacobian = geometric_jacobian(joints)
    jacobian[:, :3] /= CHARACTERISTIC_LENGTH_M
    values = np.linalg.svd(jacobian, compute_uv=False)
    return values[:, -1] / np.maximum(values[:, 0], np.finfo(float).tiny)


def clean_pose_indices(states, selected, threshold=DEFAULT_SINGULARITY_RATIO):
    """Return contiguous safe runs in the existing sampling grid and an audit.

    Examine unsampled frames too: downsampling must not hide a removed region.
    Neither speed nor uniform sampling is used as a filtering condition.
    """
    if isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("奇异比值阈值须在 0 与 1 之间")
    states, selected = np.asarray(states), np.asarray(selected, dtype=np.int64)
    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError("奇异清洗需要 [T,14] 双臂关节状态")
    if len(selected) < 2 or np.any(np.diff(selected) <= 0):
        raise ValueError("奇异清洗需要至少两帧且选帧索引递增")
    begin, end = int(selected[0]), int(selected[-1])+1
    block = states[begin:end]
    left = singularity_ratio(block[:, :6])
    right = singularity_ratio(block[:, 7:13])
    bad = (left < threshold) | (right < threshold)
    positions = selected-begin
    safe_positions = np.flatnonzero(~bad[positions])
    prefix = np.r_[0, np.cumsum(bad)]
    cuts = [0]
    for i in range(1, len(safe_positions)):
        previous, current = safe_positions[i-1:i+1]
        if current != previous+1 or prefix[positions[current]+1] > prefix[positions[previous]+1]:
            cuts.append(i)
    cuts.append(len(safe_positions))
    spans, isolated = [], 0
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b-a < 2:
            isolated += b-a
        else:
            spans.append([int(safe_positions[a]), int(safe_positions[b-1])+1])
    retained = sum(b-a for a, b in spans)
    report = dict(enabled=True, model="piper", threshold=float(threshold),
                  characteristic_length_m=CHARACTERISTIC_LENGTH_M,
                  candidate_frames=len(selected),
                  removed_singular_frames=int(bad[positions].sum()),
                  left_singular_frames=int((left[positions] < threshold).sum()),
                  right_singular_frames=int((right[positions] < threshold).sum()),
                  isolated_frames=int(isolated), retained_source_frames=retained,
                  output_episodes=len(spans), output_frames=retained-len(spans),
                  selection_spans=spans)
    return [selected[a:b] for a, b in spans], report


def episode_sources(files):
    """Expand source files to output episodes; shared with source verification."""
    result = []
    for info in files:
        cleaning = info.get("pose_cleaning")
        if not cleaning:
            result.append(info)
            continue
        retained = info.get("retained_clean_segments")
        for number, (start, stop) in enumerate(cleaning["selection_spans"]):
            if retained is not None and number not in retained:
                continue
            result.append({**info, "clean_segment": [start, stop],
                           "clean_segment_number": number,
                           "sampled_frames": stop-start, "output_frames": stop-start-1})
    return result
