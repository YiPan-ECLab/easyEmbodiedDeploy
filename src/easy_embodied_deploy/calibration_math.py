"""Camera calibration geometry. Transforms T_A_B map coordinates B -> A.

This module does not convert training actions or control a robot.
"""

import math

import cv2
import numpy as np

DEFAULT_BOARD = dict(
    kind="chessboard",
    columns=4,
    rows=6,
    square_mm=53.5,
    marker_ratio=0.7,
    dictionary="DICT_4X4_50",
)
JOBS = {
    "left_wrist": dict(
        camera="left_eye_in_hand",
        side="left",
        mounting="wrist",
        transform="T_left_flange_left_camera",
    ),
    "right_wrist": dict(
        camera="right_eye_in_hand",
        side="right",
        mounting="wrist",
        transform="T_right_flange_right_camera",
    ),
    "global_left": dict(
        camera="agentview",
        side="left",
        mounting="fixed",
        transform="T_left_base_global_camera",
    ),
    "global_right": dict(
        camera="agentview",
        side="right",
        mounting="fixed",
        transform="T_right_base_global_camera",
    ),
    "intrinsics": dict(camera=None, side=None, mounting=None, transform=None),
}


def board_config(value):
    board = {**DEFAULT_BOARD, **value}
    if board["kind"] not in ("chessboard", "charuco"):
        raise ValueError("标定板类型必须是 chessboard 或 charuco")
    for key in ("columns", "rows"):
        n = board[key]
        if (
            isinstance(n, bool)
            or not math.isfinite(float(n))
            or not 3 <= float(n) <= 30
            or int(n) != float(n)
        ):
            raise ValueError("行列数须为 3 到 30 的整数")
        board[key] = int(n)
    board["square_mm"] = float(board["square_mm"])
    board["marker_ratio"] = float(board["marker_ratio"])
    if not math.isfinite(board["square_mm"]) or not 1 <= board["square_mm"] <= 300:
        raise ValueError("单格边长须为 1 到 300 mm")
    if not 0.2 <= board["marker_ratio"] <= 0.9:
        raise ValueError("ChArUco 标记边长比例须为 0.2 到 0.9")
    if board["dictionary"] not in ("DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250"):
        raise ValueError("不支持的 ArUco 字典")
    if board["kind"] == "charuco":
        charuco_board(board)  # validate dictionary capacity too
    return {key: board[key] for key in DEFAULT_BOARD}


def charuco_board(board):
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, board["dictionary"])
    )
    if board["columns"] * board["rows"] // 2 > len(dictionary.bytesList):
        raise ValueError("标记数量超出所选字典，请减少格子数或换用更大的字典")
    square = board["square_mm"] / 1000
    return cv2.aruco.CharucoBoard(
        (board["columns"], board["rows"]),
        square,
        square * board["marker_ratio"],
        dictionary,
    )


def board_image(board, square_px=100):
    columns, rows = board["columns"], board["rows"]
    if board["kind"] == "charuco":
        return charuco_board(board).generateImage(
            (columns * square_px, rows * square_px), marginSize=0, borderBits=1
        )
    columns, rows = columns + 1, rows + 1
    return np.kron(
        (np.indices((rows, columns)).sum(axis=0) % 2 * 255).astype(np.uint8),
        np.ones((square_px, square_px), np.uint8),
    )


def detect(rgb, board, reverse=False, rotation_quarters=0):
    if type(rotation_quarters) is not int or rotation_quarters not in (0, 1, 2, 3):
        raise ValueError("角点旋转须为 0、1、2 或 3 个 90°")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if board["kind"] == "chessboard":
        shape = (board["columns"], board["rows"])
        ok, corners = cv2.findChessboardCornersSB(
            gray, shape, cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ok:
            raise ValueError(
                "未检测到完整棋盘格：检查内角点行列数、遮挡、清晰度及板面大小"
            )
        if rotation_quarters % 2 and shape[0] != shape[1]:
            raise ValueError("只有方形棋盘可额外重排 90° / 270° 角点")
        if rotation_quarters == 2:
            corners = corners[::-1].copy()
        elif rotation_quarters:
            corners = (
                np.rot90(corners.reshape(shape[1], shape[0], 2), rotation_quarters)
                .reshape(-1, 1, 2)
                .copy()
            )
        if reverse:
            corners = corners[::-1].copy()
        objects = np.zeros((shape[0] * shape[1], 3), np.float32)
        objects[:, :2] = np.mgrid[0 : shape[0], 0 : shape[1]].T.reshape(-1, 2) * (
            board["square_mm"] / 1000
        )
        ids = list(range(len(objects)))
    else:
        target = charuco_board(board)
        corners, corner_ids, _, _ = cv2.aruco.CharucoDetector(target).detectBoard(gray)
        if corners is None or len(corners) < 8:
            raise ValueError("ChArUco 至少需要 8 个可见内角点，请调整板面位置")
        ids = corner_ids.ravel().tolist()
        objects = target.getChessboardCorners()[ids]
    if np.linalg.matrix_rank(objects[:, :2] - objects[:, :2].mean(axis=0)) < 2:
        raise ValueError("可见角点共线，请露出更多行列的角点")
    image_points = corners.reshape(-1, 2)
    overlay = rgb.copy()
    for i, (x, y) in enumerate(image_points):
        cv2.circle(overlay, (round(float(x)), round(float(y))), 4, (30, 220, 100), 1)
        if i == 0 or i == board["columns"] - 1:
            cv2.putText(
                overlay,
                str(ids[i]),
                (round(float(x)) + 6, round(float(y)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 70, 50),
                2,
            )
    return objects, image_points, ids, overlay


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def checked_transform(value):
    array = np.asarray(value, dtype=float)
    if (
        array.shape != (4, 4)
        or not np.isfinite(array).all()
        or not np.allclose(array[3], [0, 0, 0, 1], atol=1e-7)
    ):
        raise ValueError("位姿须为有限的 4×4 齐次矩阵")
    rotation = array[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-5
    ):
        raise ValueError("无效旋转矩阵")
    return array


def inverse(value):
    value = checked_transform(value)
    rotation = value[:3, :3].T
    return transform(rotation, -rotation @ value[:3, 3])


def flange_transform(pose):
    """pyAgxArm get_flange_pose: meters, fixed-axis XYZ Euler radians."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("法兰位姿须为 xyz(m) + roll/pitch/yaw(rad)，共 6 个有限数")
    roll, pitch, yaw = pose[3:]
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return transform(rotation, pose[:3])


def intrinsics(value):
    matrix = np.asarray(value["K"], dtype=float)
    distortion = np.asarray(value["D"], dtype=float)
    model = value.get("model", "brown")
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or min(matrix[0, 0], matrix[1, 1]) <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
    ):
        raise ValueError("无效相机内参矩阵")
    if (
        not np.isfinite(distortion).all()
        or distortion.ndim != 1
        or len(distortion) not in (4, 5, 8, 12, 14)
    ):
        raise ValueError("无效畸变参数")
    if model not in ("brown", "fisheye"):
        raise ValueError("该 SDK 畸变模型不能直接用于 OpenCV，请先采集图像重新标定内参")
    if model == "fisheye" and len(distortion) != 4:
        raise ValueError("鱼眼畸变需要 4 个参数")
    return matrix, distortion, model


def board_pose(objects, pixels, camera):
    matrix, distortion, model = intrinsics(camera)
    objects, pixels = (
        np.asarray(objects, np.float64),
        np.asarray(pixels, np.float64).reshape(-1, 2),
    )
    if (
        objects.shape != (len(pixels), 3)
        or len(pixels) < 6
        or not np.isfinite(objects).all()
        or not np.isfinite(pixels).all()
    ):
        raise ValueError("角点数据无效")
    if model == "fisheye":
        points = cv2.fisheye.undistortPoints(
            pixels.reshape(-1, 1, 2), matrix, distortion, P=matrix
        )
        ok, rvec, tvec = cv2.solvePnP(objects, points, matrix, None)
        projected, _ = cv2.fisheye.projectPoints(
            objects.reshape(-1, 1, 3), rvec, tvec, matrix, distortion
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(objects, pixels, matrix, distortion)
        projected, _ = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)
    if not ok or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        raise ValueError("棋盘格 PnP 未得到有效正深度解")
    rotation = cv2.Rodrigues(rvec)[0]
    rms = float(
        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1)))
    )
    return transform(rotation, tvec), rms


def solve_intrinsics(samples, width, height):
    if len(samples) < 8:
        raise ValueError("至少需要 8 张不同角度、位置的标定板图像，建议 15–25 张")
    objects = [np.asarray(s["objects"], np.float32) for s in samples]
    pixels = [np.asarray(s["pixels"], np.float32) for s in samples]
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        objects, pixels, (width, height), None, None
    )
    result = dict(
        K=matrix.tolist(),
        D=distortion.ravel().tolist(),
        model="brown",
        source="opencv_calibrateCamera",
        width=width,
        height=height,
    )
    intrinsics(result)
    errors = [board_pose(o, p, result)[1] for o, p in zip(objects, pixels)]
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rvecs])
    angular_spread = float(
        np.degrees(np.max(np.arccos(np.clip(normals @ normals.T, -1, 1))))
    )
    centers = np.array([np.mean(p, axis=0) for p in pixels])
    coverage = np.ptp(centers, axis=0) / [width, height]
    warnings = []
    if angular_spread < 15:
        warnings.append("板面倾斜变化不足 15°，请增加不同倾角的图像")
    if min(coverage) < 0.15:
        warnings.append("标定板中心覆盖范围较小，请覆盖图像四周")
    if rms > 1 or max(errors) > 2:
        warnings.append("重投影误差偏大，请检查模糊图像及标定板平整度")
    return dict(
        intrinsics=result,
        rms_px=float(rms),
        per_sample_rms_px=errors,
        normal_spread_deg=angular_spread,
        center_coverage=coverage.tolist(),
        warnings=warnings,
    )


def solve_extrinsics(samples, camera, mounting):
    if mounting not in ("wrist", "fixed"):
        raise ValueError("未知安装方式")
    if len(samples) < 8:
        raise ValueError("至少需要 8 个静止姿态，建议 15–25 个，绕两个以上不同方向旋转")
    poses = [checked_transform(s["T_base_flange"]) for s in samples]
    motions = np.array(
        [cv2.Rodrigues(p[:3, :3] @ poses[0][:3, :3].T)[0].ravel() for p in poses[1:]]
    )
    singular = np.linalg.svd(motions, compute_uv=False)
    if singular[1] < 0.08:
        raise ValueError("机械臂旋转变化不足或集中在同一轴，无法可靠求解手眼关系")
    targets_errors = [board_pose(s["objects"], s["pixels"], camera) for s in samples]
    targets, errors = zip(*targets_errors)
    inputs = poses if mounting == "wrist" else [inverse(p) for p in poses]
    rotation, translation = cv2.calibrateHandEye(
        [p[:3, :3] for p in inputs],
        [p[:3, 3] for p in inputs],
        [p[:3, :3] for p in targets],
        [p[:3, 3] for p in targets],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    solution = checked_transform(transform(rotation, translation))
    constants = [p @ solution @ target for p, target in zip(inputs, targets)]
    positions = np.array([p[:3, 3] for p in constants])
    center = positions.mean(axis=0)
    u, _, vt = np.linalg.svd(sum(p[:3, :3] for p in constants))
    mean_rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    angle_errors = np.array(
        [
            np.linalg.norm(cv2.Rodrigues(p[:3, :3] @ mean_rotation.T)[0])
            for p in constants
        ]
    )
    distances = np.linalg.norm(positions - center, axis=1) * 1000
    angles = np.degrees(angle_errors)
    warnings = []
    if max(errors) > 2:
        warnings.append("部分图像重投影误差超过 2 px，请检查角点和内参")
    if np.sqrt(np.mean(distances**2)) > 5 or np.sqrt(np.mean(angles**2)) > 2:
        warnings.append(
            "静态关系残差偏大，请检查板面原点、固定方式、位姿与图像对应关系"
        )
    return dict(
        matrix=solution.tolist(),
        units="m",
        rotation="matrix_rows",
        method="OpenCV PARK",
        sample_count=len(samples),
        per_sample_rms_px=list(errors),
        translation_rms_mm=float(np.sqrt(np.mean(distances**2))),
        rotation_rms_deg=float(np.sqrt(np.mean(angles**2))),
        per_sample_translation_mm=distances.tolist(),
        per_sample_rotation_deg=angles.tolist(),
        rotation_excitation_singular_values=singular.tolist(),
        warnings=warnings,
    )
