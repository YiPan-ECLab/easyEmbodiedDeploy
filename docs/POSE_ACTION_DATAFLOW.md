# 位姿动作数据链路

## 输入与动作维度

所有模式复用同一个 **14 维 state**，来自当前帧的从臂反馈：

```text
[left_j1…j6, left_gripper, right_j1…j6, right_gripper]
关节 rad，夹爪 m。另输入三路当前图像和任务文字。
```

不把末端位姿、外参或未来目标拼入 state。不同表示能减少 state/action 直接数值复制的机会，不能保证模型不利用关节与位姿之间的相关性。

位姿 action 也是 **14 维**，按左、右手顺序，每只手：

```text
[x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper]
位置 m，旋转向量 rad，夹爪为未来帧的绝对宽度 m。
```

旋转向量是轴角，**不是原始 RPY，也不是 6D 旋转向量**。PI05 内部原有的 32 维 padding 不改变这两项 14 维物理数据契约；位姿训练仅监督有效的 14 个 action 轴。

## 三种位姿动作

| 网页 / CLI 模式 | 每步目标 | 标定 |
| --- | --- | --- |
| `absolute_base_pose` | 该手自身 base 下的未来法兰绝对 xyz / rotvec | 不需要 |
| `camera_delta_wrist` | 未来法兰相对当前法兰的运动，在该手当前腕相机方向中表达 | 左、右腕相机到各自法兰 |
| `camera_delta_global` | 左右未来法兰分别相对当前法兰的运动，在同一全局相机方向中表达 | 全局相机到左、右 base |

原始位姿来自 HDF5 的 `obs/left_ee_states`、`obs/right_ee_states`：各自 base 下的从臂实测法兰 `[xyz, roll,pitch,yaw]`，单位 m / rad。目标不是 TCP，不对主臂关节命令做猜测性 FK。夹爪来自对应未来帧的主臂宽度命令。

## camera delta 的明确计算

`T_A_B` 将 B 坐标映射到 A。B 为该手 base，F 为法兰，C 为所选相机。

```text
R_B_F = Rz(yaw) @ Ry(pitch) @ Rx(roll)

腕相机：T_B_C(t) = T_B_F(t) @ T_F_C
全局相机：T_B_C 为该手对应的固定标定结果

delta_p_C(t,k) = R_B_C(t).T @ [p_B_F(t+k) - p_B_F(t)]
delta_R_C(t,k) = R_B_C(t).T @ R_B_F(t+k) @ R_B_F(t).T @ R_B_C(t)
```

采用 [Qwen-RobotManip §3.3 公式 (5)](https://arxiv.org/html/2606.17846v2#S3.SS3) 的 camera delta 定义。整个预测段冻结当前 t 的相机方向；每个目标都相对于 t，**不逐帧累计 camera delta**。不直接相减欧拉角，也不使用包含相机平移杠杆项的 SE(3) 共轭公式。

旋转处理固定为：

```text
原始 RPY → 旋转矩阵 → 绝对 / 相对旋转矩阵
→ 前两列的 6D 表示 → Gram–Schmidt 还原 → 三维 rotvec
6D 中间顺序：[R00,R10,R20,R01,R11,R21]
```

这里对齐论文的动作几何和最终三维旋转向量；不移植其完整跨机器人 state 模板，也不声称与未公开的全部数据实现逐行一致。

## 采集、转换与训练

1. 按原有脚踏板 / Enter 流程采集，不改变 Recorder。
2. 转换只取连续 task 段，排除 home；普通录制按配置降采样，部分融合输入沿用保留帧，不再二次抽帧。关节模式沿用原裁尾规则；位姿模式只有主臂命令、从臂 state 和法兰反馈都停止变化才裁掉尾部，保留从臂收敛过程。
3. 当前帧 t 的图像和关节 state 对应处理后序列的未来 **t+1…t+50** 法兰目标，不跨 episode 或 home。支持部分融合后的非均匀采样；普通录制的异常时序需在转换页逐条检查并显式放行。训练按序列取 chunk，不要求物理采样间隔均匀，具体规则见[数据转换说明](../scripts/LerobotDataProcessor/README.md)。
4. episode 尾部不足 50 步时重复最后目标用于补齐形状，同时保存有效帧掩码；填充帧不参与 loss 和动作统计。RTC 的已知前缀也不参与 loss，末尾样本至少保留一个有效监督目标。
5. 两套训练器自动读取相同的位姿契约、外参快照和完整动作段统计。归一化方式仍可选 quantiles / mean_std，支持正常训练、trained RTC、LoRA 和完整参数训练配置。

数据文件分工：

| 文件 | 内容 |
| --- | --- |
| 原生 LeRobot parquet | 14D 当前关节 state、首个未来 action、帧 / episode 索引 |
| `meta/pose_action_contract.json` | 动作模式、单位、t+1 偏移、50 步锚点约定、标定快照 / SHA256、安装版本与源帧索引表 |
| `meta/pose/episode-*.npz` | 当前与未来原始法兰位姿、绝对夹爪目标、源帧号与实际采样时间；训练时生成完整 chunk |
| `meta/stats.json` | 原生 parquet 每行统计，保持 LeRobot 原有含义 |
| `meta/pose_action_stats.json` | 所有有效未来目标的完整 chunk 统计，用于位姿 action 归一化 |
| checkpoint `training_manifest.json` | 固化训练模式与标定契约；实际归一化参数随模型 processor / assets 保存 |

迁移数据集时必须复制**整个 bundle**，不能只复制 parquet 或 `lerobot/data`。普通原生训练入口不会自行理解位姿 sidecar，必须通过本项目训练模块启动。部署从检查点读取契约，不依赖原训练数据目录仍然存在。

## 部署反变换

模型 state 仍是 14D 关节反馈。硬件桥在位姿模式下复用同次采样中已有的法兰反馈，作为几何解码锚点，不额外输入模型。

```text
绝对 base：xyz + rotvec → 各自 base 下 xyz / RPY
camera delta：
  p_future = p_current + R_B_C(t) @ delta_p_C
  R_future = R_B_C(t) @ delta_R_C @ R_B_C(t).T @ R_current
```

整个 chunk 在推理时刻解码为绝对 base 目标，再进入同步 / RTC 队列。RTC 删掉过期步骤不会重新锚定剩余目标；下一次推理将剩余绝对目标重新编码到新的观测锚点，作为连续性前缀。

机械臂使用 SDK `move_p([xyz,RPY])`，夹爪使用绝对宽度接口。平滑在绝对位置和 SO(3) 旋转上执行。SDK 负责机械臂 IK；不自行替换关节模式的 SDK 控制逻辑。

**位姿初始关节偏移**默认 2°，可在部署页面或 `--pose-start-margin-deg` 设置 0…5°：完成原有 J6 ±45° 预定位后，新位姿分支把 J2 / J3 设为 +偏移 / −偏移。真机验证发现 J2=J3=0 的限位边界会让部分经过数值舍入的笛卡尔目标被 SDK 拒绝。0 表示关闭这项偏移。退出位姿部署时切回 SDK 关节回零流程。

camera delta 部署还需确认与训练一致的 `calibration_mount_id`，并核对真实相机序列号、传感器和采集分辨率。部署的 policy Hz 独立决定从动作队列取目标的频率，不与数据采集 FPS 绑定；融合数据可标记为 `fps: mixed`。外参错误无法靠模型补救，真实安装精度需按[标定说明](../scripts/CameraCalibration/README.md)采样验收。

原有 `absolute_joint` 仍预测绝对关节命令；`delta_joint` 仍预测相邻选帧命令之差，包括夹爪增量。相对关节现在先累计**完整预测段**再进入 RTC 队列，避免裁掉过期步骤时丢失位移；新训练屏蔽尾部填充帧。详见[数据转换说明](../scripts/LerobotDataProcessor/README.md)。
