# 位姿动作数据流方案（待审阅，尚未接入训练）

本文件供确认下一阶段实现。当前提交仅新增目录管理、相机标定与导出，以及 OpenPI 运行指标修复。采集、数据转换、训练和机械臂执行仍使用原有已验证的关节动作链路。

已确认的两种模式：

| 模式 | 左臂参考系 | 右臂参考系 | 是否依赖相机标定 |
| --- | --- | --- | --- |
| `absolute_base_pose` | 左臂自身 base | 右臂自身 base | **不需要**，即使使用图像作观测也不依赖相机外参 |
| `camera_delta_pose` + `own_wrist` | 当前左腕相机 | 当前右腕相机 | 需要两项腕部外参 |
| `camera_delta_pose` + `global` | 全局相机，相对左 base 求变换 | 同一全局相机，相对右 base 求变换 | 需要全局相机对两侧 base 的外参 |

绝对模式不应因为没有标定文件而被数据处理、训练配置或部署配置阻止。两侧位姿直接用各自 base 表达，不转换到公共世界系，也不要求两个 base 之间的标定。

## 1. 现有原始数据究竟是什么

`scripts/PiperRecorder/piper_recorder/dataset.py` 中：

- `data/demo_0/obs/left_ee_states` 与 `right_ee_states`：各侧**从臂反馈**的法兰位姿，6 维，`[x,y,z,roll,pitch,yaw]`，单位 `[m,m,m,rad,rad,rad]`。
- `obs/ee_states`：同样的左右法兰反馈拼接，12 维。
- `data/demo_0/actions`：原有 14 维主臂关节目标和夹爪目标，左右各 `[j1..j6, gripper_width]`。
- `data/demo_0/states`：从臂关节状态和夹爪宽度，14 维。
- 原始 episode 还保存三路图像、时间戳和 `trajectory_phase`，后者区分 task 与 home。

**`ee_states` 不能直接被宣称为原 `actions` 的笛卡尔等价物。** 它是从臂实际测量轨迹；原 `actions` 是主臂下发的关节目标。跟踪延迟和误差会使二者不同。

建议旧数据的首版位姿模式采用明确命名的 `pose_label_source=follower_measured_future`：以从臂未来实际到达的法兰位姿作为监督目标。这在两种位姿模式中保持一致。夹爪暂沿用原动作中的绝对宽度目标并明确标注来源。

如果要求严格保持原主臂目标的意图，应另设 `leader_command_fk`：使用经过验证的**从臂运动学模型**对原主臂关节目标做 FK，得到对应从臂 base 下的目标法兰位姿；不能从反馈位姿假装还原目标。这需要下一阶段单独验证，不在本轮修改原采集器。

这是本方案需要审阅的首要选择：测量轨迹监督与命令轨迹监督含义不同，不能静默替换。

## 2. 坐标、单位和旋转约定

记 `T_A_B` 为把 B 坐标变到 A 的齐次变换，采用列向量：

```text
p_A = R_A_B p_B + t_A_B
T_A_C = T_A_B T_B_C
T_B_A = inverse(T_A_B)
```

`B_L/B_R` 为左右 base；`F_L/F_R` 为左右法兰；`C_L/C_R` 为腕部 RGB 相机；`C_G` 为全局 RGB 相机。

Piper SDK 的 RPY 是**固定轴 XYZ 欧拉角**，不是旋转向量：

```text
R_base_flange = Rz(yaw) @ Ry(pitch) @ Rx(roll)
T_base_flange = [[R_base_flange, xyz], [0,0,0,1]]
```

整个转换先将欧拉角转为旋转矩阵，再计算相对旋转。不能逐元素相减欧拉角；也不能直接将 RPY 三个数当作 rotvec。

当前 SDK 返回的是法兰，未验证夹爪尖端或物体操作点的 TCP 偏置。首版 `end_effector_frame=flange`。如果以后用 TCP，需显式提供并验证 `T_flange_tcp`，再算 `T_base_tcp=T_base_flange@T_flange_tcp`。TCP 标定与相机标定是不同问题；当前绝对法兰位姿模式不需要它们。

RGB 相机使用与 OpenCV 内参 / PnP 一致的光学坐标：X 向图像右、Y 向图像下、Z 向前。外参矩阵 JSON 按行保存，平移为 m。不得把 SDK 的深度→RGB 内部外参当成机器人外参。

## 3. 时间选择与分段

建议固定以下契约，再生成两种标签：

1. 验证原始物理量、相机身份、位姿字段及时间戳。旧记录缺失安装信息时，由操作者确认对应标定，不能自动套用当前安装的标定。
2. 按现有规则选择 task、处理尾部静止和 home，并按目标 FPS 选择同一组时间索引；图像、法兰反馈、夹爪和其他观测用同一映射。
3. **先筛选 / 重采样，再计算位姿 action。** 不跨 episode、task/home 边界或异常时间缺口做相对变换。
4. 在锚点 `t`，观测来自当前帧；未来动作 chunk 的 `k=1..H` 对应选择后的 `t+k` 帧目标。使用未来测量位姿时，首步不能是 `t→t` 的全零动作。
5. 不足 H 步的尾部采用明确的 padding 与有效 mask；统计量只使用真实有效目标。首版可重复最后有效目标，但训练损失仍屏蔽填充位置。

现有关节模式的时序和已验证动作逻辑继续保持原样。位姿模式的 `k=1` 是对“未来实测轨迹标签”的设计选择，不能借此改动原关节模式的索引。

## 4. Absolute base pose：无相机标定链路

每侧用它自己的 base：

```text
左目标 = T_B_L_F_L(t+k)
右目标 = T_B_R_F_R(t+k)
```

建议训练 action 为每侧 `[absolute_xyz(3), absolute_rotvec(3), gripper_width(1)]`，左右合计 14 维。旋转向量是**绝对旋转矩阵**的对数映射；位置不做差，旋转也不做相对变换。这个旋转表示建议与 camera delta 共用解码工具，但**需要审阅确认**。

若希望“直接预测保存的 Euler RPY”，也可以另设 `absolute_rotation_representation=euler_xyz`，仍为 14 维且不需要相机标定；必须独立命名并训练，处理 ±π 周期边界，不能与 rotvec checkpoint 混用。无论选择哪种数值表示，绝对参考系都固定为左右自身 base。

建议状态用每側 `[xyz(3), rotation6d(6), gripper_width(1)]`，共 20 维。使用相机图像作为模型输入不会改变这些几何参考系。

## 5. Camera delta：严格采用 Qwen-RobotManip 公式 (5)

依据 [Qwen-RobotManip 技术报告 §3.2–3.3](https://arxiv.org/html/2606.17846v1) 与 [官方仓库](https://github.com/QwenLM/Qwen-RobotManip)。公开仓库提供报告及介绍；当前可核对的几何依据是论文，不声称已复用未公开的训练实现。

报告中位姿状态用 9 维（位置 3 + rotation6D 6），动作旋转用 **3 维旋转向量**。相机视角动作采用公式 (5)：

```text
R_delta_C = R_C_E · R_E_E* · R_E_C
delta_p_C = R_C_E · p_E_E*
```

展开成可直接用 Piper 数据实现的形式。以下每个 B 都是该侧自己的 base，E 暂为法兰 F：

```text
R_delta_B(t,k) = R_B_F(t+k) · transpose(R_B_F(t))
delta_p_B(t,k) = p_B_F(t+k) - p_B_F(t)

R_delta_C(t,k) = transpose(R_B_C(t)) · R_delta_B(t,k) · R_B_C(t)
delta_p_C(t,k) = transpose(R_B_C(t)) · delta_p_B(t,k)
```

使用 `T_C_F(t+k) @ inverse(T_C_F(t))` 取平移是另一种不同定义，对应报告讨论的公式 (6)，会把相机位置和旋转耦合到平移增量；**本方案不采用它**。

### 当前腕部视角

```text
T_B_L_C_L(t) = T_B_L_F_L(t) @ T_F_L_C_L
T_B_R_C_R(t) = T_B_R_F_R(t) @ T_F_R_C_R
```

左动作在当前左腕相机中表达，右动作在当前右腕相机中表达。

对于一整个未来 chunk，必须冻结观测时刻 **t** 的相机参考系。不能给每个未来步使用 `T_B_C(t+k)` 重新跟随腕部，否则相机和法兰的共同运动可能抵消目标动作。

### 全局视角

```text
左臂使用标定 T_B_L_C_G
右臂使用标定 T_B_R_C_G
```

两个结果将左右动作表达在同一台全局相机的光学方向。两个 base 的平移原点不同，并不影响正确使用各自外参。

公式 (5) 的 delta 标签本身只用外参的**旋转部分**；完整外参仍应保存，因为相机系中的绝对 EEF 状态、跨视图几何及后续可视化需要平移。标定内参用于 PnP、像素投影和图像几何，不直接出现在上述 delta 公式里。

## 6. rotation6D 中间表示 → 三维 rotvec

对需要表达的 R（相对旋转或绝对姿态），固定 6D 的排列为前两列的列拼接：

```text
d6 = [R00, R10, R20, R01, R11, R21]
```

不是直接 `R[:, :2].reshape(-1)` 的行优先顺序。解码用 Gram–Schmidt：

```text
b1 = normalize(d6[0:3])
b2 = normalize(d6[3:6] - dot(b1, d6[3:6]) * b1)
b3 = cross(b1, b2)
R  = stack_columns(b1, b2, b3)
rotvec = Log_SO3(R)             # 轴 × 旋转角，单位 rad
```

相机模式每侧最终 action 为 `[delta_p_C(3), rotvec(R_delta_C)(3), gripper_width(1)]`，总计 14 维。中间保存 rotation6D 便于对照与几何处理，训练动作的旋转输入最终为 3D rotvec；不要把 6D 当作 6 个可直接相减的角度。

从已有 R 编码再还原本质上应保持同一旋转，需要回环测试。零旋转的 rotation6D 为 `[1,0,0,0,1,0]`，不是六个 0。零角和接近 π 都要测试，rotvec 的主值在 π 附近存在符号 / 分支边界，不能仅检查向量逐元素误差，应比较还原后的旋转矩阵。

## 7. 图像、状态、训练器与数据清单

建议下阶段数据清单至少保存：

```json
{
  "action_mode": "camera_delta_pose",
  "reference_view": "own_wrist",
  "pose_label_source": "follower_measured_future",
  "end_effector_frame": "flange",
  "source_rotation": "euler_fixed_xyz_rad",
  "state_rotation": "rotation6d_columns",
  "action_rotation": "rotvec_rad",
  "position_units": "m",
  "gripper_mode": "absolute_width_m",
  "chunk_anchor": "observation_t",
  "first_target_offset": 1,
  "calibration_bundle_sha256": "<camera delta 必填，absolute base 不需要>",
  "mount_id": "<camera delta 使用>",
  "source_index_mapping": "<重采样到原始帧的映射>"
}
```

相机系状态可为 `T_C_L(t)_F_L(t)`、`T_C_R(t)_F_R(t)`，或全局相机系对应状态。腕部相机与同侧法兰刚性连接，因此同侧相机系法兰状态近似恒定，这是几何事实；建议另外保留各侧 base 位姿 / 关节状态供模型使用，最终 state 维度应在实现前确认，不能悄悄覆盖原有 14 维关节 state。

未来训练需要同时更新 feature schema、状态输入映射、normalization stats、padding mask、checkpoint 契约和推理解码。**位姿 action 也是 14 维，并不表示它能被现有 14 维关节模型互换使用。** 应按动作语义检查清单，不只比较 tensor shape。旋转向量、米和夹爪宽度分别统计，不复用关节的均值 / 方差或分位数文件。

相机原图 640×480 目前通过 `resize_with_padding` 变为 224×224：缩放到 224×168，上下各填 28 像素。新的几何处理若使用像素内参，须保存与实际 OpenCV resize 一致的像素映射；以像素中心坐标描述：

```text
fx' = sx * fx
fy' = sy * fy
cx' = sx * (cx + 0.5) - 0.5 + pad_left
cy' = sy * (cy + 0.5) - 0.5 + pad_top
sx = 224/640, sy = 168/480, pad_left = 0, pad_top = 28
```

像素角坐标实现常写 `K'=A@K`，应将 A 与使用的坐标中心约定一起保存，不能混用半像素约定。纯 resize/pad 保留原畸变系数。裁剪、反转、去畸变、改用不同传感器或 resolution 时，应同步更新几何记录，不能继续假定原 K 有效。

这里只借鉴 Qwen 的位姿表示和公式 (5) 动作几何。Qwen 还有 CaPE 相机几何注意力等模型设计；将这种 action 数据用于 LeRobot / OpenPI **并不等于实现 Qwen 的完整模型、CaPE 或训练配方**。原 PI05 是否受益须通过后续训练与验证判断。

## 8. 推理解码和执行边界

绝对 base 模式将预测旋转表示解码为 `T_B_L_F_L*` 和 `T_B_R_F_R*`，无需相机标定。

camera delta 模式使用该次观测锚点的实测位姿和冻结相机方向解码：

```text
p_B_F*(t,k) = p_B_F(t) + R_B_C(t) · delta_p_C(t,k)
R_B_F*(t,k) = R_B_C(t) · R_delta_C(t,k) · transpose(R_B_C(t)) · R_B_F(t)
```

每个 k 都相对同一锚点重建，不能把这些“相对锚点”的预测增量逐步累加。RTC 剩余 chunk 若跨到新锚点，需要先恢复 base 下的绝对目标，再重新编码到新锚点；不能直接拼接不同锚点的相机 delta，现有 joint RTC 队列逻辑不能不加验证就用于新表示。

目前硬件桥接收关节目标。下一阶段仍须选择并验证“位姿目标 → 关节目标的 IK”或明确支持的笛卡尔执行接口，涉及关节连续性、奇异点、可达性和限幅。**不能把 `[xyz,rotvec,gripper]` 的 14 个数直接传给现有关节执行器。** 本轮不修改底层 SDK、采集器控制和归位逻辑。

## 9. 审阅点及下一阶段测试

已明确：absolute 使用左右各自 base 且无需相机标定；camera delta 支持 own wrist / global；板配置可自定义，默认 4×6 内角点、53.5 mm。

实施前需要确认的设计选择：

1. 旧数据标签使用“从臂未来实测位姿”，还是要求“原主臂关节目标经从臂 FK 得到的位姿”。
2. absolute 的旋转 action 采用建议的 rotvec，还是明确保留原始 Euler RPY；两者都在各自 base 下且不需要相机标定。
3. 状态是否在位姿 20 维之外保留原有关节信息；特别是 own wrist 模式，避免只给模型近似恒定的同侧相机系法兰状态。

后续必须覆盖：绝对模式无标定可运行；RPY/矩阵/6D/rotvec 回环；零角与 π 邻域；左右不同 base；公式 (5) 与公式 (6) 的区分；冻结腕部锚点；chunk 重建与 RTC 换锚点；先重采样再求差；home 和 episode 边界；归一化有效 mask；数据集与 checkpoint 语义不匹配拒绝；独立真实轨迹回放及小步真机验证。通过文档审阅后才实现这部分。
