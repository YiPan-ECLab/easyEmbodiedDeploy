# π0.5 动作语义、真机标定与主流 VLA Benchmark 位姿规范审计

> 初次审计：2026-08-31；坐标系与端到端链路补充：2026-09-01（Asia/Shanghai）  
> 本地工作区：`/Users/ykx971654502163.com/main/vla_source`  
> 结论基于源码而不是只看 README 或张量维数。

## 0. 一页结论

1. **π0.5 模型本身没有固定的坐标系，也没有固定为 delta 或 absolute。**模型核心只生成形状为 `[batch, horizon, action_dim]` 的连续动作张量；最终语义由数据集、机器人适配器、相对动作变换和环境控制器共同决定。
2. OpenPI 官方 `pi05_libero`：预测的是 **LIBERO 仿真固定参考轴（标准场景中与 Panda 基座轴平行）下的末端 6D 增量控制量 + 夹爪开/合方向命令**。它不是腕部/腕部相机坐标系，不是第三人称相机坐标系，也不是绝对 pose。这里的 `world` 是仿真固定参考系，绝不等于第三人称相机光学系。
3. OpenPI 官方 `pi05_droid`：预测的是 **7 个关节速度 + 1 个绝对夹爪位置**。它根本不是 Cartesian pose，所以“base / wrist / global pose”三选一不适用。
4. OpenPI `pi05_aloha`：模型内部前 6+6 个关节维度使用“相对本次推理时当前关节状态”的 chunk-relative 表示；策略后处理会加回当前状态，真机最终执行的是 **双臂绝对关节位置 + 绝对归一化夹爪位置**。也不是 Cartesian pose。
5. LeRobot 原生 π0.5 默认 `use_relative_actions=False`，即 **按数据集原义预测绝对 action**；而 LeRobot 的默认机器人 action 通常是绝对关节位置。只有显式启用 `use_relative_actions=true` 时，才在模型内部改成“相对当前 observation state”的 offset，并在下发前自动恢复绝对 action。
6. **真机需要硬件标定，但 π0.5 没有一套额外的、统一的“模型标定”。**关节零位/方向/量程、夹爪范围和机器人运动学必须正确；stock DROID/ALOHA 图像策略没有读取相机外参矩阵，所以不要求额外运行 hand-eye 外参标定程序，但相机身份、视野和安装位置仍会显著影响成功率。
7. **不要把环境动作的固定缩放与模型统计归一化混为一谈。**例如 LIBERO 的 action 本来就在 `[-1,1]`，但 π0.5 仍使用 checkpoint 对应的统计归一化。OpenPI 和 LeRobot 的标准 loader 会自动归一化和反归一化；自定义训练要离线算一次 stats，推理时不应再手写一遍或用测试集重算。

---

## 1. 审计版本

| 项目 | 审计 commit | 官方仓库 |
|---|---:|---|
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) |
| LeRobot | `4aaff99be4a1d81568c08c8f0296b41b40c99ec4` | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| MetaWorld | `6e01ad7e2ffb2302e4dca04f796fcd8837df8540` | [Farama-Foundation/Metaworld](https://github.com/Farama-Foundation/Metaworld) |
| RoboTwin 2.0 | `30954692d06ba7e89f7a6b76064f4062c488fa81` | [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) |
| VLABench | `cf588fe60c0c7282174fe979f5913170cfe69017` | [OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench) |
| CALVIN | `fa03f01f19c65920e18cf37398a9ce859274af76` | [mees/calvin](https://github.com/mees/calvin) |
| CALVIN env | `797142c588c21e76717268b7b430958dbd13bf48` | [mees/calvin_env](https://github.com/mees/calvin_env) |
| RoboDojo | `ee67a1468510da7624a089164402359f2afc72c8` | [RoboDojo-Benchmark/RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) |
| RoboTwin 内嵌 XPolicyLab | `c37109c500be67d0dea6b36bf7337bbd26e763cd` | [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) |
| RoboDojo 内嵌 XPolicyLab | `432f82b1758c5b1202e42a3dfe014546dbc50871` | [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) |

## 2. 先固定坐标系、数值表示和三种 delta 的定义

### 2.1 本文只使用五个明确的参考系

| 符号 | 本文名称 | 原点与坐标轴 | 是否等于第三人称相机系 |
|---|---|---|---|
| `W` | 仿真固定参考系（源码叫 `world`） | 由 MuJoCo/SAPIEN/Isaac Sim 建场景时固定；不随机器人和相机移动 | **否** |
| `B` | 机器人基座系（`base`/`root`） | 原点固定在机器人基座，轴由机器人模型定义 | 否 |
| `E` | 末端工具系（`eef`/`TCP`/`tool`） | 固定在夹爪或 TCP，随机械臂移动 | 否 |
| `C3` | 第三人称相机光学系 | 原点在第三人称相机光心，轴按相机成像约定定义 | 它就是第三人称相机系本身 |
| `Cw` | 腕部相机光学系 | 原点在腕部相机光心，随腕部移动 | 否 |

“第三人称全局相机”只是“从全局视角拍摄”的口头说法，不意味着它的光学坐标轴就是 `W`。相机在场景中的外参是一个变换，例如 `W_T_C3`；同一点满足：

```text
p_W = W_T_C3 @ p_C3_homogeneous
```

所以 `W` 与 `C3` 是两个不同坐标系，通过相机外参连接。LIBERO 的 OSC 动作没有先投影到任何相机光学系，第三人称图像和腕部图像只是模型输入。

LIBERO 标准 Panda 场景只平移机器人基座，没有旋转基座，因此 `W` 与 `B` 的轴方向平行，即 `W_R_B = I`，但原点不同。由此：

- 对**增量向量**而言，LIBERO 中写“沿 `W` 轴”或“沿 `B` 轴”数值相同；为了对应源码，本文称“仿真固定参考轴/基座平行轴”。
- 对**绝对位置**而言，`p_W` 与 `p_B` 因原点不同而不是同一个三维数值。
- 无论哪一种，都不等于 `C3` 或 `Cw` 相机光学系。

### 2.2 4×4、3×3、四元数、rotation vector 分别是什么

严格数学记号如下：

- `T = [[R,p],[0,1]]` 是 4×4 齐次刚体变换，属于 **SE(3)**。小写 **se(3)** 指它的李代数/切空间，不应把普通 4×4 pose 直接叫“se3 矩阵”。
- `R` 是 3×3 旋转矩阵，属于 **SO(3)**。
- quaternion 是同一个三维旋转的四参数表示；使用前必须明确顺序是 `xyzw` 还是 `wxyz`。
- LIBERO action 的后三个旋转数不是欧拉角，也不是四元数，而是三维 **rotation vector（旋转向量/指数坐标）**：`delta_r = axis * angle`。其方向是旋转轴，模长是旋转角（rad）。它只是一个三维参数向量，不能直接与 3×3 矩阵相乘。

robosuite 实际先做：

```text
delta_r (3,) -> quaternion (4,) -> R_delta (3,3)
```

然后才计算 `R_delta @ R_current`。因此源码中的矩阵乘法两边都是 SO(3) 旋转矩阵，不是拿三维 axis-angle 数组直接乘矩阵。

一个 pose 也可以用 `[x,y,z,qx,qy,qz,qw]`、`[x,y,z,qw,qx,qy,qz]` 或 `[x,y,z,rx,ry,rz]` 保存；“是不是 pose”取决于语义，不取决于是否已经物化成 4×4 数组。需要组合、变换或送入 IK 时，再转换成 `R`/`T`。

### 2.3 LIBERO 环境 controller delta：逐变量解释

一次环境策略步的原始 action 记为：

```text
u_t = [ux, uy, uz, urx, ury, urz, ug]，每维通常位于 [-1, 1]
```

其中 `t` 是第几个 `env.step`，不是连续物理仿真的每个微小积分步。

1. `u_t[0:3] = [ux,uy,uz]`：无量纲平移命令，三个方向依次是仿真固定参考轴 `W` 的 x/y/z；在标准 LIBERO 中也与基座 `B` 的 x/y/z 平行。
2. `u_t[3:6] = [urx,ury,urz]`：无量纲旋转向量命令，不是 roll/pitch/yaw 欧拉角。
3. `u_t[6] = ug`：Panda 夹爪开合方向命令。robosuite 1.4.1 中 `-1` 表示张开、`+1` 表示闭合；它不属于 6D pose，也不是绝对夹爪 qpos。源码见 [`PandaGripper.format_action`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/grippers/panda_gripper.py#L43-L58)。
4. `clip(u_t,-1,1)`：避免策略给出控制器允许范围以外的值。
5. `scale`：将无量纲命令线性映射成物理增量。默认对称配置下：

```text
delta_p = [0.05*ux, 0.05*uy, 0.05*uz]                 # m
delta_r = [0.5*urx, 0.5*ury, 0.5*urz]                 # rotation vector, rad
```

6. `p_current = current_eef_position_t`：该动作开始执行时，末端 `E` 原点在 `W` 中的实际三维位置，单位 m，shape `(3,)`。
7. `R_current = current_eef_rotation_t`：该动作开始执行时末端朝向的 3×3 旋转矩阵；它把一个用末端工具轴表示的向量变换成用 `W` 轴表示的向量。
8. `R_delta = Exp_SO3(delta_r)`：把 rotation vector 转成合法的 3×3 旋转矩阵。robosuite 代码路径等价于 `axisangle2quat(delta_r)` 再 `quat2mat(...)`。
9. `p_goal`、`R_goal`：这一次 controller action 要追踪的**绝对目标末端 pose**的平移和旋转部分：

```text
p_goal = p_current + delta_p
R_goal = R_delta @ R_current
T_goal = [[R_goal, p_goal],
          [0, 0, 0,      1]]                         # 属于 SE(3)
```

为什么这么算：

- `delta_p` 已经用 `W`/基座平行轴表达，所以可直接加到同样用 `W` 表达的 `p_current`。如果它是腕部工具系增量，代码应当是 `p_goal = p_current + R_current @ delta_p_E`，但 LIBERO 没有这一乘法。
- `R_delta @ R_current` 是**左乘**，表示绕固定参考轴旋转。若是绕当前腕部/工具轴旋转，通常会写成 `R_current @ R_delta`，即右乘。
- action 的 6 个数先分别构造 `delta_p` 与 `R_delta`，再得到合法的 `T_goal`。实现并没有把 `[delta_p,delta_r]` 当作一个耦合 twist 直接做完整 `Exp_SE3`；因此不要把它简写成标准的 `T_goal = Exp_SE3(xi) @ T_current`。它是“平移相加 + 旋转左乘”的解耦更新。

每次 `env.step(u_t)` 开始时，OSC 都读取**当时真实**的 `p_current,R_current` 并生成新的 `p_goal,R_goal`。所以模型一次预测 10 项 action chunk 且客户端依次执行 5 项时，第 `k` 项不是相对“生成 chunk 时的初始 pose”解释，而是在它真正送入第 `k` 次 `env.step` 时，相对那一刻的实际末端 pose 解释。

### 2.4 OSC 是什么，输入和最底层输出分别是什么

OSC 是 **Operational Space Control，操作空间控制**。这里的“操作空间”就是末端笛卡尔位置/姿态空间。它不是一个 VLA 模型，也不是相机坐标变换；它是把末端目标 pose 转成各关节电机力矩的反馈控制器。

robosuite 的核心过程可概括为：

```text
position_error = p_goal - p_actual
orientation_error = SO3_error(R_goal, R_actual)       # 3D 小角度误差向量
desired_force  = Kp_pos * position_error    - Kd_pos * linear_velocity
desired_torque = Kp_rot * orientation_error - Kd_rot * angular_velocity
desired_wrench = [desired_force, desired_torque]
joint_torque   = J^T * operational_space_inertia * desired_wrench
                 + gravity_compensation + nullspace_torque
```

- `J` 是机器人末端关于关节的 Jacobian，把关节速度与末端线速度/角速度关联起来。
- `Kp/Kd` 是位置和姿态反馈增益。
- `desired_wrench` 是末端期望力/力矩。
- 最底层输出是 Panda 七个手臂关节的**电机力矩**，不是 4×4 pose；`T_goal` 只是控制器的追踪目标。

源码证据为 robosuite 1.4.1 [`osc.py`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/osc.py) 和 [`control_utils.py`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/utils/control_utils.py)。

### 2.5 chunk-relative（OpenPI 源码类名叫 `DeltaActions`）

OpenPI/LeRobot 的相对动作变换是：

```text
relative_action[t] = absolute_action[t] - current_state_at_inference
```

整个 chunk 的所有未来动作都减同一个当前 state。它不是相邻动作差分。源码见：

- OpenPI [`DeltaActions` / `AbsoluteActions`](openpi/src/openpi/transforms.py)，约 203–244 行。
- LeRobot [`to_relative_actions` / `to_absolute_actions`](lerobot/src/lerobot/processor/relative_action_processor.py)，约 40–81 行。

### 2.6 sequential delta

```text
delta[0] = action[0] - current_state
delta[t] = action[t] - action[t-1]
```

需要逐项累加才能恢复轨迹。这和 OpenPI 的 `DeltaActions` **不是同一种表示**。LeRobot 文档也明确区分了 absolute、relative-to-current 与 sequential delta：[`action_representations.mdx`](lerobot/docs/source/action_representations.mdx)，约 55–116 行。

---

## 3. 为什么不能直接说“π0.5 就是某个坐标系”

OpenPI 的 `Pi0Config` 只定义 `action_dim=32`、`action_horizon=50` 等张量规格；π0.5 相对 π0 的核心区别是 state token 化方式和 action expert 的 adaRMSNorm，并没有定义 `xyz`、关节、base frame 或 wrist frame。见 [`pi0_config.py`](openpi/src/openpi/models/pi0_config.py)，约 18–41、82–86 行。

真实数据链是：

```text
数据集/机器人原始 action
  -> embodiment-specific transform
  -> 可选 absolute→chunk-relative
  -> checkpoint 统计归一化
  -> π0.5 连续动作张量
  -> 统计反归一化
  -> 可选 chunk-relative→absolute
  -> embodiment-specific output transform
  -> 环境或机器人执行
```

OpenPI 的策略构造顺序可在 [`policy_config.py`](openpi/src/openpi/policies/policy_config.py) 约 58–89 行看到；模型推理后再执行 output transform，见 [`policy.py`](openpi/src/openpi/policies/policy.py) 约 68–103 行。

因此必须按 checkpoint / data config 分开判断。

---

## 4. OpenPI 中 π0.5 的仿真与真机动作语义

### 4.1 总表

| 配置/入口 | 场景 | 模型学习的动作 | 策略对外输出/最终执行 | Cartesian 坐标系 |
|---|---|---|---|---|
| `pi05_libero` | LIBERO 仿真 | 原生 OSC 6D delta + 夹爪方向命令 | 同样的 7D controller action | **仿真固定参考轴（与基座轴平行）**；不是腕部或相机系 |
| `pi05_droid` 官方 checkpoint | DROID 真机 | 7D joint velocity + 1D gripper position | 同上，clip 到 `[-1,1]` 后送 `RobotEnv` | 不适用；关节空间 |
| `pi05_aloha` | ALOHA 真机 | 双臂 arm joint chunk-relative；gripper absolute | 自动恢复为 14D absolute joint target | 不适用；关节空间 |
| `pi05_aloha_pen_uncap` | ALOHA 真机微调 | 同上 | 同上 | 不适用；关节空间 |
| `pi05_full_droid_finetune` | DROID 全量复现 | absolute joint position 先转为 chunk-relative；gripper absolute | 自动恢复 absolute joint position | 不适用；关节空间 |
| `pi05_droid_jointpos_polaris` | PolaRiS/DROID joint-pos | 同上一类 joint position 相对表示 | absolute joint position | 不适用；关节空间 |

注意：OpenPI 默认 `ALOHA_SIM` checkpoint 是 `pi0_aloha_sim`，不是 π0.5；[`serve_policy.py`](openpi/scripts/serve_policy.py) 约 58–75 行清楚列出了默认映射。也就是说，仓库里官方现成的 π0.5 仿真评测主线是 LIBERO，而 ALOHA 官方 sim 示例当前默认仍是 π0。

### 4.2 `pi05_libero`：仿真固定参考轴/基座平行轴下的末端增量，不是相机系或腕部系

证据链：

1. `pi05_libero` 使用 `extra_delta_transform=False`；配置注释明确说 LIBERO 原始 action 已经是 delta，不再执行 OpenPI 的 `DeltaActions`。见 [`training/config.py`](openpi/src/openpi/training/config.py) 约 281–343、743–760 行。
2. observation state 是 `eef_pos(3) + axis-angle orientation(3) + gripper_qpos(2)`；模型输出的前 7 维未经位姿积分，直接 `env.step(action)`。见 [`examples/libero/main.py`](openpi/examples/libero/main.py) 约 127–153 行，以及 [`libero_policy.py`](openpi/src/openpi/policies/libero_policy.py) 约 86–100 行。
3. LeRobot 也把 LIBERO 标准动作写成 `Box(-1,1,7)` 的“6D end-effector delta + gripper”，默认 `control_mode=relative`；见 [`libero.mdx`](lerobot/docs/source/libero.mdx) 约 87–93、109–128 行及 [`envs/libero.py`](lerobot/src/lerobot/envs/libero.py) 约 339–361、445–453 行。
4. OpenPI 的 LIBERO 环境锁定 robosuite 1.4.1，并使用 `OSC_POSE`。该控制器用：
   - `goal_pos = current_world_eef_pos + scaled_delta_pos`
   - `goal_ori = R(delta_axis_angle) @ current_world_eef_ori`

   左乘旋转以及直接加到仿真固定参考系中的 position，都表明 delta 按**仿真固定参考轴**解释，而非绕腕部/工具轴解释。标准 LIBERO 中基座只平移而不旋转，故这些轴也与基座轴平行。robosuite 1.4.1 官方源码：[`osc.py`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/osc.py) 与 [`control_utils.py`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/utils/control_utils.py)。

严格答案：

- 位置：仿真固定参考轴下的增量；因标准 Panda 基座轴与其平行，也可理解为基座轴方向的增量，但不能理解成第三人称相机光学轴。
- 姿态：仿真固定参考轴下的 rotation-vector 增量；先转成 `R_delta`，再计算 `R_delta @ R_current`。不是欧拉角直接相加。
- 夹爪：原生开/合方向命令，`-1` 张开、`+1` 闭合；它不是末端 pose delta，也不是绝对夹爪位置。
- 绝对末端 pose 在 controller 内部会构造成 `T_goal in SE(3)`，但模型的 6D action 本身不是 4×4 矩阵，也没有直接做完整的 `Exp_SE3`。
- 标准 OSC 输入 `[-1,1]` 会被控制器固定缩放；robosuite 1.4.1 默认平移范围 ±0.05 m、旋转向量范围 ±0.5 rad。这是 controller scaling，不是数据均值方差归一化。

### 4.3 `pi05_droid`：关节速度，不是 pose

官方部署入口明确创建：

```python
RobotEnv(action_space="joint_velocity", gripper_action_space="position")
```

并将预测 chunk 解释为 7 个 joint velocities + 1 个 gripper position，见 [`examples/droid/main.py`](openpi/examples/droid/main.py) 约 73–80、113–151 行。数据转换脚本也直接拼接 `step["action"]["joint_velocity"]` 与 `gripper_position`，见 [`convert_droid_data_to_lerobot.py`](openpi/examples/droid/convert_droid_data_to_lerobot.py) 约 68–81、135–144 行。

所以官方 `pi05_droid` 的准确答案是：

- 不是 base pose、wrist pose 或 global pose。
- 不是增量 pose，也不是绝对 pose。
- 前 7 维是关节速度命令；第 8 维是绝对夹爪位置。

OpenPI 的复现 README 还特别说明：原始 π0.5-DROID 用 joint velocity；为方便仿真，全量复现配置改用 joint position。见 [`README_train.md`](openpi/examples/droid/README_train.md) 约 34–46 行。因此不能把 `pi05_full_droid_finetune` 的 joint-position 语义反推到公开的 `pi05_droid` checkpoint。

### 4.4 `pi05_aloha`：模型内部是关节 offset，真机执行绝对关节目标

`LeRobotAlohaDataConfig` 默认 `use_delta_joint_actions=True`，mask 为每臂前 6 个关节，夹爪维保持绝对；训练前做 `action -= current_state`，推理后做 `action += current_state`。见 [`training/config.py`](openpi/src/openpi/training/config.py) 约 228–278、572–579 行。

这里的 “delta” 是**整个 chunk 相对本次推理当前关节 state**，不是相邻时间步差分，也不是 Cartesian pose delta。

真实 ALOHA 环境文档和执行代码明确规定 action 为：

```text
left absolute arm qpos(6), left normalized gripper position(1),
right absolute arm qpos(6), right normalized gripper position(1)
```

并调用 `set_joint_positions`，见 [`examples/aloha_real/real_env.py`](openpi/examples/aloha_real/real_env.py) 约 18–38、150–156 行。

### 4.5 DROID joint-position 复现/PolaRiS 的特殊点

`RLDSDroidDataConfig` 在 `action_space == JOINT_POSITION` 时，把 absolute joint position 减去当前 joint state，再在输出端加回；见 [`training/config.py`](openpi/src/openpi/training/config.py) 约 357–409 行。`pi05_full_droid_finetune` 和 PolaRiS joint-pos 配置走的是这条路径。

它们的最终执行语义是 absolute joint target；模型内部仍是相对当前 state 的 joint offset。不要称为增量 pose。

---

## 5. LeRobot 原生 π0.5 的语义

### 5.1 默认是数据集原生 absolute action

LeRobot `PI05Config` 默认：

```python
use_relative_actions = False
relative_exclude_joints = ["gripper"]
```

见 [`configuration_pi05.py`](lerobot/src/lerobot/policies/pi05/configuration_pi05.py) 约 54–70 行。官方文档也明确写着“By default, π0.5 predicts absolute actions”，见 [`pi05.mdx`](lerobot/docs/source/pi05.mdx) 约 331–367 行。

但这里的 absolute 仍然要按数据集 feature 解释：

- SO-100/SO-101 等典型 LeRobot 数据：通常是 absolute motor/joint positions。
- LIBERO 数据：数据集 action 本身就是 OSC delta，因此即便 `use_relative_actions=False`，模型预测的仍是 LIBERO controller delta。
- 如果自定义数据存 absolute EE pose，则预测 absolute EE pose。

所以 `use_relative_actions=False` 不等于“必然是全局 Cartesian pose”，而是“不额外从 action 中减 observation state”。

### 5.2 启用 relative action 后的顺序

LeRobot π0.5 processor 明确规定：

```text
raw -> relative -> normalize -> model -> unnormalize -> absolute
```

见 [`processor_pi05.py`](lerobot/src/lerobot/policies/pi05/processor_pi05.py) 约 133–165 行。`AbsoluteActionsProcessorStep` 使用预处理阶段缓存的同一份当前 state 给整个 chunk 加回，见 [`relative_action_processor.py`](lerobot/src/lerobot/processor/relative_action_processor.py) 约 162–203 行。

最终由 rollout/inference 在下发前执行 postprocessor，见 [`sync.py`](lerobot/src/lerobot/rollout/inference/sync.py) 约 114–134 行。因此使用标准 LeRobot 推理入口时，用户拿到/发给机器人的通常已经是恢复后的 absolute action。

---

## 6. 真机部署 π0.5 到底要不要标定

### 6.1 必须做的：机器人硬件/关节标定

如果动作或 proprioception 在关节空间中，必须保证：

- encoder 零位/中心一致；
- 正负方向一致；
- 每个关节最小/最大量程正确；
- 夹爪开合极性和范围正确；
- 机器人 ID 对应正确的 calibration 文件；
- 若走 EE/IK，URDF、base frame、tool/TCP frame 正确。

LeRobot SO follower 在 `connect()` 时会检查 calibration，不匹配或没有文件就进入 `calibrate()`；校准记录 homing offset、range min/max，并写入电机与本地文件。见 [`so_follower.py`](lerobot/src/lerobot/robots/so_follower/so_follower.py) 约 92–157 行。

电机读写时正是用这些数把原始 encoder 值映射为 degree、`[-100,100]` 或 `[0,100]`，再把策略命令反映射为原始 motor units；见 [`motors_bus.py`](lerobot/src/lerobot/motors/motors_bus.py) 约 854–911 行。没有正确标定时，同一个“30 度”或“夹爪 80”会对应错误的物理位置，模型统计也随之失效。

### 6.2 stock OpenPI DROID/ALOHA 不要求额外 hand-eye 外参标定

- DROID runner 只读取 RGB、joint position、gripper position；虽然提取了 Cartesian position，但没有传给策略，也没有读取相机内外参。见 [`examples/droid/main.py`](openpi/examples/droid/main.py) 约 221–240 行。
- DROID README 要求填写 camera ID 并确认相机能看到工作区；没有外参矩阵标定步骤。
- ALOHA runner 同样直接读取多个 RGB 和 joint state，README 要求按硬件教程安装并配置相机 serial number；见 [`examples/aloha_real/README.md`](openpi/examples/aloha_real/README.md) 约 5–10 行。

原因：stock VLA 是从像素直接学到动作，不做“像素点反投影到 base 坐标再规划”的显式几何计算，所以代码不消费 hand-eye transform。

这不代表相机可以随意放：视角、裁剪、左右相机身份、图像旋转、遮挡和工作区布局必须接近训练分布。它是视觉分布对齐问题，不是代码层的外参矩阵需求。

### 6.3 何时必须额外做几何标定

以下情况需要更严格的 base/tool/camera 标定：

- 策略输出 Cartesian EE pose，真实执行端通过 FK/IK 解析；
- 使用 RGB-D/点云，把相机坐标中的点变换到 robot base；
- 使用视觉伺服、目标检测后显式计算抓取 pose；
- 相机/机器人 base 会移动，且软件需要在线维护坐标变换。

### 6.4 normalization stats 不是机器人标定

统计量描述的是训练数据分布；电机标定描述 encoder 与物理关节的映射。两者都重要，但不能互相替代。

---

## 7. 主流 VLA Benchmark 的位姿/action 规范

### 7.1 汇总表

| Benchmark | 标准 pose/state | 标准 action | 坐标系 | 增量/绝对 | 环境原生缩放 | 是否自带“模型统计归一化” |
|---|---|---|---|---|---|---|
| **LIBERO** | absolute EEF pos + quat/rotation-vector | 3D Δpos + 3D Δrotation-vector + 夹爪方向命令 | `W` 固定参考轴；标准场景与 `B` 基座轴平行；非腕部/相机轴 | 每个 env step 相对当时 EEF 的 controller delta；夹爪 `-1` 开/`+1` 合 | action `[-1,1]`；OSC 再映射到物理增量 | benchmark 本身不做模型 stats；OpenPI/LeRobot checkpoint 会做 |
| **RoboTwin 2.0** | 同时记录 absolute joint state 与 absolute dual-EEF pose | `qpos` 或 `ee` 两种 absolute target | EE pose 在 `W` 仿真固定参考系；joint 无 Cartesian frame | 原生数据/API 为 absolute | 关节是原始弧度；只有 gripper 归一化；EE 为 m+quat | 无统一模型 stats |
| **MetaWorld** | absolute hand xyz（`W`）+ gripper；完整 obs 还有物体 pose | 3D Δxyz + gripper effort；无旋转 action | MuJoCo 的 `W` 固定参考轴 | position delta | `[-1,1]` clip 后乘 `0.01 m` | 无 |
| **VLABench** | absolute EEF pose；LeRobot 数据为 robot-base position + world/base-aligned Euler | 标准 VLA 数据是 absolute EEF target，底层由 IK 转 absolute joint qpos | position 为 robot-base frame（源码只减 base translation）；执行时加回 world base offset；非 wrist | absolute | pose 原始 m/rad；gripper 0/1 | HDF5 本身无；转换 LeRobot 时会计算 stats 元数据 |
| **CALVIN** | absolute TCP xyz+Euler in world；robot_obs 另含 joints | 同时提供 `actions` absolute 与常用 `rel_actions` | world frame；不是 wrist | 两种都有；常用 `rel_actions` | relative pos×50、rot×20 后 clip `[-1,1]`，即单位动作约 0.02 m/0.05 rad；gripper ±1 | 无 benchmark 统一 mean/std |
| **RoboDojo** | absolute joint state；EE position 从 Isaac 全局网格坐标减去该并行环境的平移原点，orientation 不变 | absolute joint position 或 absolute EE pose→IK | 每个并行环境自己的固定参考系 `W_i`；非腕部/相机系 | 当前公开执行路径是 absolute | joint 原始单位；gripper clip `[0,1]`；pose 原始 m+quat | benchmark 无统一 stats；其 π0.5 checkpoint 自带训练 stats |
| **SimplerEnv** | 随具体机器人 | 常见 3D Δxyz + 3D Δaxis-angle + gripper | ManiSkill robot root/base frame（具体 controller config 可改 body/root） | delta | 通常 controller action `[-1,1]` 再固定缩放 | 无统一模型 stats |
| **ManiSkill** | 依 obs/control mode | `pd_joint_pos`、`pd_ee_pose`、`pd_ee_delta_pose` 等均可 | EE 默认 root/base；也可配置 body/wrist-aligned 组合 | 取决于 control mode | `normalize_action` 通常可把 controller range 映射到 `[-1,1]` | 无统一模型 stats |

### 7.2 LIBERO

除第 4.2 节证据外，还要注意 LeRobot 现在允许 `control_mode=absolute`，它通过设置 `robot.controller.use_delta=False` 改变 controller 解释方式。但标准 LIBERO demonstration 和 `pi05_libero` checkpoint 是 relative/OSC-delta 语义；不能只把评测参数改成 absolute 而不重新训练/转换 action。

### 7.3 RoboTwin 2.0

官方数据采集 `get_obs()` 同时保存：

- `joint_action`: 左右臂 drive target + gripper；
- `endpose`: 左右绝对 EE pose + gripper。

源码见固定 commit 的 [`_base_task.py`](https://github.com/RoboTwin-Platform/RoboTwin/blob/30954692d06ba7e89f7a6b76064f4062c488fa81/envs/_base_task.py#L437-L494)。`take_action(action_type='qpos'|'ee')` 对两者都按目标执行；EE 模式交给 IK/path planner，见同文件约 1486–1599 行。官方配置文档也把 `endpose` 定义为 world coordinate frame 中的 EE pose：[RoboTwin configurations](https://robotwin-platform.github.io/doc/usage/configurations.html)。

只有 gripper 被明确归一化；arm qpos 是实际 joint target，不能因为某个 Gym wrapper 把 Box 写成 `[-1,1]` 就误判为统计归一化。

LeRobot 有一个特例：`RoboTwinEnv(action_mode="ee")` 为 LingBot-VA 一类 world model 把 16D 输出解释为**相对 episode 初始 EEF pose**的 delta，再合成为 absolute pose。见 [`lerobot/envs/robotwin.py`](lerobot/src/lerobot/envs/robotwin.py) 约 53–86、350–374、487–503 行。该表示中：

- translation 是直接加到初始 pose 的 world-axis offset；
- rotation 用 `R_init * R_delta`，即相对初始 EE orientation 复合；
- 这属于 LeRobot 模型适配层，不是 RoboTwin 原始 demo 的 action 定义。

### 7.4 MetaWorld

MetaWorld action space 是 `Box([-1]*4, [1]*4)`；`set_xyz_action()` 对前三维 clip 后乘 `action_scale=0.01`，再直接加到 MuJoCo mocap 的 world position；第四维控制两指反向 effort。官方源码：[`sawyer_xyz_env.py`](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/sawyer_xyz_env.py#L174-L243) 和[同文件 step](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/sawyer_xyz_env.py#L579-L596)。

所以它是 world-axis position delta，不控制 EEF orientation。

### 7.5 VLABench

官方 trajectory generator 保存 key-point waypoint，并从 position 中减去 robot base position后写入 `trajectory`；见 [`trajectory_generation.py`](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/scripts/trajectory_generation.py#L97-L103)。LeRobot converter 保留 action，并把 state 的 world position 同样减 base translation；见 [`convert_to_lerobot.py`](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/scripts/convert_to_lerobot.py#L66-L101)。

推理时官方 OpenPI adapter 把输出拆成 `target_pos, target_euler, gripper`，给 position 加回 base offset，再通过 IK 执行；没有把输出加到 current EEF pose，见 [`openpi.py`](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/VLABench/evaluation/model/policy/openpi.py#L61-L86)。因此标准 VLABench VLA action 是 absolute EE target。

仓库的 augmentation 脚本确实能另外生成 sequential `delta_actions`，但标准 LeRobot converter 用的是 `actions`/`trajectory`，不能因此把主数据误判为 delta。

### 7.6 CALVIN

官方 dataset README 已直接定义：

- `actions`: TCP position 与 Euler orientation 都是 absolute world coordinates，gripper ±1；
- `rel_actions`: position 与 Euler 都是 relative world coordinates，分别×50/×20并 clip 到 `(-1,1)`。

见 [`calvin/dataset/README.md`](https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/dataset/README.md#L69-L89)。环境代码把 relative position/orientation 分别乘 `0.02`/`0.05` 后加到当前 target pose，见 [`robot.py`](https://github.com/mees/calvin_env/blob/797142c588c21e76717268b7b430958dbd13bf48/calvin_env/robot/robot.py#L228-L280)。官方常用 datamodule 配置选择 `rel_actions`。

这里的 `[-1,1]` 是有物理意义的固定 controller scaling，不是 mean/std。

### 7.7 RoboDojo

RoboDojo 最新固定版本已经包含 XPolicyLab 的 π0.5 数据转换、训练和评测适配，不再只是 eval-only。源码的 observation manager 把 joint state 原样写入，同时可提供 EE pose；[`robot_manager.get_link_pose(..., is_relative=True)`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env/robot_manager/robot_manager.py#L131-L170) 只从 Isaac 全局网格 position 中减去该并行环境的平移原点，orientation 不变。因此源码变量名 `is_relative` 在这里表示“换了平移原点后的绝对坐标”，不是相对当前 EEF 的动作 delta。本文把这个参考系明确记为每个并行环境自己的固定参考系 `W_i`。

公开 action 路径支持：

- arm joint position target；
- absolute EE pose，经 IK 变为 joint target；
- gripper `[0,1]`。

见 [`eval_env.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/src/eval_client/eval_env.py#L337-L446)。虽然 validator 中列出了 `delta_ee_pose` 字段，但当前 action-type 识别与执行分支没有消费这一字段，因此不能把它当成已打通的公开 delta-action API。更重要的是，当前随仓库发布的 π0.5 默认 recipe 走 **joint action**，并不走这个 EE-pose 分支；完整链路见第 9.5 节。

### 7.8 SimplerEnv / ManiSkill 补充

SimplerEnv 官方示例把 7D action 定义为 `delta xyz + delta axis-angle + gripper`：[SimplerEnv repository](https://github.com/allenai/SimplerEnv)。其底层 ManiSkill controller 支持明确的 frame 选项：root/base translation、body/wrist translation，以及 root/body aligned rotation。当前 ManiSkill `pd_ee_delta_pose` 默认是 root/base-aligned delta；`pd_ee_pose` 则是 base-frame absolute pose。官方控制器源码见 [`pd_ee_pose.py`](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/controllers/pd_ee_pose.py)。

因此对 ManiSkill 系 benchmark 必须同时报告 `control_mode` 和 controller 的 `frame`；只写“7D EE action”是不够的。

---

## 8. 是否要手算均值方差、推理时手动反归一化

### 8.1 最直接的答案

| 使用方式 | 训练前是否要计算 stats | 推理时是否手动反归一化 |
|---|---|---|
| OpenPI 官方 checkpoint + `create_trained_policy` / policy server | 否；checkpoint 已带对应 stats | **否**，loader 自动做 |
| LeRobot 完整 checkpoint + 官方 pre/postprocessor | 否；processor/checkpoint 已保存 stats | **否**，postprocessor 自动做 |
| OpenPI 自定义数据微调 | **是，离线算一次** | 否，保存进 assets 后由 loader 自动做 |
| LeRobot 自定义数据微调 | dataset 需要 stats；通常创建/整理 dataset 时计算 | 否，训练保存的 processor 自动做 |
| 自己只加载裸模型权重，不加载官方 processor | 必须使用训练时同一套 stats | **是**，但这是自定义部署责任，不是 benchmark 要求 |

### 8.2 OpenPI

π0.5 默认使用 quantile normalization，而不是只使用 mean/std：`q01 -> -1`，`q99 -> +1`。OpenPI 的 `DataConfigFactory` 对非 π0（包括 π0.5）启用 quantile norm，见 [`training/config.py`](openpi/src/openpi/training/config.py) 约 179–188 行；公式见 [`transforms.py`](openpi/src/openpi/transforms.py) 约 114–181 行。

自定义训练先运行：

```bash
uv run scripts/compute_norm_stats.py --config-name <your_config>
```

脚本在 embodiment/data transform 之后统计 `state` 和 `actions`，并写入 config assets，见 [`compute_norm_stats.py`](openpi/scripts/compute_norm_stats.py) 约 89–113 行。

推理时 `create_trained_policy` 从 **checkpoint assets** 加载与训练完全一致的 stats，输入端 Normalize，输出端 Unnormalize；见 [`policy_config.py`](openpi/src/openpi/policies/policy_config.py) 约 58–88 行。因此不要在 client 端重复反归一化。

### 8.3 LeRobot

LeRobot π0.5 默认 `STATE` 与 `ACTION` 使用 q01/q99 quantiles，见 [`configuration_pi05.py`](lerobot/src/lerobot/policies/pi05/configuration_pi05.py) 约 89–95 行；公式与 inverse 见 [`normalize_processor.py`](lerobot/src/lerobot/processor/normalize_processor.py) 约 386–401 行。

但**具体 checkpoint 的保存配置优先于模型默认值**。例如当前 LeRobot LIBERO π0.5 recipe 显式使用 `MEAN_STD` 来匹配其发布 checkpoint，见 [`pi05.mdx`](lerobot/docs/source/pi05.mdx) 约 77–104 行。不能看到“π0.5”就硬编码 quantile。

官方推理从 checkpoint 同时加载 preprocessor 和 postprocessor，见 [`policies/factory.py`](lerobot/src/lerobot/policies/factory.py) 约 220–249 行；postprocessor 会把输出恢复到环境单位。

若 `use_relative_actions=true`，必须对**变换后的 relative action**重算 stats，不能继续使用 absolute-action stats。LeRobot 文档提供 `lerobot-edit-dataset --operation.relative_action true`，见 [`pi05.mdx`](lerobot/docs/source/pi05.mdx) 约 331–357 行。

### 8.4 为什么 raw action 已是 `[-1,1]` 仍可能需要 checkpoint stats

这是两层不同变换：

```text
环境层：controller_command [-1,1] -> 物理位移/速度/力
模型层：dataset action -> mean/std 或 q01/q99 标准化 -> 模型输出 -> inverse stats
```

LIBERO、MetaWorld、CALVIN-relative 都有环境层固定范围；这不自动取消模型层统计。是否使用模型层 stats，由 checkpoint 的 processor/config 决定。

### 8.5 禁止做法

- 不要用评测/真机 rollout 临时计算 mean/std。
- 不要把一个 embodiment 的 stats 用到另一个 action layout。
- 不要把 absolute-action stats 用于 relative-action 训练。
- 不要在 server 已经 Unnormalize 后又在 client 端反归一化一次。
- 不要把 gripper 的 `[0,1]`、`[-1,1]` 固定映射误当成数据统计标准化。

---

## 9. π0.5 + LIBERO / RoboTwin / RoboDojo 的完整端到端链路

### 9.1 先固定本节的变量、坐标轴和六个数据层

本节只讨论当前固定版本真正打通的 recipe：OpenPI `pi05_libero`、RoboTwin 内嵌 XPolicyLab `Pi_05 + joint`、RoboDojo 内嵌 XPolicyLab `Pi_05 + joint`。RoboTwin 和 RoboDojo 可选的 EE-pose API 会单独标成“旁路”，不能混进当前 joint checkpoint 主链路。

本节使用的变量全部在这里定义：

| 变量 | 唯一含义 |
|---|---|
| `t` | 当前数据样本或当前发起一次模型推理的帧 |
| `k` | action chunk 中第几行，从 `0` 开始 |
| `a_data` | LeRobot 中尚未做 checkpoint 统计归一化的 action |
| `a_train_before_norm` | 做完 joint absolute→offset 等 embodiment 变换、但尚未做 checkpoint 统计归一化的 action；仅在 RoboTwin/RoboDojo joint 链路使用 |
| `a_train` | 做完 embodiment 变换和 checkpoint 统计归一化后，真正作为模型训练目标的 action |
| `a_model` | π0.5 网络直接生成的统计归一化空间动作 |
| `a_data_hat` | 对 `a_model` 做 checkpoint 逆归一化后，恢复到 LeRobot 原生 action 空间的预测值 |
| `u_t` | 从 π0.5 输出 chunk 中取出、准备送给 LIBERO `env.step()` 的一行 7D action |
| `q_left[t]`、`q_right[t]` | 第 `t` 帧左右臂各 6 个绝对关节位置，单位 rad |
| `g_left[t]`、`g_right[t]` | 第 `t` 帧左右夹爪各 1 个归一化绝对开度 |
| `q01`、`q99` | 训练集某一 action 维度的 1% 和 99% 分位点 |

本节只使用前文已经定义的坐标系：`W` 是仿真固定参考系，`B` 是机器人基座系，`E_t` 是第 `t` 时刻末端工具系，`C3` 是第三人称相机光学系，`Cw` 是腕部相机光学系。矩阵记号严格按以下规则：

| 记号 | 源坐标轴 → 目标坐标轴 |
|---|---|
| `^W R_E` | 把用末端轴 `E` 表示的三维向量转换为用固定轴 `W` 表示；3×3，属于 SO(3) |
| `^W T_E` | 把用末端系 `E` 表示的齐次点转换到固定系 `W`；4×4，属于 SE(3) |
| `^W T_C3` | 把第三人称相机系 `C3` 中的齐次点转换到固定系 `W`；它是相机外参，不参与当前三条 action 主链路 |
| `W_i` | RoboDojo 第 `i` 个并行仿真环境自己的固定参考系；只减去 Isaac 全局网格中的环境平移 origin，轴方向仍是固定仿真轴 |

`R_delta^W` 不是“从某个新坐标系变到另一个坐标系”的被动外参；它表示一个旋转轴用 `W` 表达的主动旋转算子。它左乘 `^W R_E`，把末端三根轴在 `W` 中的列向量一起绕 `W` 固定轴旋转。

六个数据层统一写成：

```text
原生 benchmark 文件（RLDS / PKL / HDF5）
 -> LeRobot 单帧数据
 -> 训练时未来 action chunk
 -> π0.5 网络直接输出的推理 chunk
 -> checkpoint 后处理后的环境 action
 -> 环境内部目标与最底层 actuator 数据
```

值域也分两种：环境规定的合法值域可能是 `[-1,1]` 或 `[0,1]`；模型统计归一化空间没有硬边界。π0.5 输出层没有 `tanh`，所以 `a_model` 理论上属于全体实数，只是正常输出通常集中在训练目标分布附近。

### 9.2 为什么有时夹爪是 2 维，有时是 1 维

夹爪“观测维数”“策略动作维数”和“底层物理关节数”不是一回事。

#### LIBERO Panda

- observation 中有两个夹爪 qpos，因为 MuJoCo Panda 模型有两个滑动手指关节。第一指的物理范围是 `[0,0.04] m`，第二指是 `[-0.04,0] m`；两者符号相反。
- action 中只有一个 `ug`，因为 robosuite 把两指耦合成一个抽象开/合自由度：`ug=-1` 张开，`ug=+1` 闭合。
- 底层再把这一维方向命令变成两个方向相反的 finger actuator target。因此“state 2D、action 1D”没有矛盾。

对应范围和单维耦合逻辑见 robosuite 1.4.1 的 [`panda_gripper.xml`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/assets/grippers/panda_gripper.xml) 与 [`PandaGripper.format_action`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/grippers/panda_gripper.py#L43-L58)。

#### RoboTwin / RoboDojo 双臂

- 每只机械臂有一个夹爪抽象标量，所以双臂模型向量中共有两个夹爪维度：左 1D、右 1D。
- 每个标量都是绝对归一化开度，合法值域 `[0,1]`，不是相对上一时刻的增量。RoboTwin 中 `0` 是闭合、`1` 是张开；RoboDojo 再根据机器人配置中的方向符号把 `[0,1]` 映射到实际关节范围。
- 一个抽象夹爪标量仍可在底层生成两个或更多物理夹爪关节 target；第二个物理关节常由 mimic 关系计算。因此“模型每夹爪 1D”不等于“硬件只有一个关节”。

### 9.3 π0.5 + LIBERO

#### 9.3.1 原始 RLDS：逐帧保存的内容

LIBERO 原始数据经过 TFDS/RLDS 读取。每个 `step` 的有效字段如下：

| RLDS 字段 | shape / dtype | 每项代表什么 | 原始值域或单位 |
|---|---:|---|---|
| `observation/image` | `(256,256,3)` RGB | 第三人称相机 `C3` 图像，只是模型视觉输入 | `uint8 [0,255]` |
| `observation/wrist_image` | `(256,256,3)` RGB | 腕部相机 `Cw` 图像，只是模型视觉输入 | `uint8 [0,255]` |
| `observation/state[0:3]` | `(3,)` | 末端原点在 `W` 中的绝对位置 | m；工作空间决定，不是 `[-1,1]` |
| `observation/state[3:6]` | `(3,)` | 绝对朝向 `^W R_E` 的 rotation vector 参数化 | rad；标准四元数转换下模长在 `[0,π]` |
| `observation/state[6]` | 标量 | Panda 第一指当前滑动关节位置 | `[0,0.04] m` |
| `observation/state[7]` | 标量 | Panda 第二指当前滑动关节位置 | `[-0.04,0] m` |
| `action[0:3]` | `(3,)` | 无量纲平移 controller command，轴为 `W`；标准 LIBERO 中也与 `B` 轴平行 | 合法值域逐维 `[-1,1]` |
| `action[3:6]` | `(3,)` | 无量纲增量 rotation-vector command，旋转轴用 `W` 表达；不是欧拉角、不是四元数 | 合法值域逐维 `[-1,1]` |
| `action[6]` | 标量 | 一维夹爪开/合方向 | `-1` 张开，`+1` 闭合 |
| `language_instruction` | 字符串 | 当前任务文本 | 无数值值域 |

`state[3:6]` 和 `action[3:6]` 都是三个数，但含义不同：前者参数化绝对 `^W R_E`，后者参数化本次增量旋转。二者都不是欧拉角。

RLDS 的 7D action 是 demonstration 当时实际交给 `env.step()` 的 controller command。它不是根据相邻绝对 pose 离线做减法得到的，也不是 4×4 `^W T_E`。源码入口见 [`convert_libero_data_to_lerobot.py`](openpi/examples/libero/convert_libero_data_to_lerobot.py) 和 [`main.py`](openpi/examples/libero/main.py)。

#### 9.3.2 LeRobot：只换容器，不改 action 数值

转换脚本把 RLDS 单帧原样写成 LeRobot：

```text
image            <- RLDS observation/image
wrist_image      <- RLDS observation/wrist_image
state[8]         <- RLDS observation/state
actions[7]       <- RLDS action
task             <- RLDS language_instruction
fps              = 10
```

此阶段没有统计归一化、没有乘 `0.05/0.5`、没有 rotation vector 转矩阵、没有 absolute/delta 转换。因此 LeRobot `actions[t]` 与 RLDS `action[t]` 是完全相同的 7D controller-command 空间。

#### 9.3.3 模型训练 chunk：每行到底相对谁

以第 `t` 帧 observation 为条件，loader 读取：

```text
a_data chunk = [actions[t], actions[t+1], ..., actions[t+9]]
shape        = (10,7)
```

episode 末尾未来帧不足时，LeRobot loader 复制最后一个有效 action，并记录 padding 标记。

同一个训练样本的模型输入为：第三人称图像映射到 `base_0_rgb`，腕部图像映射到 `left_wrist_0_rgb`，不存在的右腕图像用全零图占位并标成无效；图像 resize/pad 到 `224×224`，进入模型时从 `uint8 [0,255]` 转为 `float32 [-1,1]`。8D state 按自己的 q01/q99 统计归一化后补到 32D，prompt 被 token 化。这些输入处理不改变 action 的坐标轴或 delta 定义。

这 10 行的语义逐行相同：每一行都是一个未来 `env.step()` 要解释的 controller delta。它们：

- 不是绝对末端 pose；
- 不是全部相对 chunk 第一帧 pose 的预积分增量；
- 不是 `actions[t+k]-actions[t+k-1]` 这种相邻 action 差分；
- 是 demonstration 在各自未来时刻记录下来的原生 OSC command，执行时应相对该行真正执行时的实际末端 pose 解释。

`pi05_libero` 明确设置 `extra_delta_transform=False`，所以 OpenPI 不执行 `DeltaActions`，不会从这 7 维中再减 observation state。

接下来对 `state` 和 `actions` 分维使用 checkpoint 的 quantile stats。对任意一个 action 维度：

```text
a_train = 2 * (a_data - q01) / (q99 - q01 + 1e-6) - 1
```

`a_data=q01` 映射为约 `-1`，`a_data=q99` 映射为约 `+1`；实现没有硬裁剪，所以 outlier 可以小于 `-1` 或大于 `+1`。最后从 7 维补零到 32 维：

```text
模型训练目标 shape = (10,32)
前 7 维             = 统计归一化后的原生 LIBERO controller command
后 25 维            = 0 padding
```

#### 9.3.4 模型推理 chunk：网络值和环境值分开看

π0.5 网络直接生成：

```text
a_model.shape = (10,32)
a_model 理论值域 = 全体实数；没有硬限制为 [-1,1]
```

Policy output pipeline 对前 7 个有效维度做逆变换：

```text
a_data_hat = (a_model + 1) / 2 * (q99 - q01 + 1e-6) + q01
```

后 25 维没有对应 stats，随后由 `LiberoOutputs` 丢弃。Policy server 最终返回：

```text
predicted_chunk.shape = (10,7)
每行 = [ux,uy,uz,urx,ury,urz,ug]
```

这 7 维已经回到 RLDS/LeRobot 原生 controller-command 空间。合法环境范围是逐维 `[-1,1]`，但 policy 后处理没有硬裁剪，所以模型外推时返回值可能暂时越界；真正的 controller 会再裁剪。客户端不能再次手动反归一化。

#### 9.3.5 `env.step(u_t)` 收到什么

默认客户端只取预测 chunk 前 5 行，依次执行，然后重新推理：

```text
u_t = predicted_chunk[k]       # shape (7,)，k=0...4
env.step(u_t.tolist())
```

从 policy server 到 `env.step()` 之间只发生“取一行”和 NumPy→list，不发生物理缩放，也不构造旋转矩阵。

进入环境后才进行：

```text
u_used  = clip(u_t, -1, 1)
delta_p = 0.05 * u_used[0:3]       # 每分量 [-0.05,0.05] m
delta_r = 0.5  * u_used[3:6]       # 每分量 [-0.5,0.5] rad
R_delta^W = RotVecToMatrix(delta_r)
```

`delta_r` 是 rotation vector：旋转轴为 `delta_r/||delta_r||`，旋转角为 `||delta_r||`。它不是依次绕 xyz 旋转的欧拉角。当三个分量都达到 `0.5` 时，旋转角模长可达到 `sqrt(3)*0.5≈0.866 rad`，不是总角度上限 `0.5 rad`。

每一行真正执行时，环境重新读取：

```text
p_current = 末端 E_t 原点在 W 中的实际位置
R_current = ^W R_Et              # 源轴 E_t -> 目标轴 W

p_goal = p_current + delta_p
R_goal = R_delta^W @ R_current   # R_goal = ^W R_Egoal，源轴 E_goal -> 目标轴 W

T_goal = ^W T_Egoal              # 源坐标 E_goal -> 目标坐标 W
```

这里 4×4 目标矩阵明确为：

```text
^W T_Egoal = [[^W R_Egoal, p_goal],
              [0, 0, 0,       1]]
```

其左上 3×3 把 `E_goal` 轴表示的向量转换到 `W` 轴，右上 3×1 是 `E_goal` 原点在 `W` 中的位置。

因为 `delta_p` 直接加到 `W` 中的位置、`R_delta^W` 左乘，所以平移方向和旋转轴都按 `W` 固定轴解释；标准 LIBERO 中 `W` 与 `B` 轴平行。它不是 `C3`、不是 `Cw`，也不是当前腕部轴。

chunk 第 `k` 行使用的是它真正执行时的 `p_current/R_current`。即使上一行没有完全到达目标，下一行也从新的实际 pose 出发。因此 LIBERO chunk 是“逐执行时刻解释的 controller delta”，不是“统一相对 chunk 第一帧的 pose offset”。

#### 9.3.6 最底层接收什么

OSC 使用 `^W T_Egoal` 和实际 `^W T_Et` 的误差计算末端期望力/力矩，再经 Jacobian 映射为 Panda 七个手臂关节 torque。MuJoCo 手臂 actuator 最终接收的是 7D 关节力矩。

夹爪不进入 `^W T_Egoal`：一维 `ug` 按符号更新两个方向相反的 finger position actuator target。Panda MuJoCo [`robot.xml`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/assets/robots/panda/robot.xml) 对手臂 motor command 的最终限幅为 joint1–joint5 各 `[-80,80] N·m`、joint6–joint7 各 `[-12,12] N·m`；这属于 actuator 限幅，不是模型归一化。LIBERO 是仿真 benchmark，因此这里的“最底层”指 MuJoCo actuator，并没有真实硬件通信层。

### 9.4 π0.5 + RoboTwin 2.0 当前 joint recipe

#### 9.4.1 原生 RoboTwin PKL：不经过 RLDS

RoboTwin 当前链路不是 RLDS。原生采集的 PKL 每帧可同时包含：

| PKL 字段 | shape | 每项代表什么 | 值域 / 表示 | 是否进入当前 π0.5 joint recipe |
|---|---:|---|---|---|
| `joint_action/left_arm` | `(6,)` | 左臂当前绝对 drive target | rad；机器人关节限位内，不是 `[-1,1]` | 是 |
| `joint_action/left_gripper` | 标量 | 左夹爪绝对归一化开度 | `[0,1]`，0 闭合、1 张开 | 是 |
| `joint_action/right_arm` | `(6,)` | 右臂当前绝对 drive target | rad；机器人关节限位内 | 是 |
| `joint_action/right_gripper` | 标量 | 右夹爪绝对归一化开度 | `[0,1]` | 是 |
| `endpose/left_endpose` | `(7,)` | 左末端绝对 pose `[x,y,z,qw,qx,qy,qz]` | position 为 m；quaternion 每分量 `[-1,1]` 且范数 1 | 否 |
| `endpose/right_endpose` | `(7,)` | 右末端绝对 pose `[x,y,z,qw,qx,qy,qz]` | 同上 | 否 |
| `observation/<camera>/rgb` | 图像 | 头部/腕部 RGB | `uint8 [0,255]` | 是 |

`joint_action` 这个名字容易误解：源码读取的是当前关节 drive target，不是相邻帧 joint delta。`endpose` 中 quaternion 是 `wxyz`，不是 rotation vector、不是欧拉角；但当前 π0.5 converter 根本不读取它。

若把可选 endpose 写成矩阵，`^W T_E` 表示源坐标轴 `E` 到目标坐标轴 `W` 的绝对变换；当前 joint recipe 不构造也不训练这个矩阵。

#### 9.4.2 XPolicyLab HDF5：state/action 的时间对齐

`pkl2hdf5.py` 对原生序列进行：

```text
HDF5 state[t]  = PKL joint_action[t]
HDF5 action[t] = PKL joint_action[t+1]
```

所以 HDF5 `state` 和 `action` 都是 14D：

```text
[left_q6, left_g1, right_q6, right_g1]
```

手臂为绝对关节位置，夹爪为绝对开度。HDF5 还可以保存 `left/right_ee_poses` 的下一帧绝对 `[xyz,wxyz]`，但当前 π0.5 joint converter 不使用这些字段。源码见 [`pkl2hdf5.py`](https://github.com/RoboTwin-Platform/RoboTwin/blob/30954692d06ba7e89f7a6b76064f4062c488fa81/envs/utils/pkl2hdf5.py#L112-L160)。

#### 9.4.3 LeRobot 单帧：14D absolute joint target

Pi_05 converter 从 HDF5 `state` 打包：

```text
observation.state[t]
= [q_left[t](6), g_left[t](1), q_right[t](6), g_right[t](1)]
shape = (14,)
```

converter 不读取 HDF5 `action` group，而是从 state 再构造：

```text
action[t] = observation.state[t+1]
最后一帧 action = 最后一帧 observation.state
```

对非末尾帧，这与 HDF5 下一帧 action 的语义一致。LeRobot action 仍是 absolute joint target，不是 delta。图像 feature 为 `cam_high`、`cam_left_wrist`、`cam_right_wrist`，converter 声明为 CHW `(3,480,640)`，RGB 像素 `uint8 [0,255]`。源码见固定 XPolicyLab 版本的 [`process_data.py`](https://github.com/XPolicyLab/XPolicyLab/blob/c37109c500be67d0dea6b36bf7337bbd26e763cd/policy/Pi_05/openpi/scripts/process_data.py#L94-L132)。

#### 9.4.4 模型训练 chunk：手臂统一相对 q[t]，夹爪保持绝对

π0.5 horizon 为 50。以 observation frame `t` 为条件，LeRobot loader 先取绝对未来目标：

```text
a_data[k]
= [q_left[t+k+1], g_left[t+k+1], q_right[t+k+1], g_right[t+k+1]]
k = 0...49
shape = (50,14)
```

若靠近 episode 末尾不足 50 个未来 action，LeRobot loader 复制最后一个有效 action，并用 padding 标记指出复制的行；上式中的 `t+k+1` 默认表示尚未越过 episode 末尾的普通情况。

输入侧，`cam_high`、`cam_left_wrist`、`cam_right_wrist` 分别映射到模型的第三人称、左腕和右腕图像槽，resize 到 `224×224` 并在进入网络时转为 `float32 [-1,1]`；14D `observation.state[t]` 用自己的 q01/q99 归一化并补到 32D。π0.5 还把归一化 state 与 prompt 一起编码为 token。输入处理与下面的 action offset 处理是两条独立路径。

`LeRobotAlohaDataConfig` 的 mask 是：

```text
[True×6, False×1, True×6, False×1]
```

训练前变成：

```text
a_train_before_norm[k]
= [q_left[t+k+1]  - q_left[t],
   g_left[t+k+1],
   q_right[t+k+1] - q_right[t],
   g_right[t+k+1]]
```

因此：

- 12 个手臂维度都是相对同一个 chunk 查询帧 `q[t]` 的 offset；
- 不是相邻差分 `q[t+k+1]-q[t+k]`；
- 两个夹爪维度完全不做减法，仍是未来绝对 `[0,1]` 开度；
- 全程没有 position、rotation vector、欧拉角、quaternion 或 SE(3)，因为这是 joint recipe。

然后才对上述 14D 变换结果逐维做 q01/q99 quantile normalization，再补零到：

```text
a_train.shape = (50,32)
前 14 维 = 归一化后的 joint-offset + absolute-gripper
后 18 维 = 0 padding
```

手臂 offset 和夹爪虽然一起进入 stats，但原始单位不同；每个维度使用自己的 q01/q99。归一化结果没有硬 `[-1,1]` 边界。

#### 9.4.5 模型推理 chunk 与恢复 absolute joint target

π0.5 网络直接输出：

```text
a_model.shape = (50,32)
a_model 理论值域 = 全体实数
```

后处理顺序为：

```text
Unnormalize
 -> 得到 12D arm offset + 2D absolute gripper
 -> 每个 k 的左臂 offset 都加同一份当前 q_left[t]
 -> 每个 k 的右臂 offset 都加同一份当前 q_right[t]
 -> 夹爪不加当前值
 -> 截取前 14 维
```

最终第 `k` 行是：

```text
[q_left_target[k](6), g_left_target[k](1),
 q_right_target[k](6), g_right_target[k](1)]
```

其中手臂已经恢复为绝对 rad target；夹爪是绝对 `[0,1]` target。网络外推可能产生超出物理关节范围或 `[0,1]` 的数值：夹爪在环境端会 clip，手臂没有统一的 `[-1,1]` clip，合法范围由机器人关节限位和规划器决定。

XPolicyLab 再把每行拆成字典：

```text
left_arm_joint_state[6]
left_ee_joint_state[1]
right_arm_joint_state[6]
right_ee_joint_state[1]
```

#### 9.4.6 环境接口与 chunk 执行

RoboTwin 这里不调用 Gym 风格 `env.step()`；等价执行入口是：

```text
flat_action
= [left absolute q6, left absolute g1,
   right absolute q6, right absolute g1]
shape = (14,)

task_env.take_action(flat_action, action_type="qpos")
```

当前评测循环会依次执行返回 chunk 的全部 50 行，除非 episode 提前结束。每执行一行后虽会刷新 observation，但不会在 chunk 中途重新调用 π0.5；下一次模型推理发生在当前 chunk 执行完之后。

由于每一行在 policy 后处理时已经恢复为 absolute q target，所以环境执行第 `k` 行时不再加 `q[t]`，也不进行相邻累加。

#### 9.4.7 最底层 actuator 数据

对每个 14D absolute target：

1. RoboTwin 从当前关节 target 到本行目标 q 生成 TOPP 时间参数化轨迹；
2. 每个稠密仿真控制步产生左右臂各 6D absolute joint position target 和 6D joint velocity target；
3. SAPIEN joint drive 接收 `set_drive_target(position)` 和 `set_drive_velocity_target(velocity)`；
4. 每侧夹爪标量先 clip 到 `[0,1]`，映射到配置的真实 gripper joint range，再按 mimic 系数生成多个物理关节 position/velocity target。

所以最底层是 joint drive position/velocity target，不是 torque action，也不是 `^W T_E`。position 的合法范围来自该机器人 URDF/SAPIEN joint limits，velocity 来自 TOPP 规划结果；二者都没有统一 `[-1,1]` 值域。当前 benchmark 是仿真环境，不存在报告能继续追踪的真实硬件总线命令。源码见 [`take_action`](https://github.com/RoboTwin-Platform/RoboTwin/blob/30954692d06ba7e89f7a6b76064f4062c488fa81/envs/_base_task.py#L1486-L1584) 与 [`set_arm_joints`](https://github.com/RoboTwin-Platform/RoboTwin/blob/30954692d06ba7e89f7a6b76064f4062c488fa81/envs/robot/robot.py#L609-L617)。

### 9.5 π0.5 + RoboDojo 当前 joint recipe

#### 9.5.1 原生 RoboDojo HDF5：不经过 RLDS

RoboDojo 当前链路也不经过 RLDS。原生 observation/HDF5 中与 π0.5 joint recipe 有关的字段为：

| HDF5 字段 | shape | 每项代表什么 | 值域 / 单位 |
|---|---:|---|---|
| `state/left_arm_joint_states` | `(6,)` | 左臂当前绝对实际关节位置 | rad；Isaac articulation 关节限位内 |
| `state/left_ee_joint_states` | `(1,)` | 左夹爪当前绝对归一化开度 | `[0,1]` |
| `state/right_arm_joint_states` | `(6,)` | 右臂当前绝对实际关节位置 | rad；关节限位内 |
| `state/right_ee_joint_states` | `(1,)` | 右夹爪当前绝对归一化开度 | `[0,1]` |
| `state/left_ee_pose`、`right_ee_pose`（若启用） | 每侧 `(7,)` | 绝对 `[x,y,z,qw,qx,qy,qz]` | m + 单位 quaternion |
| `vision/cam_head/colors` | RGB | 第三人称图像 | `uint8 [0,255]` |
| `vision/cam_left_wrist/colors` | RGB | 左腕图像 | `uint8 [0,255]` |
| `vision/cam_right_wrist/colors` | RGB | 右腕图像 | `uint8 [0,255]` |

RoboDojo 原生 observation 还可带 `action` group，但当前 Pi_05 converter 不读取它，而只读取 `state`。因此本链路不会因为原生 action group 的命名产生额外 delta。

可选 EE pose 中的 quaternion 顺序是 `wxyz`，不是 rotation vector、不是欧拉角。若转换为矩阵，则 `^W_i T_E` 的源坐标轴是末端 `E`，目标坐标轴是该并行环境自己的固定参考系 `W_i`；`W_i` 的原点是 Isaac 全局位置减去该环境的平移 origin，轴方向仍是固定仿真轴。当前 joint recipe 不使用这个 pose。

#### 9.5.2 LeRobot 单帧：converter 用下一帧 state 构造 action

Pi_05 converter 打包：

```text
observation.state[t]
= [q_left[t](6), g_left[t](1), q_right[t](6), g_right[t](1)]
shape = (14,)

action[t] = observation.state[t+1]
最后一帧 action = 最后一帧 observation.state
```

LeRobot action 因此是下一帧 absolute joint position + absolute gripper opening。图像映射为 `cam_high`、`cam_left_wrist`、`cam_right_wrist`，converter 声明为 CHW `(3,480,640)`、RGB `uint8 [0,255]`。转换阶段没有统计归一化，也没有 EE pose、rotation vector、欧拉角或 quaternion 进入 action。源码见固定 XPolicyLab 版本的 [`process_data.py`](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/scripts/process_data.py#L94-L132)。

#### 9.5.3 模型训练 chunk

horizon 同样是 50。以 observation frame `t` 为条件：

```text
a_data[k]
= [q_left[t+k+1], g_left[t+k+1],
   q_right[t+k+1], g_right[t+k+1]]
k = 0...49
```

若 episode 剩余帧数不足 50，loader 同样复制最后一个有效 action 并设置 padding 标记；下面的公式先按未越过 episode 边界的普通情况书写。

模型输入图像/state 的处理与第 9.4.4 节相同：三路 RGB resize 到 `224×224`，进入模型时为 `float32 [-1,1]`；14D 当前 state 按 checkpoint stats 归一化并补到 32D，再与 prompt 一起编码。它不会把图像相机坐标轴变成 action 坐标轴，因为当前 action 是关节量。

训练前的 `DeltaActions` 只作用于左右臂各 6 维：

```text
a_train_before_norm[k]
= [q_left[t+k+1]  - q_left[t],
   g_left[t+k+1],
   q_right[t+k+1] - q_right[t],
   g_right[t+k+1]]
```

所以它与 RoboTwin 一样：所有手臂行统一相对 chunk 第一帧 `q[t]`，不是 sequential delta；夹爪始终是 absolute `[0,1]`。随后逐维 quantile normalization 并从 14 维补到 `(50,32)`。

#### 9.5.4 模型推理 chunk 与环境 action

网络生成 `(50,32)` 的 `a_model`，理论值域无硬边界。Policy 后处理：

```text
Unnormalize
 -> 每个手臂 offset 加回同一份本次查询 q[t]
 -> 两个夹爪保持 absolute
 -> 截取前 14 维
 -> 每行拆成四个字段
```

每行字典为：

```text
left_arm_joint_state[6]     # absolute rad
left_ee_joint_state[1]      # absolute normalized gripper
right_arm_joint_state[6]    # absolute rad
right_ee_joint_state[1]     # absolute normalized gripper
```

RoboDojo 也不使用 Gym `env.step(action)` 作为策略入口；XPolicyLab 依次对全部 50 行调用 `TASK_ENV.take_action(action_dict)`，除非 episode 提前结束。chunk 中途刷新 observation，但不重新推理。

#### 9.5.5 环境和最底层 Isaac actuator

对每行 action dict：

1. 左右臂 6D 直接作为 absolute joint position target；没有再加当前 q；
2. 每个夹爪标量 clip 到 `[0,1]`，再按机器人配置的方向和 `gripper_scale` 映射成物理位置；
3. 一个标量根据 mimic 关系展开成两个物理夹爪关节 target；
4. 在一个 observation interval 内，前 80% 仿真控制步从当前 q 线性插值到目标 q，后 20% 保持目标；
5. Isaac articulation 最终接收手臂 `set_joint_position_target`、`set_joint_velocity_target`，夹爪接收 `set_joint_position_target`。position 合法范围来自机器人 articulation joint limits，velocity 来自控制序列；均不是归一化 `[-1,1]` action。

因此当前 π0.5+RoboDojo 主链路没有 SE(3) 计算。只有显式选择通用 EE action 分支时，环境才把 absolute `[xyz,qw,qx,qy,qz]` 解释为 `^W_i T_E` 并通过 IK 变成 q；仅切换评测 action type 不能让当前 joint checkpoint 自动学会这条表示。执行源码见 [`eval_env.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/src/eval_client/eval_env.py#L337-L560) 和 [`robot_manager.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env/robot_manager/robot_manager.py#L356-L405)。

### 9.6 三条链路逐层对照

| 层级 | π0.5 + LIBERO | π0.5 + RoboTwin joint | π0.5 + RoboDojo joint |
|---|---|---|---|
| 原生文件 | RLDS：8D state、7D controller action | PKL/HDF5：14D absolute q+gripper；另存的 EE `[xyz,wxyz]` 不使用 | HDF5：14D absolute q+gripper；可选 EE `[xyz,wxyz]` 不使用 |
| LeRobot action | 7D 原生 OSC command，逐维合法 `[-1,1]` | 下一帧 14D absolute q(rad)+gripper(`[0,1]`) | 下一帧 14D absolute q(rad)+gripper(`[0,1]`) |
| 训练 chunk | `(10,7)` 每行仍是原生 controller delta；不是统一首帧 offset、不是相邻差分 | `(50,14)` 原始 absolute；变换后 12D arm offset 全部相对同一 q[t]，2D grip absolute | 与 RoboTwin 相同 |
| 模型目标 | quantile normalize 后 `(10,32)`，后 25D 为 0 | quantile normalize 后 `(50,32)`，后 18D 为 0 | 同左 |
| 网络直接输出 | `(10,32)` `a_model`，理论无硬值域 | `(50,32)` `a_model`，理论无硬值域 | 同左 |
| postprocess | Unnormalize→前 7D；不做 AbsoluteActions | Unnormalize→arm 加同一 q[t]→前 14D absolute target | 同左 |
| 策略接口 | 前 5 行逐项 `env.step(u_t)` | 全 50 行逐项 `take_action(qpos14)` | 全 50 行逐项 `take_action(dict)` |
| 旋转表示 | action 中为 3D delta rotation vector；controller 内转 `R_delta^W` | 当前主链路没有旋转表示 | 当前主链路没有旋转表示 |
| 夹爪 | obs 2 个物理 qpos；action 1 个开/合方向 | 双臂各 1 个 absolute `[0,1]`，共 2D | 双臂各 1 个 absolute `[0,1]`，共 2D |
| 最底层 | MuJoCo：7D arm joint torque + 两指 position actuator target | SAPIEN：arm joint position/velocity drive target + gripper joint target | Isaac：arm joint position/velocity target + gripper joint position target |

最关键的 chunk 判定只有三句：

1. **LIBERO**：chunk 每行是将来某次 `env.step()` 的 controller delta，每行相对它实际执行时的末端 pose；不是相对 chunk 第一帧，也不是相邻 action 差分。
2. **RoboTwin joint**：LeRobot action 原本是 absolute q；模型训练时所有手臂行都减同一个 `q[t]`，推理后又加回同一个 `q[t]`；夹爪从头到尾是 absolute。
3. **RoboDojo joint**：与 RoboTwin 的 chunk 定义完全相同，区别只在环境插值和最终仿真 actuator API。

这里三条 benchmark 链路的终点都是仿真 actuator API，而不是真实 CAN/串口/EtherCAT 硬件协议。若把相同 checkpoint 部署到真机，还必须继续审计真机驱动究竟接收 torque、velocity 还是 position target；不能直接把本节的 MuJoCo/SAPIEN/Isaac 接口名称当成真机总线命令。

---

## 10. 真机部署检查清单

1. 确认 checkpoint 的 embodiment 与 action layout：关节位置、关节速度还是 EE pose。
2. 确认 action frame：world/base/root/body/wrist；不要只看 `action_dim`。
3. 确认 absolute、controller delta、chunk-relative、sequential delta 中的哪一种。
4. 确认夹爪是 `[0,1]`、`[-1,1]`、物理宽度还是电机位置。
5. 使用 checkpoint 自带的 pre/postprocessor 与 stats；不要手写第二套。
6. 完成关节零位、方向、量程、夹爪和 robot ID 标定。
7. Cartesian 执行时核对 URDF、base、TCP/tool frame 与 IK。
8. 使用显式几何/RGB-D 时再核对相机内参和 hand-eye 外参。
9. 确认图像 camera ID、左右/腕部语义、旋转、裁剪、分辨率与训练一致。
10. 真机首次运行限制速度、工作空间和动作幅度，并保留急停。

## 11. 最终判定句

如果问题特指 OpenPI 官方 `pi05_libero`：**仿真固定参考轴（标准场景与基座轴平行）下的末端增量控制 action，夹爪为开/合方向命令；不是第三人称相机系或腕部系，也不是绝对 pose。控制器内部才把它构造成绝对 `T_goal in SE(3)`。**

如果问题特指 OpenPI 官方 `pi05_droid`：**关节速度 + 绝对夹爪位置，不属于 Cartesian pose。**

如果问题特指 OpenPI `pi05_aloha`：**模型内部是相对当前 state 的关节 offset chunk，后处理恢复为绝对关节位置并下发；不属于 Cartesian pose。**

如果泛指 LeRobot 的 π0.5：**默认保留数据集原生 action 语义，通常是绝对关节位置；启用 relative action 后模型内部预测相对当前 observation state 的 offset，但官方 postprocessor 会在下发前恢复 absolute。**
