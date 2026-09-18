# Qwen-RobotManip 原始数据、动作 chunk 与部署链路审计

> 审计日期：2026-09-01（Asia/Shanghai）
> 审计对象：[Qwen-RobotManip Technical Report v2](https://arxiv.org/html/2606.17846v2)（arXiv:2606.17846，2026-06-17）与[官方仓库](https://github.com/QwenLM/Qwen-RobotManip)
> 官方仓库审计 commit：[`37f406303e13f0840a5e087c523b9b99108e7351`](https://github.com/QwenLM/Qwen-RobotManip/commit/37f406303e13f0840a5e087c523b9b99108e7351)
> 术语和链路粒度参照本地文档 [`PI05_ACTION_AND_VLA_BENCHMARK_AUDIT.md`](./PI05_ACTION_AND_VLA_BENCHMARK_AUDIT.md)。

## 0. 一页结论

1. **Qwen-RobotManip 不是只用 qpos，也不是只用 EEF；它的 80 维统一 state/action 向量兼容并可同时容纳两者。**每条机械臂分到 29 个语义槽：7 维 qpos、9 维 EEF pose、1 维平行夹爪、12 维灵巧手关节。论文以 7-DoF 单臂夹爪为例，明确说该样本会同时填 qpos、EEF 和夹爪槽；不存在或不可信的维度通过 mask 排除 loss。因此同一个未来时刻可以同时有“绝对 qpos 目标”和“相机参考系增量 EEF 目标”。部署时选 joint-control 链路或 EEF-control 链路执行。
2. **state 全部是绝对状态。**qpos state 是当前绝对关节位置；EEF state 是当前绝对位置 3 维加当前绝对朝向的 6D continuous rotation 表示。这里的 6D rotation 只是绝对 $SO(3)$ 朝向的参数化，不是 $6\times6$ 矩阵。
3. **qpos action 是绝对未来关节命令/目标。**训练 chunk 的第 $k$ 行 qpos 保存目标时刻 $\tau_k$ 的 absolute joint action；不减当前 qpos，也不减 chunk 上一行。它不一定等于 $\tau_k$ 时刻实际测得的 joint state，因为真实控制可能有跟踪误差和延迟。
4. **EEF action 是相对本次推理锚点末端的 camera-frame delta。**在锚点时刻 $t$ 选定参考相机 $C_{\mathrm{ref},t}$ 后，chunk 中每个未来目标 $E_{\tau_k}$ 都相对同一个当前末端 $E_t$ 计算，再把这个相对运动改用参考相机光学轴表达。它既不是绝对 pose，也不是相邻两行差分。
5. **EEF action 的 6 个几何数是 $\Delta\mathbf p_C\in\mathbb R^3$ 与 $\mathbf r_{\Delta,C}\in\mathbb R^3$。**旋转向量方向是转轴，模长是弧度角；先经 $\operatorname{Exp}_{SO(3)}$ 变成 $3\times3$ 旋转矩阵才能参与矩阵乘法。论文展示 $4\times4$ 矩阵是为了定义几何语义，模型并不直接预测 16 个矩阵元素。
6. **“camera-frame”指所选参考相机的三维光学坐标轴，不是像素坐标，也不固定等于第三人称相机。**单臂训练会在可用的外部相机和腕部相机中随机选择；双臂会在“两个手臂共用头部/第三人称相机”和“左右臂各用自己的腕部相机”之间随机选择。
7. **训练 action chunk 的关键规则是“同一锚点、多个未来目标”。**若未来三个 EEF 位置相对锚点分别前进 2、4、6 cm，chunk 中就是 2、4、6 cm，而不是相邻差分的 2、2、2 cm。qpos 同理保存三个绝对目标，而不是差分。
8. **state 和 action 都做分位数归一化。**论文给出每个 embodiment type、每一维的 $\left[q_{0.01},q_{0.99}\right]\mapsto[-1,1]$。这不会把绝对 state 变成 delta，只改变数值尺度。报告未说明是否硬裁剪、分母 epsilon、模型输出裁剪和精确 stats 文件格式。
9. **EEF camera-frame 链路在训练和推理都需要相机内外参标定。**没有相机标定时，论文用 auxiliary flag 切换到“robot-base relative”模式，但没有公开该 fallback 的精确公式。不能把它自行等同于某个 benchmark 的 delta controller。
10. **官方没有公开模型权重、训练代码、数据转换器或真机控制适配器。**因此本报告可以精确还原论文公式定义的 camera delta 及其逆变换，但不能诚实地给出 chunk 长度、target offset、控制频率、IK 实现、RTC 拼接策略、夹爪映射或电机协议。任何声称这些细节来自“Qwen 官方源码”的说法都没有公开证据。

---

## 1. 证据边界

本文将结论分为三级：

- **论文明确**：技术报告正文或补充实验直接写出的定义。
- **公式必然推出**：依据论文采用的公式和本文固定的坐标变换约定做出的代数逆变换；这不是猜测模型语义，但具体工程代码仍未公开。
- **未公开**：技术报告和官方仓库都没有给出的实现细节，本文不补造默认值。

截至审计日期，官方仓库根目录只有 `README.md` 和 `assets/`。README 还明确写明目前没有发布模型权重的计划；仓库没有 Python 源码、配置、数据 converter、checkpoint stats、仿真 wrapper 或真机 driver。因此下面不能像 OpenPI/LeRobot 审计那样给出文件行号级的完整代码证据，只能以[技术报告](https://arxiv.org/html/2606.17846v2)和[官方 README](https://github.com/QwenLM/Qwen-RobotManip)为准。

关键证据可直接定位到：

| 问题 | 官方报告位置 |
|---|---|
| 机器人、人类和合成数据规模 | [Section 2: Data Sources](https://arxiv.org/html/2606.17846v2#S2) |
| 人手如何变为绝对 EEF 与机器人轨迹 | [Section 2.3: Human-to-Robot](https://arxiv.org/html/2606.17846v2#S2.SS3) |
| 清洗、absolute 恢复、FK 和分位数归一化 | [Section 2.4: Curation](https://arxiv.org/html/2606.17846v2#S2.SS4) |
| 80D 布局，state absolute、joint action absolute、EEF action relative | [Section 3.2: State and Action Representation](https://arxiv.org/html/2606.17846v2#S3.SS2) |
| camera delta 公式、CaPE、标定与参考相机选择 | [Section 3.3: Unified EEF](https://arxiv.org/html/2606.17846v2#S3.SS3) |
| 历史 context chunk | [Section 3.5: In-Context Policy Adaptation](https://arxiv.org/html/2606.17846v2#S3.SS5) |
| $T\times D$ chunk、mask 与 flow matching | [Section 4: Training](https://arxiv.org/html/2606.17846v2#S4) |
| WiFi + RTC；同时也体现底层部署细节未给出 | [Section 5: Deployment](https://arxiv.org/html/2606.17846v2#S5) |
| EEF delta 相对 initial EEF pose 的补充证据 | [Section 6.4: Action-space ablation](https://arxiv.org/html/2606.17846v2#S6.SS4) |

---

## 2. 本文只使用这些坐标系和数学对象

### 2.1 坐标系

| 符号 | 固定名称 | 含义 |
|---|---|---|
| $A$ | 数据的共同绝对 pose 参考系 | 同一条样本内，EEF pose 与相机外参共同使用的三维参考系；论文没有给它一个对所有 source 都相同的原点定义 |
| $B$ | 机器人基座系 | 固定在当前机器人基座；固定基座数据常可令 $A=B$，移动机器人或某些数据源则需用 $ {}^{B}\mathbf T_A$ 连接二者 |
| $E_t$ | 锚点末端工具系 | 本次训练/推理锚点时刻 $t$ 的 EEF/TCP 坐标系 |
| $E_{\tau_k}$ | 第 $k$ 个未来目标末端系 | chunk 第 $k$ 行对应的未来目标 EEF 坐标系 |
| $C_{\mathrm{ref},t}$ | 锚点参考相机光学系 | 在时刻 $t$ 为该手臂选中的参考相机；可以是外部、头部、第三人称或本臂腕部相机 |

本文不使用含义不清的“全局相机坐标系”。第三人称相机只是一个相机，它的光学系仍记为 $C_{\mathrm{ref},t}$；共同绝对参考系 $A$、机器人基座系 $B$ 与相机光学系都通过外参连接，三者不能因相机固定不动就视为同一坐标系。

本文所说“相机光学系”只表示三维 camera frame，不擅自指定 OpenCV/OpenGL 的正轴方向。报告未公开 $+x/+y/+z$、外参是 camera-to-reference 还是 reference-to-camera 的文件 convention；实际 adapter 必须明确后再转换为本文的 $ {}^{A}\mathbf T_C$ 约定。

### 2.2 变换方向

统一使用：

- $ {}^{A}\mathbf T_{B}$：把用 $B$ 轴表示的齐次坐标变成用 $A$ 轴表示的坐标。
- $ {}^{A}\mathbf R_{B}$：$ {}^{A}\mathbf T_{B}$ 的 $3\times3$ 旋转部分，把用 $B$ 轴表示的向量变成用 $A$ 轴表示的向量。
- $ {}^{A}\mathbf p_{B}$：$B$ 原点在 $A$ 中的位置。

于是：

$$
\begin{aligned}
\mathbf p_A
  &= {}^{A}\mathbf R_B\,\mathbf p_B,\\
{}^{A}\mathbf T_C
  &= {}^{A}\mathbf T_B\,{}^{B}\mathbf T_C,\\
\left({}^{A}\mathbf T_B\right)^{-1}
  &= {}^{B}\mathbf T_A.
\end{aligned}
$$

数学对象严格区分如下：

- $\mathbf T\in SE(3)$ 是 $4\times4$ 齐次刚体变换。
- $\mathbf R\in SO(3)$ 是 $3\times3$ 旋转矩阵。
- $\mathbf r\in\mathbb R^3$ 是 rotation vector；$\mathbf r=\mathbf u\theta$，其中 $\mathbf u$ 是单位转轴，$\theta$ 是弧度角。
- $\operatorname{Exp}_{SO(3)}(\mathbf r)$ 把 rotation vector 转成 $3\times3$ 旋转矩阵。
- $\operatorname{Log}_{SO(3)}(\mathbf R)$ 把 $3\times3$ 旋转矩阵转成 rotation vector。
- 6D continuous rotation 是一个 $\mathbb R^6$ 朝向参数，需正交化后恢复 $\mathbf R\in SO(3)$；它不是矩阵，也不是欧拉角。

### 2.3 本文固定的六个数据层

| 层 | 固定名称 | 内容 |
|---|---|---|
| 1 | 原始 episode 数据 | 各开源数据集或人类视频原本记录的图像、state、action、pose、相机参数和语言 |
| 2 | 统一单帧数据 | 清洗、坐标修正并装入 80 维统一语义槽后的单帧 state/action |
| 3 | 训练 action chunk | 以锚点 $t$ 组织出的 $\mathbf a_{\mathrm{data}}\in\mathbb R^{T\times80}$ |
| 4 | 模型空间 chunk | 分位数归一化后的 $\mathbf a_{\mathrm{train}}$，以及 flow matching 推理得到的 $\mathbf a_{\mathrm{model}}$ |
| 5 | 控制器目标 | 绝对 qpos 目标，或先恢复 $ {}^{A}\mathbf T_{E_{\mathrm{goal}}}$、必要时再转换得到的基座系 $ {}^{B}\mathbf T_{E_{\mathrm{goal}}}$ |
| 6 | 最底层硬件命令 | 由平台控制器进一步生成的关节位置/速度/力矩、电流等；Qwen 报告未公开 |

---

## 3. 他们训练机器人的原始数据到底是什么

### 3.1 不是一种统一原始格式，而是三类异构来源

报告汇总的规模如下：

| 数据类别 | 组成 | 时长 |
|---|---|---:|
| 原生机器人 demonstration | 单臂 3,808 h；双臂 6,744 h；移动/人形 868 h | 约 11,420 h |
| 原始第一人称人手视频 | EgoDex 732 h；VITRA 247 h；EgoVerse 954 h | 1,933 h |
| Human-to-Robot 合成机器人 demonstration | 每条人类演示渲染到 15 种双臂平台 | 24,808 h |
| 总计 | 上述三类 | 约 38,100 h |

原生机器人来源为 OXE、AgiBotWorld-Beta、RoboMIND/RoboMIND 2.0、Galaxea Open-World、RoboCOIN、DROID、RH20T、RDT-1B 和 InternData-A1。报告只给出汇总与清洗策略，**没有发布每个 source 的原始 key 名、shape、action convention 或 converter**。

因此不能说其所有原始 action 本来都是 qpos，也不能说本来都是 EEF：

- 有的数据源记录关节 state/action。
- 有的数据源记录 EEF state/action。
- 有的数据源同时记录二者。
- 原始 action 可能是 absolute，也可能是 delta；报告的趋势对齐阶段明确说，对 delta-action 数据会先积分成 absolute 再比较。
- 相同机器人在不同数据源中还可能有不同的关节正负号、关节顺序、TCP、旋转表示和基座定义。

### 3.2 原始第一人称人手数据

- EgoDex 原生提供双手每帧 25 个关节的 $SE(3)$ 标注、标定相机和 VIO-SLAM；团队实际使用 732 h。
- VITRA 提供重建的三维手、相机轨迹与动作片段；使用 Ego4D 和 EPIC-KITCHENS 子集 247 h。
- EgoVerse 提供每只手 21 个关键点和头部 6-DoF pose；使用 954 h。
- 三种来源统一转换为 MANO 参数和每只手 21 个三维关键点。

这类数据的原始“手 pose”不是某台机器人的 qpos。只有完成手到夹爪 retargeting 和机器人 IK 后，才产生具体机器人对应的 EEF 轨迹与 qpos 轨迹。

### 3.3 Human-to-Robot 如何从人手得到 EEF 和 qpos

对每只手、每一帧，先构造绝对 EEF 目标：

$$
\begin{aligned}
\mathbf k_{\mathrm{vf}}
  &= 0.7\,\mathbf k_{\mathrm{index}}
   + 0.3\,\mathbf k_{\mathrm{middle}},\\
\mathbf p
  &= \frac{1}{2}
     \left(\mathbf k_{\mathrm{thumb}}+\mathbf k_{\mathrm{vf}}\right),\\
w
  &= \left\|\mathbf k_{\mathrm{thumb}}-\mathbf k_{\mathrm{vf}}\right\|_2,\\
\mathbf z
  &= \frac{s\left(\mathbf k_{\mathrm{thumb}}-\mathbf k_{\mathrm{vf}}\right)}{w},
     \qquad
     s=
     \begin{cases}
     +1,&\text{右手},\\
     -1,&\text{左手},
     \end{cases}\\
\mathbf d
  &= \mathbf k_{\mathrm{vf}}-\mathbf k_{\mathrm{wrist}},\\
\mathbf y
  &= \frac{\mathbf z\times\mathbf d}
           {\left\|\mathbf z\times\mathbf d\right\|_2},\\
\mathbf x
  &= \mathbf y\times\mathbf z,\\
\mathbf R
  &= \begin{bmatrix}\mathbf x&\mathbf y&\mathbf z\end{bmatrix}
     \in SO(3).
\end{aligned}
$$

其中：

- $\mathbf p\in\mathbb R^3$ 是绝对 EEF 位置。
- $\mathbf R\in SO(3)$ 是绝对夹爪朝向，三列分别是构造出的夹爪 $x/y/z$ 轴。
- $w\ge 0$ 是绝对夹爪宽度。
- 位置和宽度用 Savitzky-Golay 平滑；朝向用 Gaussian-weighted SLERP 平滑。

然后针对 15 种机器人形态分别搜索机器人基座 pose，使关键帧尽量都有可行 IK；再在 MuJoCo 中用 IK 跟踪平滑的绝对 EEF 轨迹。由此：

- 管线显式拥有绝对 EEF 目标 $(\mathbf p,\mathbf R,w)$；
- IK 会产生该机器人对应的 qpos 轨迹；
- 渲染器产生与该 qpos/EEF 同步的机器人图像。

但报告**没有公开最终合成 episode 的磁盘 schema**，也没有说明每个合成样本是否始终同时保存 qpos action 与 EEF action。只能确认这两类量在生成过程中都可获得，并且统一 80 维表示支持同时监督。

### 3.4 Human-to-Robot 的视觉和速度处理

1. SAM3 分割人手/手臂。
2. ProPainter 移除人手并补全背景。
3. MuJoCo 从标定相机视角渲染机器人和深度。
4. 使用场景深度完成遮挡关系和图像合成。
5. 为使人手演示速度接近机器人数据，在训练采样时按 source 降采样：EgoDex 保留 60% 帧率、EgoVerse 45%、VITRA 25%。

这一步会改变训练目标时刻的时间间隔，但不改变 qpos 是 absolute、EEF 是 anchor-relative camera delta 的语义。

---

## 4. 原始数据进入统一表示前做了什么

### 4.1 五阶段 state/action 清洗

1. **突变检测**：用平滑残差、二阶差分和三阶差分检测异常帧/episode。
2. **state-action 趋势对齐**：对共享关节维平滑并做互相关寻找时间 lag，再检查一阶变化方向的一致率；delta action 先积分成 absolute。
3. **极值过滤**：按 embodiment type、按维计算 $q_{0.01}/q_{0.99}$，剔除扩展分位数带之外的帧。夹爪因双峰分布不参与该极值过滤。
4. **qpos/EEF FK 一致性**：使用 URDF 和 Pinocchio 从 qpos 做 FK，与记录的 EEF pose 比较；修正关节顺序/正负号、TCP offset、旋转表示和基座假设，并把“相对左右肩”的双臂 EEF pose 变到共同参考系。
5. **基座轴与 EEF 朝向对齐**：按 dataset 施加固定旋转修正，让统一坐标约定中的正 x 轴始终指向机器人前方。

还有 instruction-video 一致性、robot reprojection 与图像一致性、黑帧/模糊/静止段等检查。

注意：这些过程说明团队确实把异构 source 变到统一几何语义，但未公开每个数据集的修正矩阵和 converter。无法从报告重建一套字节级相同的数据转换代码。

### 4.2 80 维统一 state/action 布局

$$
\begin{aligned}
80
  &= 29_{\text{左/第一臂}}
   + 29_{\text{右/第二臂}}
   + 22_{\text{公共预留}},\\
29_{\text{每臂}}
  &= 7_{\text{qpos}}
   + 9_{\text{EEF pose}}
   + 1_{\text{parallel gripper}}
   + 12_{\text{dexterous hand joints}}.
\end{aligned}
$$

| 语义槽 | state | action |
|---|---|---|
| qpos 7 | 当前测得的绝对关节位置 | 未来绝对关节命令/目标；不保证等于未来测得的 joint state |
| EEF position | 当前绝对位置 3D | 相对锚点、用参考相机轴表达的位移 3D |
| EEF rotation | 当前绝对朝向的 6D continuous rotation | 相对锚点、用参考相机轴表达的 rotation vector 3D |
| parallel gripper 1 | 当前绝对夹爪关节位置 | 最可信解释是未来绝对夹爪位置/宽度目标；报告未给 delta 公式 |
| dexterous hand 12 | 当前绝对手指关节位置 | 按“joint action absolute”的定义为未来绝对手指关节位置 |
| 公共预留 22 | 按 embodiment 定义 | 可容纳移动底盘速度等；完整语义表未公开 |

这里有一个官方报告没有补齐的布局细节：state 的 EEF rotation 用 6 个数，因此 EEF state 正好 $3+6=9$；action 的 EEF rotation 改用 3 个数，因此有效 EEF action 只有 $3+3=6$。**论文没有说明这 6 个 action 数在 9 个 EEF 槽内的精确索引，也没有说明剩余 3 槽如何 mask。**本文只固定其几何语义，不伪造索引。

### 4.3 qpos 与 EEF 是“兼容两种”，还是“二选一”

最准确的结论是：

- 统一向量设计为可同时装入 qpos 与 EEF。
- 论文明确举例：7-DoF 单臂夹爪会填 qpos、EEF 和 gripper fields。
- 有效字段都可进入 masked flow-matching loss，因此同一物理轨迹可以提供两种监督。
- 某个 source 缺少或无法可靠恢复某类字段时，用 per-dimension mask 排除。
- 实验分别报告 joint-control 和 EEF-control，说明部署可选择两条输出链路。

报告没有公布所有 dataset/fine-tune recipe 的 mask 配置，因此不能进一步断言“每一条训练样本一定同时监督两者”或“所有下游微调一定同时训练两者”。

---

## 5. Camera-frame delta 的精确定义

### 5.1 先从两个绝对 EEF pose 得到锚点局部增量

在锚点 $t$ 和未来目标 $\tau_k$，若统一数据在同一个绝对参考系 $A$ 中有：

$$
\begin{aligned}
{}^{A}\mathbf T_{E_t}
&=
\begin{bmatrix}
{}^{A}\mathbf R_{E_t} & {}^{A}\mathbf p_{E_t}\\
\mathbf 0_{1\times3} & 1
\end{bmatrix},\\[4pt]
{}^{A}\mathbf T_{E_{\tau_k}}
&=
\begin{bmatrix}
{}^{A}\mathbf R_{E_{\tau_k}} & {}^{A}\mathbf p_{E_{\tau_k}}\\
\mathbf 0_{1\times3} & 1
\end{bmatrix}.
\end{aligned}
$$

则“未来目标相对锚点末端”的旋转和位移为：

$$
\begin{aligned}
{}^{E_t}\mathbf R_{E_{\tau_k}}
&=
\left({}^{A}\mathbf R_{E_t}\right)^{\mathsf T}
{}^{A}\mathbf R_{E_{\tau_k}},\\[4pt]
{}^{E_t}\Delta\mathbf p_{E_{\tau_k}}
&=
\left({}^{A}\mathbf R_{E_t}\right)^{\mathsf T}
\left(
{}^{A}\mathbf p_{E_{\tau_k}}
-
{}^{A}\mathbf p_{E_t}
\right).
\end{aligned}
$$

第一式是相对朝向；第二式是从锚点 EEF 原点指向未来 EEF 原点的位移，并用锚点 EEF 的轴表示。

### 5.2 再把这个增量改用参考相机轴表达

锚点时刻参考相机和当前 EEF 的相对旋转为：

$$
\begin{aligned}
{}^{C_{\mathrm{ref},t}}\mathbf R_{E_t}
&=
\left({}^{A}\mathbf R_{C_{\mathrm{ref},t}}\right)^{\mathsf T}
{}^{A}\mathbf R_{E_t},\\
{}^{E_t}\mathbf R_{C_{\mathrm{ref},t}}
&=
\left(
{}^{C_{\mathrm{ref},t}}\mathbf R_{E_t}
\right)^{\mathsf T}.
\end{aligned}
$$

论文实际采用的 camera-frame delta 是：

$$
\begin{aligned}
\mathbf R_{\Delta,C}^{(k)}
&=
{}^{C_{\mathrm{ref},t}}\mathbf R_{E_t}\,
{}^{E_t}\mathbf R_{E_{\tau_k}}\,
{}^{E_t}\mathbf R_{C_{\mathrm{ref},t}},\\[4pt]
\Delta\mathbf p_C^{(k)}
&=
{}^{C_{\mathrm{ref},t}}\mathbf R_{E_t}\,
{}^{E_t}\Delta\mathbf p_{E_{\tau_k}},\\[4pt]
\mathbf r_{\Delta,C}^{(k)}
&=
\operatorname{Log}_{SO(3)}
\left(\mathbf R_{\Delta,C}^{(k)}\right).
\end{aligned}
$$

因此 chunk 第 $k$ 行有效 EEF action 是：

$$
\mathbf a_{\mathrm{EEF,data}}^{(k)}
=
\begin{bmatrix}
\Delta\mathbf p_C^{(k)}\\
\mathbf r_{\Delta,C}^{(k)}
\end{bmatrix}
\in\mathbb R^6.
$$

它是 6 个数，不是 $4\times4$ 矩阵。报告用下面的 $4\times4$ 形式只是把旋转和平移块放在一起定义：

$$
\mathbf A_{\mathrm{pose}}^{(k)}
=
\begin{bmatrix}
\mathbf R_{\Delta,C}^{(k)} & \Delta\mathbf p_C^{(k)}\\
\mathbf 0_{1\times3} & 1
\end{bmatrix}
\in SE(3).
$$

### 5.3 为什么旋转要左右各乘一次，平移只乘一次

$$
{}^{E_t}\mathbf R_{E_{\tau_k}}
$$

描述的是同一个物理相对旋转，但它的转轴坐标用 $E_t$ 轴表示。要把转轴坐标改成 $C_{\mathrm{ref},t}$ 轴，必须做基变换/共轭：

$$
\boxed{
\mathbf R_{\Delta,C}
=
{}^{C}\mathbf R_E\,
\mathbf R_{\Delta,E}\,
{}^{E}\mathbf R_C
}
$$

左右两项互为逆矩阵。这样变换的是“旋转所围绕的坐标轴”，不会把相机与 EEF 的固定朝向错误地混入物理动作。

平移 $\Delta\mathbf p$ 本来就是一个三维向量。向量换坐标只需左乘一次：

$$
\boxed{
\Delta\mathbf p_C
=
{}^{C}\mathbf R_E\,\Delta\mathbf p_E
}
$$

还可代数化简为：

$$
\boxed{
\Delta\mathbf p_C^{(k)}
=
\left({}^{A}\mathbf R_{C_{\mathrm{ref},t}}\right)^{\mathsf T}
\left(
{}^{A}\mathbf p_{E_{\tau_k}}
-
{}^{A}\mathbf p_{E_t}
\right)
}
$$

即先在共同绝对参考系 $A$ 中做“未来位置减锚点位置”，再把结果旋转到参考相机光学轴。

### 5.4 旋转向量到底是什么

若未来相对当前需要绕参考相机 $z$ 轴旋转 $10^\circ$：

$$
\begin{aligned}
\mathbf u_C
&=
\begin{bmatrix}0&0&1\end{bmatrix}^{\mathsf T},\\
\theta
&=
10^\circ
=
\frac{\pi}{18}
\approx 0.174533\ \mathrm{rad},\\
\mathbf r_{\Delta,C}
&=
\mathbf u_C\theta
=
\begin{bmatrix}0&0&0.174533\end{bmatrix}^{\mathsf T}.
\end{aligned}
$$

经 Rodrigues/指数映射得到：

$$
\begin{aligned}
\mathbf R_{\Delta,C}
&=
\operatorname{Exp}_{SO(3)}\left(\mathbf r_{\Delta,C}\right)\\
&=
\begin{bmatrix}
\cos 10^\circ & -\sin 10^\circ & 0\\
\sin 10^\circ &  \cos 10^\circ & 0\\
0              &  0              & 1
\end{bmatrix}.
\end{aligned}
$$

rotation vector 的三个数是“一个整体旋转的轴乘角”，不是依次绕 $x,y,z$ 做三次旋转的欧拉角。只有先转成 $\mathbf R_{\Delta,C}$ 后，才能按上面的公式与其他 $3\times3$ 矩阵相乘。

### 5.5 一个包含相机换轴和 chunk 的完整小例子

为简化数值，假设这个固定基座样本取 $A=B$。锚点末端位置为：

$$
{}^{B}\mathbf p_{E_t}
=
\begin{bmatrix}0.40&0.00&0.30\end{bmatrix}^{\mathsf T}\ \mathrm m.
$$

三个未来位置为：

$$
\begin{aligned}
{}^{B}\mathbf p_{E_{\tau_0}}
&=
\begin{bmatrix}0.42&0.01&0.30\end{bmatrix}^{\mathsf T}\ \mathrm m,\\
{}^{B}\mathbf p_{E_{\tau_1}}
&=
\begin{bmatrix}0.44&0.02&0.30\end{bmatrix}^{\mathsf T}\ \mathrm m,\\
{}^{B}\mathbf p_{E_{\tau_2}}
&=
\begin{bmatrix}0.46&0.03&0.30\end{bmatrix}^{\mathsf T}\ \mathrm m.
\end{aligned}
$$

参考相机相对 $B$ 绕 $z$ 轴旋转 $+90^\circ$，所以：

$$
{}^{C_{\mathrm{ref},t}}\mathbf R_B
=
\operatorname{Rot}_z(-90^\circ).
$$

三个位置增量先都减同一个锚点，再换成相机轴：

$$
\begin{aligned}
\Delta\mathbf p_C^{(0)}
&=
\begin{bmatrix}0.01&-0.02&0.00\end{bmatrix}^{\mathsf T}\ \mathrm m,\\
\Delta\mathbf p_C^{(1)}
&=
\begin{bmatrix}0.02&-0.04&0.00\end{bmatrix}^{\mathsf T}\ \mathrm m,\\
\Delta\mathbf p_C^{(2)}
&=
\begin{bmatrix}0.03&-0.06&0.00\end{bmatrix}^{\mathsf T}\ \mathrm m.
\end{aligned}
$$

若三个未来朝向分别相对锚点绕 $B$ 的 $x$ 轴旋转 $5^\circ,10^\circ,15^\circ$，那么 $B$ 的 $x$ 轴在该相机中是负 $y$ 轴，rotation-vector chunk 为：

$$
\begin{aligned}
\mathbf r_{\Delta,C}^{(0)}
&=
\begin{bmatrix}0&-0.087266&0\end{bmatrix}^{\mathsf T}\ \mathrm{rad},\\
\mathbf r_{\Delta,C}^{(1)}
&=
\begin{bmatrix}0&-0.174533&0\end{bmatrix}^{\mathsf T}\ \mathrm{rad},\\
\mathbf r_{\Delta,C}^{(2)}
&=
\begin{bmatrix}0&-0.261799&0\end{bmatrix}^{\mathsf T}\ \mathrm{rad}.
\end{aligned}
$$

这个例子体现两点：

1. 同一物理动作在相机轴下的数值可能与基座轴不同。
2. 第 2、3 行仍相对锚点 $t$，所以是累计到 $10^\circ,15^\circ$，不是相邻行差分的 $5^\circ,5^\circ$。

### 5.6 论文为何不用完整相对 $SE(3)$ 共轭

论文还比较了：

$$
{}^{C_{\mathrm{ref},t}}\mathbf T_{E_{\tau_k}}\,
{}^{E_t}\mathbf T_{C_{\mathrm{ref},t}}.
$$

这种形式会让平移项额外耦合相对旋转和相机到 EEF 的 lever-arm offset，形成更长尾的数值分布，并对外参误差更敏感。最终采用的公式只对旋转做共轭，对位移向量只做旋转换轴。

所以不要把 Qwen 的实现误写成普通的：

$$
{}^{C}\mathbf T_{E_{\mathrm{future}}}\,
{}^{E_{\mathrm{current}}}\mathbf T_C.
$$

二者平移部分不同。

---

## 6. 训练 action chunk 是怎样逐行组织的

### 6.1 一个训练样本的锚点

从 episode 选择锚点时刻 $t$，准备：

```text
o_t          当前一个或多个相机图像 + language instruction
s_data[t]    当前绝对 80D state
C_ref,t      当前手臂选择的参考相机及其标定参数
tau_0...tau_(T-1)  chunk 中 T 个目标时刻
```

报告没有公布：

- $T$ 的具体数值；
- $\tau_0$ 是 $t$、$t+1$ 还是其他 offset；
- 相邻 $\tau_k$ 的 stride；
- 目标时刻是否因不同 FPS 做重采样；
- episode 尾部是重复 padding、零 padding，还是只用 mask。

可以确认的是 action chunk shape 为 $T\times D$，统一 action 的 $D=80$，且 mask 也为 $T\times80$。

### 6.2 qpos 行的构造

对第 $k$ 个目标时刻：

$$
\boxed{
\mathbf a_{\mathrm{qpos,data}}^{(k)}
=
\mathbf q_{\mathrm{action}}^{\mathrm{abs}}(\tau_k)
}
$$

若 demonstration 在三个目标时刻记录/恢复出的 absolute joint commands 为：

$$
\begin{aligned}
\mathbf q_{\mathrm{action}}^{\mathrm{abs}}(\tau_0)
&=
\begin{bmatrix}0.10&0.20&\cdots\end{bmatrix}^{\mathsf T},\\
\mathbf q_{\mathrm{action}}^{\mathrm{abs}}(\tau_1)
&=
\begin{bmatrix}0.12&0.23&\cdots\end{bmatrix}^{\mathsf T},\\
\mathbf q_{\mathrm{action}}^{\mathrm{abs}}(\tau_2)
&=
\begin{bmatrix}0.15&0.25&\cdots\end{bmatrix}^{\mathsf T}.
\end{aligned}
$$

chunk 就保存这三个 absolute vector。它不是以下两种差分：

$$
\begin{aligned}
\mathbf q_{\mathrm{action}}^{\mathrm{abs}}(\tau_k)
-\mathbf q_{\mathrm{state}}(t)
&\qquad\text{（不是 chunk-relative qpos）},\\
\mathbf q_{\mathrm{action}}^{\mathrm{abs}}(\tau_k)
-\mathbf q_{\mathrm{action}}^{\mathrm{abs}}(\tau_{k-1})
&\qquad\text{（不是 sequential delta）}.
\end{aligned}
$$

这里的 $\mathbf q_{\mathrm{state}}(t)$ 是锚点时刻实际测得的当前关节位置，$\mathbf q_{\mathrm{action}}^{\mathrm{abs}}(\tau_k)$ 是 demonstration 的绝对关节命令/目标；报告没有声称两者在对应时刻严格相等。

### 6.3 EEF 行的构造

对同一个 $k$，锚点 pose $ {}^{A}\mathbf T_{E_t}$ 和参考相机 $C_{\mathrm{ref},t}$ 对整个 chunk 固定，只有未来目标 pose $ {}^{A}\mathbf T_{E_{\tau_k}}$ 随 $k$ 改变。第 $k$ 行由以下公式构造：

$$
\begin{aligned}
\Delta\mathbf p_C^{(k)}
&=
\left({}^{A}\mathbf R_{C_{\mathrm{ref},t}}\right)^{\mathsf T}
\left(
{}^{A}\mathbf p_{E_{\tau_k}}
-
{}^{A}\mathbf p_{E_t}
\right),\\[4pt]
\mathbf R_{\Delta,C}^{(k)}
&=
{}^{C_{\mathrm{ref},t}}\mathbf R_{E_t}\,
{}^{E_t}\mathbf R_{E_{\tau_k}}\,
{}^{E_t}\mathbf R_{C_{\mathrm{ref},t}},\\[4pt]
\mathbf r_{\Delta,C}^{(k)}
&=
\operatorname{Log}_{SO(3)}
\left(\mathbf R_{\Delta,C}^{(k)}\right).
\end{aligned}
$$

第一行是三维位移向量的换轴，第二行是相对旋转的共轭换轴；二者都不是把三维点投影成二维像素。

因此：

$$
\begin{array}{c|c}
\text{chunk 行} & \text{该行 EEF 目标的相对关系}\\
\hline
0 & E_{\tau_0}\text{ 相对 }E_t\\
1 & E_{\tau_1}\text{ 相对 }E_t\\
\vdots & \vdots\\
T-1 & E_{\tau_{T-1}}\text{ 相对 }E_t
\end{array}
$$

它不是相邻行差分：

$$
\begin{array}{c|c}
1 & E_{\tau_1}\text{ 相对 }E_{\tau_0}\\
2 & E_{\tau_2}\text{ 相对 }E_{\tau_1}
\end{array}
\qquad\text{（不是 Qwen-RobotManip 的 EEF chunk 语义）}.
$$

这一结论由三处文字共同支持：action 是 “relative deltas from the current state”；公式把 $e$ 定义为 current EEF、$e^*$ 定义为 future desired EEF；消融将基础 EEF 表示称为 “axis-angle deltas relative to the initial end-effector pose”。由于 loader 未公开，精确 target index 仍无法验证，但 anchor-relative 而非 sequential-delta 的语义是明确的。

### 6.4 双臂和多视角时怎么组织

单臂：

- 每个训练样本从可用 external view 或 wrist view 中随机选一个 $C_{\mathrm{ref},t}$。
- 该臂整个目标 chunk 都用这个参考相机轴表达。
- 每张输入图像仍用它自己的相机 pose 做 CaPE；只有 state/action token 用所选 $C_{\mathrm{ref},t}$。

双臂：每个样本随机使用下列一种方式。

1. 左右臂共享同一个头部相机或第三人称相机作为 $C_{\mathrm{ref},t}$。
2. 左臂用左腕相机，右臂用右腕相机；两臂 action 数值分别位于不同的相机光学轴下。

若腕部相机被选为参考相机，其坐标系随手臂运动。构造当前 chunk 时使用锚点时刻的 $C_{\mathrm{ref},t}$；不能拿未来腕部相机 $C_{\mathrm{ref},\tau_k}$ 分别重定义每一行，否则各行不再共享同一参考轴。

### 6.5 把各字段装进 80 维并生成 mask

每一行先初始化为 80 维零向量：

$$
\mathbf a_{\mathrm{data}}^{(k)}
=
\mathbf 0_{80}.
$$

随后把有效的 absolute qpos target、camera-frame delta EEF target，以及有效的 gripper/hand/base target 写入各自语义槽。slot mask 定义为：

$$
m_{\mathrm{slot}}^{(k,j)}
=
\begin{cases}
1,&\text{该 embodiment 的第 }j\text{ 维有语义且有可靠标注},\\
0,&\text{该维不存在、越界或标注无效}.
\end{cases}
$$

最终 mask 还会与两类 mask 做 AND：

- step-validity mask：异常帧或超过 episode 边界的时间步无 loss；一旦一步无效，后续步骤也 mask，保持因果一致。
- human hand-validity mask：人手离开相机后，从该时刻起整只手臂槽无 loss。

零填充本身不代表“目标值必须为零”；只有 mask=0 才表示这一个零不参与监督。

---

## 7. 归一化和模型训练空间

### 7.1 state 是 absolute，为什么也要归一化

absolute/delta 是几何语义；normalization 是数值尺度，二者无冲突。

例如 qpos 可能是弧度量级、EEF position 是米、rotation vector 是弧度、夹爪可能是米或设备关节单位。如果直接混合：

- 数值幅度较大的维度会更容易主导 MSE/flow loss；
- 不同机器人量程差异会让同一网络更难共享统计规律；
- 网络优化的条件数更差，梯度尺度不均衡；
- flow matching 从标准高斯噪声出发，而 target 各维尺度悬殊会使 velocity field 更难拟合。

所以当前 absolute state 也应归一化；这只把它映射到更适合学习的数值范围，不会把它变为 relative state。

### 7.2 论文公开的分位数映射

报告明确使用按 embodiment type、按维度统计的分位数映射：

$$
\left[q_{0.01},q_{0.99}\right]
\longmapsto
\left[-1,1\right].
$$

按这个映射必然对应：

$$
\boxed{
x_{\mathrm{train}}
=
2\,
\frac{x_{\mathrm{data}}-q_{0.01}}
     {q_{0.99}-q_{0.01}}
-1
}
$$

反归一化为：

$$
\boxed{
x_{\mathrm{data}}
=
q_{0.01}
+
\frac{x_{\mathrm{train}}+1}{2}
\left(q_{0.99}-q_{0.01}\right)
}
$$

其中第二式是推理必须使用的反归一化。

但上述是根据端点映射写出的代数公式；报告没有公开：

- 是否在映射前后 clip；
- 是否加 epsilon；
- stats 是否按 train split、dataset 与 embodiment 的更细层级组织；
- gripper 是否用不同映射；
- 具体 stats 文件和字段名。

Stage 3 只明确说夹爪免于“极值帧过滤”，不能据此推断夹爪完全不做 normalization。

### 7.3 camera delta 先算，还是先归一化

必须先完成几何转换，再统计/应用该 action 维的分位数：

```text
absolute EEF poses
  -> anchor-relative delta
  -> express in C_ref,t axes
  -> rotation matrix Log to rotation vector
  -> per-dimension quantile normalization
```

否则 q01/q99 对应的是另一种坐标系或另一种 action representation，推理时无法正确反归一化。

### 7.4 flow-matching 训练和模型输出

训练目标为：

$$
\begin{aligned}
\mathbf a_{\mathrm{train}}
&\in \mathbb R^{T\times80},\\
\boldsymbol\epsilon
&\sim\mathcal N(\mathbf 0,\mathbf I),\\
t_{\mathrm{flow}}
&\sim\operatorname{Beta}(1,1.5),\\
\mathbf x_{t_{\mathrm{flow}}}
&=
\left(1-t_{\mathrm{flow}}\right)\boldsymbol\epsilon
+
t_{\mathrm{flow}}\mathbf a_{\mathrm{train}},\\
\mathbf v_{\mathrm{target}}
&=
\mathbf a_{\mathrm{train}}-\boldsymbol\epsilon.
\end{aligned}
$$

DiT 学习 $\mathbf v_{\mathrm{target}}$，并用 composed mask 只计算有效时间步/维度。基础模型推理用 4 次 Euler integration 产生完整 action chunk；context 版本的消融显示 10 步更稳定。

$\mathbf a_{\mathrm{model}}$ 是连续实数张量。报告没有说输出末端有 `tanh`，因此不能保证严格落在 $[-1,1]$。$[-1,1]$ 是 $q_{0.01}$ 到 $q_{0.99}$ 的标称映射区间，不是已经证实的硬限幅。

---

## 8. qpos 链路：原始数据到最底层接口

### 8.1 训练链路

```text
原始 dataset qpos/state/action
  -> 修正 joint order、sign、fixed joint、时间 lag
  -> 必要时把原始 delta action 恢复成 absolute target
  -> 与 URDF FK / EEF logging 做一致性校验
  -> 写入 80D qpos 语义槽
  -> 每个未来 tau_k 写 absolute q_action[tau_k]
  -> 组成 T×80 action chunk
  -> 按 embodiment、按维 quantile normalize
  -> masked flow-matching loss
```

报告没有发布“原始 delta action 如何恢复到统一 absolute qpos target”的 source-specific 代码。可能依赖 action 积分、未来 state 或控制器日志，具体数据集不能一概而论。

### 8.2 推理链路

```text
模型输出 a_model[:, qpos_slots]
  -> 用 checkpoint 对应 q01/q99 反归一化
  -> 取本 embodiment 有效的 qpos 维
  -> 撤销/适配统一 joint order 与硬件 joint order
  -> 得到未来 absolute q_goal[tau_k]
  -> 送机器人 joint-position controller
  -> controller 生成位置/速度/力矩闭环命令
  -> 电机驱动
```

qpos action 本身：

- 没有 base/wrist/camera 三维坐标系问题；每一维属于一个关节。
- 对旋转关节通常是 rad，对直线关节通常是 m，但报告未声明统一单位。
- 不是 EEF pose，不需要做 $SE(3)$ 变换。
- action chunk 每行是 absolute qpos 目标，不需要累加。

joint-control 链路本身不需要把动作通过相机外参转回基座系。不过 Qwen 的视觉编码/CaPE 是否在某个具体 joint-control 部署配置中仍要求完整相机参数，官方 adapter 未公开，不能仅凭 action 是 qpos 就断言整个系统不需要相机标定。

### 8.3 夹爪为什么统一成 1 维

统一槽把 parallel gripper 定义为 1 个 joint position。这个标量可代表：

- 两指对称机构的主驱动关节位置；或
- 两指之间的 opening width。

真实夹爪可能有两个手指 qpos，但它们常通过机械耦合或 mimic joint 由一个自由度决定，因此模型只需要一个独立标量。部署 adapter 再把它映射成硬件需要的一个或两个关节命令。

报告未公开各平台的：

- 0 表示开还是闭；
- 数值单位与 min/max；
- 一维到双指的映射；
- 是否二值化；
- 夹爪控制是 position、velocity 还是 effort。

因此只能确认统一语义是“一个平行夹爪位置量”，不能给出通用的硬件数值范围。

---

## 9. EEF 链路：原始数据到最底层接口

### 9.1 训练链路

```text
原始 EEF pose / qpos + URDF
  -> 统一 TCP、旋转表示、base 约定和朝前轴
  -> 得到每帧在共同绝对参考系 A 中可靠的 absolute ^A T_E
  -> 在锚点 t 选择 C_ref,t
  -> 对所有未来 tau_k，计算相对同一个 E_t 的 delta
  -> 把 delta 换到 C_ref,t 光学轴
  -> R_delta_C 通过 Log_SO3 变成 rotation vector
  -> 写入 80D EEF action 槽
  -> 组成 T×80 action chunk
  -> quantile normalize
  -> masked flow-matching loss
```

### 9.2 模型输出的几何含义

反归一化后，第 $k$ 行有效 EEF 输出是：

$$
\begin{aligned}
\Delta\mathbf p_C^{(k)}
&\in\mathbb R^3
&&\text{（m，或统一数据实际使用的长度单位）},\\
\mathbf r_{\Delta,C}^{(k)}
&\in\mathbb R^3
&&\text{（rotation vector，rad）},\\
g_{\mathrm{goal}}^{(k)}
&\in\mathbb R
&&\text{（最可信解释为平行夹爪绝对位置/宽度目标）}.
\end{aligned}
$$

它们都相对锚点时刻 $t$，不是“执行第 $k$ 行时再相对当时实际末端”的 controller delta。Qwen action chunk 与 LIBERO `env.step()` 的逐步 OSC delta 是不同的数据层，若要接 LIBERO 必须由 adapter 转换。

### 9.3 从 camera delta 恢复未来绝对 EEF pose

先反归一化，再把 rotation vector 恢复为旋转矩阵：

$$
\mathbf R_{\Delta,C}^{(k)}
=
\operatorname{Exp}_{SO(3)}
\left(\mathbf r_{\Delta,C}^{(k)}\right).
$$

然后从相机轴换回锚点末端轴：

$$
\begin{aligned}
{}^{E_t}\mathbf R_{E_{\tau_k}}
&=
{}^{E_t}\mathbf R_{C_{\mathrm{ref},t}}\,
\mathbf R_{\Delta,C}^{(k)}\,
{}^{C_{\mathrm{ref},t}}\mathbf R_{E_t},\\[4pt]
{}^{E_t}\Delta\mathbf p_{E_{\tau_k}}
&=
{}^{E_t}\mathbf R_{C_{\mathrm{ref},t}}\,
\Delta\mathbf p_C^{(k)}.
\end{aligned}
$$

再用锚点绝对 pose 得到共同参考系 $A$ 中的绝对目标：

$$
\begin{aligned}
{}^{A}\mathbf R_{E_{\mathrm{goal}}}^{(k)}
&=
{}^{A}\mathbf R_{E_t}\,
{}^{E_t}\mathbf R_{E_{\tau_k}},\\[4pt]
{}^{A}\mathbf p_{E_{\mathrm{goal}}}^{(k)}
&=
{}^{A}\mathbf p_{E_t}
+
{}^{A}\mathbf R_{E_t}\,
{}^{E_t}\Delta\mathbf p_{E_{\tau_k}},\\[4pt]
{}^{A}\mathbf T_{E_{\mathrm{goal}}}^{(k)}
&=
\begin{bmatrix}
{}^{A}\mathbf R_{E_{\mathrm{goal}}}^{(k)}
&
{}^{A}\mathbf p_{E_{\mathrm{goal}}}^{(k)}
\\
\mathbf 0_{1\times3} & 1
\end{bmatrix}.
\end{aligned}
$$

位置也可直接写成：

$$
\boxed{
{}^{A}\mathbf p_{E_{\mathrm{goal}}}^{(k)}
=
{}^{A}\mathbf p_{E_t}
+
{}^{A}\mathbf R_{C_{\mathrm{ref},t}}\,
\Delta\mathbf p_C^{(k)}
}
$$

若控制器使用机器人基座系且 $A\ne B$，再做一次明确的绝对 pose 变换：

$$
\boxed{
{}^{B}\mathbf T_{E_{\mathrm{goal}}}^{(k)}
=
{}^{B}\mathbf T_A\,
{}^{A}\mathbf T_{E_{\mathrm{goal}}}^{(k)}
}
$$

以上是论文 action 定义的代数逆变换，属于“公式必然推出”。官方没有公开实际部署函数名或数值容错实现。

### 9.4 恢复 absolute pose 后如何接控制器

取决于机器人/仿真环境的底层接口：

1. **接口直接收 absolute EEF pose**：将目标转换到接口指定的参考系；若接口收基座系，则发送 $ {}^{B}\mathbf T_{E_{\mathrm{goal}}}^{(k)}$ 和 gripper target。
2. **接口收 EEF delta**：必须用当前控制器规定的参考轴、左右乘约定和 scale，把 absolute goal 再转换成该 controller 的 delta；不能直接发送 $\Delta\mathbf p_C^{(k)}$ 和 $\mathbf r_{\Delta,C}^{(k)}$。
3. **接口只收 qpos**：用机器人 URDF/运动学和当前 qpos 对 $ {}^{B}\mathbf T_{E_{\mathrm{goal}}}^{(k)}$ 做 IK，得到 $\mathbf q_{\mathrm{goal}}^{(k)}$，再送 joint-position controller。
4. **接口收 joint velocity/torque**：还需轨迹插值、Jacobian/IK/OSC 等控制层把 pose target 变成对应命令。

报告只说真实部署在远程服务器推理、通过 WiFi 传 observation/action，并用 RTC 异步生成下一 chunk。它没有公开：

- EEF controller 是 IK、OSC 还是厂商 Cartesian servo；
- goal pose 的插值方式和控制频率；
- 每次执行 chunk 的多少行；
- 是否依据新 observation 截断旧 chunk；
- IK seed、joint limit、collision check；
- 最终给电机的是 position、velocity、torque 还是 current；
- 各真机的 gripper mapping。

因此本文不能把某一种控制器实现冒充成 Qwen 官方实现。

### 9.5 与 LIBERO 风格 controller delta 的区别

Qwen camera delta 的第 $k$ 行：

- 相对“生成整个 chunk 时的 $E_t$”定义。
- 参考轴是“生成整个 chunk 时选中的 $C_{\mathrm{ref},t}$”。

LIBERO OSC 的第 $k$ 个 `env.step(u_k)`：

- 通常相对“第 $k$ 行实际执行时的 current EEF”重新定义。
- 参考轴是 LIBERO controller 规定的仿真固定轴。

所以 adapter 不能把 Qwen 的 6D 输出逐行原样当作 LIBERO 6D action。稳妥转换是先按 9.3 恢复每行 absolute goal、再转换到 LIBERO controller 使用的绝对参考系，然后在每个实际执行时刻根据 controller 的 action convention 计算它需要的 delta。

---

## 10. 两条完整链路逐层对照

| 数据层 | qpos 路径 | EEF 路径 |
|---|---|---|
| 原始 episode | source-specific joint state/action；可能 absolute 或 delta | source-specific EEF pose/action，或由 qpos+URDF 做 FK；旋转原格式可能不同 |
| 统一单帧 state | 当前 absolute qpos | 当前 absolute position + absolute 6D rotation representation |
| 训练 chunk 第 k 行 | future absolute $\mathbf q_{\mathrm{action}}^{\mathrm{abs}}(\tau_k)$ | $E_{\tau_k}$ 相对同一 $E_t$ 的 camera-frame position delta + rotation vector |
| 是否相邻差分 | 否 | 否 |
| 是否相对 chunk 锚点 | 否，qpos 是 absolute | 是，所有行相对 $E_t$ |
| 模型训练值 | quantile-normalized continuous value | quantile-normalized continuous value |
| 模型原始输出值域 | 实数；报告未证明硬限幅 | 实数；报告未证明硬限幅 |
| 反归一化后 | absolute qpos target | camera-frame delta position + rotation vector |
| 控制前转换 | joint order/sign/hardware mapping | $\operatorname{Exp}_{SO(3)}$ + 外参逆换轴 + 与锚点 pose 组合，恢复 $ {}^{A}\mathbf T_{E_{\mathrm{goal}}}$；需要时再变到 $ {}^{B}\mathbf T_{E_{\mathrm{goal}}}$ |
| 控制器目标 | absolute joint position | absolute Cartesian pose，或再经 IK 变 absolute joint position |
| 最底层硬件命令 | 未公开 | 未公开 |

最重要的关系可压缩为：

$$
\begin{aligned}
\mathbf a_{\mathrm{qpos,data}}^{(k)}
&=
\mathbf q_{\mathrm{action}}^{\mathrm{abs}}(\tau_k),
\\[4pt]
\mathbf a_{\mathrm{EEF,data}}^{(k)}
&=
\operatorname{CameraDelta}
\left(
E_t\rightarrow E_{\tau_k};
C_{\mathrm{ref},t}
\right).
\end{aligned}
$$

其中每个 $k$ 的 qpos 都是目标时刻 $\tau_k$ 的 absolute joint command；每个 EEF 目标都相对同一个锚点 $E_t$，并用同一个所选参考相机 $C_{\mathrm{ref},t}$ 的轴表达。

---

## 11. Context chunk 不等于当前要预测的 target chunk

Qwen-RobotManip-Context 还使用历史执行上下文。一个历史 context chunk 是：

$$
\left(\mathbf o_h,\mathbf s_h,\mathbf a_h\right).
$$

其中：

- $\mathbf o_h$：该历史决策点看到的图像；
- $\mathbf s_h$：该历史决策点的 absolute state；
- $\mathbf a_h$：从该点实际执行过的 $K$-step action sequence。

历史 state/action 经两个 MLP 变成 token；历史图像与当前图像一起进入视觉编码。训练时不会总取紧邻当前的最近历史，而是在 episode 内随机位置采样 context window，防止模型只复制最近 action。部署时才使用最近 $H$ 个 rolling context chunks。

这和第 6 节的当前监督 target chunk 是两件事：

- target chunk 是模型现在要预测的未来动作。
- context chunks 是已经执行过的历史 observation-state-action，用来识别当前机器人的运动风格。

论文使用 $K$ 表示执行 chunk 的步数，又用 $K'$ 表示 MLP 压缩后的 action tokens 数量，但没有公开 $K,K',H$ 的具体值。episode 开始时历史不足会用全零 placeholder；报告指出这可能导致 context 模型起步犹豫。

---

## 12. 标定要求

### 12.1 Camera-frame EEF 模式必须具备的量

论文明确说 camera-frame delta 在训练和推理都需要 calibrated camera intrinsics and extrinsics。至少需要：

- 每个输入相机的内参 $\mathbf K$，供 image-patch 的视场编码使用；
- 每个相机相对共同绝对参考系的外参，例如 $ {}^{A}\mathbf T_C$；
- 当前 EEF pose $ {}^{A}\mathbf T_{E_t}$；
- 共同绝对参考系与机器人基座不同时的 $ {}^{B}\mathbf T_A$；
- wrist camera 场景下可靠的手眼外参 $ {}^{E}\mathbf T_{C_w}$，再结合当前 FK 得到 $ {}^{A}\mathbf T_{C_w,t}$；
- 正确的 TCP 定义和机器人运动学。

若这些量不准：

- 同一物理位移会被旋转到错误的相机轴；
- rotation conjugation 会产生错误转轴；
- 推理逆变换得到的 absolute EEF goal 会系统性偏移；
- CaPE 中图像 token 与 action token 的几何关系也会错误。

### 12.2 只有第三人称图像并不意味着“第三人称相机就是 world/base”

必须有外参把 $C_{\mathrm{ref}}$ 与共同绝对参考系 $A$ 连接；若底层接口用 $B$，还需知道 $A$ 与 $B$ 的关系：

$$
{}^{A}\mathbf T_{C_{\mathrm{ref}}},
\qquad
{}^{B}\mathbf T_A.
$$

相机固定不动只意味着该外参在 episode 中可保持常量，不意味着其原点和坐标轴天然等于 $B$。

### 12.3 没有标定时的 fallback

论文用一个 binary auxiliary flag 表示当前样本是否有 calibrated camera parameters：

- 有：预测 camera-frame delta。
- 无：切换为 robot-base relative mode。

但是报告没有写 robot-base relative 的公式、旋转左/右乘约定、chunk anchor 细节或部署逆变换。不能把 fallback 擅自写成 LIBERO OSC、普通 $\mathbf T_t^{-1}\mathbf T_{\mathrm{future}}$ 或逐步 controller delta。

---

## 13. 官方公开材料无法回答的清单

以下细节在复现或接真机前必须向作者获得代码/配置，或自行定义并重新训练/适配：

1. action horizon $T$、目标 offset、stride、控制 FPS。
2. episode 末尾 padding 与 validity-mask 的实际实现。
3. 80 维向量的精确索引，特别是 9D EEF action 槽中 3D rotation vector 的位置。
4. 每个原始 dataset 的字段、单位、action convention 和 converter。
5. 哪些 source/sample 同时监督 qpos 与 EEF，哪些只监督一种。
6. q01/q99 stats 的文件格式、分组键、epsilon、clip 和 gripper 处理。
7. 模型输出是否 clip、异常值安全限制和 joint/Cartesian workspace limit。
8. camera calibration 的文件格式、光学轴 convention 和 timestamp 同步。
9. robot-base relative fallback 的完整数学定义。
10. qpos 与 EEF 两种 execution mode 的选择开关和 output mask。
11. 各 benchmark 的 action adapter。
12. 真机 IK/OSC/Cartesian servo、控制频率、插值和安全控制器。
13. RTC 如何衔接旧/新 chunk、每次实际执行几行、是否 temporal ensemble。
14. 平行夹爪的一维数值范围、开闭方向和硬件 mapping。
15. context 模型的 $K/K'/H$ 和零 padding 实现。

---

## 14. 最终判定句

**Qwen-RobotManip 同时兼容 qpos 与 EEF：qpos 路径预测 future absolute joint commands/targets；EEF 路径预测相对当前 chunk 锚点 EEF、并用所选参考相机光学轴表达的 future delta position 与 rotation vector。一个 EEF action chunk 的每一行都相对同一个锚点 $E_t$，不是相邻 action 差分。推理后必须先按训练 stats 反归一化，再用相机外参与当前绝对 EEF pose 把 camera delta 恢复为共同绝对参考系下的 EEF goal，必要时再转换成基座系 goal，之后才可能送 Cartesian controller 或 IK。Camera-frame 模式必须标定；真正的真机 controller、chunk 时序和硬件命令因官方未发布源码而仍不可验证。**
