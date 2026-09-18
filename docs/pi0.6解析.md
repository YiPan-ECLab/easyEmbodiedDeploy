# π0.6 与 RECAP 解析

## 1. 先区分两个模型

- **π0.6**：从 π0.5 升级而来的基础 VLA。
- **π\*0.6**：在 π0.6 上增加 advantage conditioning，并使用 RECAP 学习真实执行经验。

## 2. π0.6 的结构

相对 π0.5，π0.6 的主要变化是：

- VLM 换成 **Gemma 3 4B**，其中包含约 400M 参数的视觉编码器；
- flow-matching action expert 增大到约 **860M**；
- 预训练数据增加了更多机器人平台；
- 使用 **KI（Knowledge Insulation）**训练方式。

数据流为：

```text
相机图像 + 机器人状态 + 任务指令
                 │
                 ▼
          Gemma 3 VLM
          ├─ 预测下一高层子任务
          ├─ 预测 FAST 离散动作 token
          └─ 输出视觉语言特征
                       │
                       ▼
             860M flow action expert
                       │
                       ▼
                连续 action chunk
```

同一段真实动作具有两种训练目标：

1. VLM 使用交叉熵预测 FAST 离散动作 token；
2. action expert 使用 flow matching 预测连续 action chunk。

action expert 可以读取 VLM 特征，但 **flow loss 不反向传播到 VLM**。因此，VLM 主要由稳定的离散 token 损失训练，避免连续动作损失破坏其视觉和语言能力。这就是 KI 的核心。

## 3. π*0.6 的训练损失

先定义本节唯一使用的符号：

- $o_t$：时刻 $t$ 的相机图像和机器人状态；
- $\ell$：总任务指令；
- $\hat{\ell}$：模型预测的下一个高层子任务；
- $I_t$：advantage 条件，值为 `positive` 或 `negative`；
- $a_{t:t+H}$：真值连续 action chunk；
- $a^{\mathrm{FAST}}_{t:t+H}$：同一 action chunk 经 FAST 离散化得到的 token 序列。

π*0.6 不是只用一个「强化学习 loss」替换原有 loss。RECAP 先用 value model 产生 $I_t$，然后仍用 KI 的监督学习损失训练 advantage-conditioned policy。

### 3.1 FAST 离散动作 token loss

VLM 根据观测、指令、子任务和 advantage 条件，自回归预测 FAST token：

$$
\mathcal{L}_{\mathrm{FAST}}
=
-\log p_{\theta}\!\left(
a^{\mathrm{FAST}}_{t:t+H}
\mid I_t,o_t,\ell,\hat{\ell}
\right).
$$

实际计算时，它就是对 action chunk 的每个真值 FAST token 计算 next-token cross-entropy，再对有效 token 求和或平均。

### 3.2 连续动作 flow-matching loss

对真值连续 action chunk $a_{t:t+H}$ 采样噪声和 flow 时间：

$$
\omega\sim\mathcal{N}(0,I),
\qquad
\eta\sim\mathcal{U}(0,1),
$$

$$
a^{\eta,\omega}_{t:t+H}
=
\eta a_{t:t+H}+(1-\eta)\omega.
$$

action expert $f_{\theta}$ 按论文的时间方向定义回归 flow，其目标向量为 $\omega-a_{t:t+H}$：

$$
\mathcal{L}_{\mathrm{flow}}
=
\mathbb{E}_{\eta,\omega}
\left[
\alpha_{\eta}
\left\|
\omega-a_{t:t+H}
-f_{\theta}\!\left(
a^{\eta,\omega}_{t:t+H},I_t,o_t,\ell,\hat{\ell}
\right)
\right\|_2^2
\right].
$$

$\alpha_{\eta}$ 是 flow loss 的权重，可以随噪声时间 $\eta$ 变化。这个 $\alpha_{\eta}$ 不是 advantage 阈值，也不是正负标签的权重。

### 3.3 完整 policy loss

若当前样本还提供下一子任务真值，则另有文本 next-token loss：

$$
\mathcal{L}_{\mathrm{subtask}}
=
-\log p_{\theta}\!\left(\hat{\ell}\mid o_t,\ell\right).
$$

因此，π*0.6 的 policy 训练目标可概括为：

$$
\boxed{
\mathcal{L}_{\mathrm{policy}}
=
\mathcal{L}_{\mathrm{subtask}}
+\mathcal{L}_{\mathrm{FAST}}
+\mathcal{L}_{\mathrm{flow}}
}
$$

其中与低层动作直接相关的是：

$$
\boxed{
\mathcal{L}_{\mathrm{action}}
=
\mathcal{L}_{\mathrm{FAST}}
+\mathcal{L}_{\mathrm{flow}}
}
$$

FAST 分支和 flow 分支预测的是同一个 action chunk，但两者独立预测；action expert 不会读取真值 FAST token。同时，flow 分支到 VLM 主干之间使用 stop-gradient，所以 $\mathcal{L}_{\mathrm{flow}}$ 不更新 VLM 主干，$\mathcal{L}_{\mathrm{FAST}}$ 则会训练 VLM。

### 3.4 Value model loss

RECAP 还会单独训练 value model。它将经验回报 $R_t$ 离散成 $B=201$ 个 value bins，并使用交叉熵：

$$
\mathcal{L}_{V}
=
\mathbb{E}_{\tau\sim D}
\left[
\sum_{o_t\in\tau}
H\!\left(R_t^B(\tau),p_{\phi}(V\mid o_t,\ell)\right)
\right].
$$

$\mathcal{L}_{V}$ 只训练独立的 value model，不是直接加到上面的 policy loss 中。value model 输出用来计算 advantage，再生成 $I_t$。

> **一句话结论：**value model 不预测单步奖励 $r_t$，而是预测「从第 $t$ 步开始到 episode 结束的累计回报 $R_t$」落在 201 个 value bins 中各个 bin 的概率。

#### 公式中各项的具体含义

$R_t^B(\tau)$ 和 $p_{\phi}(V\mid o_t,\ell)$ 的含义分别是：

1. 对轨迹 $\tau$ 中的第 $t$ 步，先计算从当前步到 episode 结束的累计回报：

$$
R_t(\tau)=\sum_{k=t}^{T}r_k.
$$

2. 将 $R_t(\tau)$ 归一化，映射到 201 个 value bins 中的一个，并转换成 one-hot 分类标签 $R_t^B(\tau)$。因此，$R_t^B(\tau)$ 不是一个新的奖励，而是「真实累计回报所在 bin」的 one-hot 表示。

3. $p_{\phi}(V\mid o_t,\ell)$ 是 value model 输出的 201 维概率向量：

$$
p_{\phi}(V\mid o_t,\ell)
=
[p_1,p_2,\ldots,p_{201}],
$$

其中 $p_b$ 表示「从当前第 $t$ 步开始的累计回报落入第 $b$ 个 value bin」的概率。它是**累计回报的概率分布**，不是单步奖励 $r_t$ 的概率分布。

交叉熵就是比较真实 one-hot 标签与模型预测概率向量：

$$
H\!\left(
\underbrace{R_t^B(\tau)}_{\text{真实累计回报的 one-hot 标签}},
\underbrace{p_{\phi}(V\mid o_t,\ell)}_{\text{预测的累计回报概率向量}}
\right).
$$

需要实际的标量 value 时，再用每个 bin 的概率对其代表值 $v_b$ 加权求和：

$$
V(o_t,\ell)
=
\sum_{b=1}^{201}p_{\phi}(V=b\mid o_t,\ell)v_b.
$$

### 3.5 这些 loss 出现在哪些阶段

| 阶段 | RECAP / advantage conditioning | FAST token loss | Flow loss |
|---|---|---|---|
| π*0.6 通用预训练 | 有；先训练 value model，再为演示数据计算 $I_t$ | 有 | 有 |
| 目标任务 SFT | 属于 RECAP 流程；演示数据的 $I_t$ 固定为 `positive` | 有 | 有 |
| 目标任务 RECAP 迭代微调 | 有；使用演示、自主 rollout 和人工纠正重新计算 $I_t$ | 有 | 有 |

所以，对论文中的 π*0.6 可以明确总结为：

> RECAP 贯穿通用预训练和下游任务微调；FAST token loss 与 flow-matching loss 也都在这些 policy 训练阶段保留。RECAP 改变的是动作生成的条件 $I_t$ 和训练数据，而不是用一个新的 RL loss 取代 FAST/flow loss。

## 4. RECAP 是什么

RECAP 全称为：

> Reinforcement Learning with Experience and Corrections via Advantage-conditioned Policies

它不是直接使用 PPO 更新策略，而是：

1. 使用 value model 判断轨迹中每一步的质量；
2. 将动作标记为 `Advantage: positive` 或 `Advantage: negative`；
3. 训练 VLA 在指定 advantage 条件下生成动作；
4. 推理时要求模型生成 `positive` 条件对应的动作。

## 5. Value model

value model 的输入是当前图像、机器人状态和任务指令，输出当前状态的预期回报。论文采用的稀疏奖励可以简写为：

$$
r_t=
\begin{cases}
0, & t=T\ \text{且任务成功} \\
-C_{\mathrm{fail}}, & t=T\ \text{且任务失败} \\
-1, & \text{其他情况}
\end{cases}
$$

这里：

- $r_t$ 是执行第 $t$ 步动作后得到的单步奖励；
- $T$ 是该 episode 实际终止时的最后一个有效时间步，不一定等于预设的最大步数；
- $C_{\mathrm{fail}}$ 是人为设置的较大正数，用来保证失败轨迹的 value 明显低于成功轨迹；论文未公布其具体数值；
- $V(o_t)$ 是从状态 $o_t$ 开始，继续按照当前数据策略行动时，未来累计奖励的期望估计：

$$
V^{\pi}(o_t)=
\mathbb{E}_{\pi}\left[
\sum_{k=t}^{T}r_k
\;\middle|\;o_t
\right]
$$

这里写的是 RECAP 实际采用的未折扣 empirical return，因此公式中没有额外的折扣因子 $\gamma$。

episode 可能因为以下原因终止：

| 终止原因 | 常见判定方式 |
|---|---|
| 达到最大时间或最大步数 | 程序自动判定 |
| 碰撞、超力矩或越过安全限制 | 机器人安全系统自动判定 |
| 物体掉出工作区 | 视觉/传感器检测，无法可靠检测时由人工判断 |
| 任务成功 | 任务检测器、环境传感器或人工标注 |
| 进入无法恢复的状态 | 通常由人工判断并提前终止 |
| 人工按下停止按钮 | 直接记录为人工终止 |

RECAP 不要求人工逐帧设置 $r_t$。最低要求是为整条 episode 提供 `success=True/False`，并记录终止时间步 $T$；程序随后自动生成每一步奖励、return、value 训练标签和 advantage。论文中的真实系统仍需要人工参与结果标注、必要时的纠正和场景复位，因此不是完全自动的强化学习系统。

不同任务的 episode 长度差异很大，因此论文会按每个任务的最大 episode 长度缩放 value，使其落在 $(-1,0)$ 范围内。$C_{\mathrm{fail}}$ 的作用是区分成功与失败，而不是作为最终的 advantage 分类阈值。

因此：

- 成功所需剩余步骤越少，状态价值越高；
- 失败轨迹具有很低的状态价值；
- value model 实际学习的是经过归一化的“距离成功还有多远”。

RECAP 后训练不是只看下一帧，而是使用 $N=50$ 的多步 advantage：

$$
A_t=
\sum_{k=t}^{t+N-1}r_k
+V(o_{t+N})-V(o_t),
\qquad N=50
$$

其中：

- $V(o_t)$：执行动作前的状态价值；
- $\sum_{k=t}^{t+N-1}r_k$：未来 50 步实际获得的累计奖励；
- $V(o_{t+N})$：50 步后状态的预期价值；
- $A_t$：从 $t$ 开始的动作 chunk 相对当前策略平均行为的改进程度。

随后按照每个任务自己的阈值 $\epsilon_\ell$ 二值化：

$$
I_t=\mathbb{1}\!\left[A_t>\epsilon_\ell\right]
$$

阈值不是固定为 0：

- 预训练时选择 $\epsilon_\ell$，使每个任务约 30% 的示教样本标为 `positive`；
- RECAP 微调时通常使约 40% 的 evaluation rollout 样本标为 `positive`；
- T-shirt/shorts 折叠任务使用更严格的阈值，只有约 10% 的数据标为 `positive`。

因此，`negative` 只表示 $A_t$ 没有超过该任务的筛选阈值，不一定表示 $A_t<0$。

标签附加在以时间 $t$ 为锚点的 action chunk 上，而不是给整条 episode 一个统一的 advantage 标签。同一条 episode 中可以同时存在 `positive` 和 `negative` chunk，例如失败前的错误动作可能是 `negative`，中途有效的恢复动作仍可能是 `positive`。

训练时随机丢弃 advantage 条件的概率为 30%：其余样本携带 `positive/negative` 条件，模型因此同时学习条件策略和无条件策略。人工介入产生的纠正 chunk 强制标记为 `positive`，推理时则指定 `Advantage: positive`，也可以通过 CFG 进一步强化该条件。

$A_t$ 较高表示该 chunk 让任务进展好于预期，较低表示它没有明显推进任务或使状态变差。

### Positive/negative 如何影响训练

`positive/negative` 是模型的条件输入，不是 loss 的正负号。每个训练样本可以写成：

```text
输入：图像 + 状态 + 任务指令 + advantage 标签
目标：该时刻真实执行的 action chunk
```

模型分别学习两个条件动作分布：

$$
\pi_\theta\!\left(a_{t:t+H}\mid o_t,\ell,I_t=\mathrm{positive}\right)
$$

$$
\pi_\theta\!\left(a_{t:t+H}\mid o_t,\ell,I_t=\mathrm{negative}\right)
$$

这里不是训练三套模型权重，而是**同一套参数 $\theta$ 配合三类输入条件**：

| 模式 | Advantage 输入 | 模型权重 |
|---|---|---|
| Positive | `Advantage: positive` | 共享同一套 $\theta$ |
| Negative | `Advantage: negative` | 共享同一套 $\theta$ |
| Unconditional | 不提供 advantage 标签 | 共享同一套 $\theta$ |

所谓三个动作分布，是同一个神经网络在三种输入条件下产生的条件分布，不是三套 checkpoint。

negative 样本仍使用真实动作正常计算 FAST token loss 和 flow-matching loss。它不会产生反向惩罚，也不会被丢弃；模型学习的是“这个动作属于 negative 条件下的行为”。经过训练后：

- `positive` 条件对应更有效、更可能推进任务的动作分布；
- `negative` 条件对应停滞、失误或质量较低的动作分布；
- 推理时指定 `Advantage: positive`，使模型只生成前一种行为。

30% condition dropout 让同一模型额外学习无条件分布：

$$
\pi_\theta\!\left(a_{t:t+H}\mid o_t,\ell\right)
$$

因此推理时可以使用 classifier-free guidance（CFG），将连续动作进一步推向 positive 分布：

$$
f_{\mathrm{guided}}
=
f_{\mathrm{uncond}}
+\beta\left(f_{\mathrm{positive}}-f_{\mathrm{uncond}}\right)
$$

其中 $\beta$ 是推理时的引导强度。RECAP 的策略提升并不是“惩罚 negative 动作”，而是通过条件建模分开学习好坏行为，并在推理时主动选择 positive 条件。

## 6. RECAP 训练流程

```text
大量多任务示教
    ↓
训练通用 value model
    ↓
计算 advantage 并训练 π*0.6
    ↓
使用目标任务示教进行 SFT
    ↓
部署到真机收集数据
    ├─ 成功的自主轨迹
    ├─ 失败的自主轨迹
    └─ 人工接管纠正
    ↓
使用累计数据更新 value model
    ↓
重新计算 positive / negative
    ↓
更新目标任务 policy
    ↓
按需重复
```

人工接管产生的纠正动作直接标记为 `positive`。自主轨迹无论成功还是失败都可以用于训练，因为 value model 会区分其中动作的质量。

## 7. 主要来源

- [π\*0.6 / RECAP 官方论文](https://www.physicalintelligence.company/download/pistar06.pdf)
