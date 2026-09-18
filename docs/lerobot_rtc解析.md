# LeRobot 两种 RTC：运行机制速查

> 适用快照：本地 LeRobot `4aaff99be4a1d81568c08c8f0296b41b40c99ec4`，核对日期 2026-09-04。  
> 本文先讲清运行过程，不展开去噪、梯度和损失函数的数学推导；需要时再补充。

## 1. 一句话区别

| 模式 | 连续性如何实现 | 是否要 RTC 专用训练 | 推理额外开销 |
|---|---|---:|---:|
| `guided` | 推理时让新 chunk 的开头逐渐摆脱旧 chunk | 否，普通 flow-matching checkpoint 可用 | 较高 |
| `trained` | 模型训练时学会“接着一段已确定的动作继续生成” | 是，目前要求兼容的 Pi0.5 checkpoint | 较低 |

两种模式共用同一个异步执行框架：控制线程持续取动作，后台线程在旧 chunk 尚未执行完时生成新 chunk。

## 2. 先分清四个容易混淆的量

```text
H = chunk_size：模型一次生成的动作总数
q = queue_threshold：执行队列剩余不超过 q 个动作时，后台开始推理
d = inference_delay：一次推理期间经过了多少个控制周期
s = 本文对 execution_horizon 的简称，不是 LeRobot 源码变量名
```

`s` 既不是“开始推理时队列剩余的动作数”，也不是“每次加入队列的动作数”。

```text
q 决定什么时候开始推理
s 决定 Guided RTC 中旧 chunk 影响新 chunk 多长
H - d_real 才是新 chunk 最终进入执行队列的动作数
```

以下统一使用一组与当前源码默认值一致的例子：

```text
控制频率              = 30 Hz
H = chunk_size         = 50
q = queue_threshold    = 30
s = execution_horizon  = 10
prefix_attention_schedule = LINEAR（当前默认值）
一次推理耗时           = 200 ms
d = 200 ms 内的控制步数 = 6
```

`H=50` 是 Pi0、Pi0.5 和 SmolVLA 当前配置的默认值；实际部署时以所加载 checkpoint 的配置为准。

## 3. Guided RTC：完整运行过程

继续使用上面的固定例子。

```text
1. 执行队列还剩 30 个动作
                  |
2. 因为 queue_size <= q=30，后台开始下一次推理
                  |
3. 后台保存旧队列接下来的前 s=10 个动作，供 Guided RTC 参考
                  |
4. 推理持续 200 ms；与此同时机器人继续执行旧队列中的 6 个动作
   此时旧队列还剩 24 个动作
                  |
5. 模型生成包含 H=50 个动作的新 chunk
                  |
6. 新 chunk 的前 10 个位置受到旧计划影响：
   位置 0～5 强烈跟随旧计划，位置 6～9 逐渐减弱，位置 10 以后不受影响
                  |
7. 新 chunk 的位置 0～5 对应已经过去的 200 ms，因此删除
                  |
8. 丢弃旧队列剩余的 24 个动作，用新 chunk 的位置 6～49 替换整个队列
                  |
9. 新执行队列共有 H-d = 50-6 = 44 个动作
```

注意第 8 步：LeRobot 不是把 `s=10` 个新动作追加到旧队列后面，而是用 `44` 个仍有效的新动作替换旧队列。旧队列只在生成新 chunk 时作为连续性参考。

Guided RTC 中各参数的含义：

```text
q=30：队列剩余不超过30个动作时开始推理
s=10：旧计划影响新计划前10个位置
d=6 ：推理期间已经过去6个动作位置
44  ：本次真正进入执行队列的新动作数量
```

Guided RTC 不要求专门训练。它在推理内部把新计划的开头拉向旧计划，并让这种影响逐渐减弱，所以切换不会突然跳变；代价是推理计算量比普通生成更大。第一个 chunk 没有旧动作可参考，因此就是普通生成。

## 4. Trained RTC：完整运行过程

Trained RTC 使用同样的异步队列框架，但连续性不是临时施加的软引导，而是模型在训练时已经学会的能力。

先增加一个前提：

```text
policy.type = pi05
checkpoint 的 rtc_training_max_delay = 10

也就是模型训练时见过最长10步的已确定动作前缀。
```

### 4.1 模型在训练时学了什么

训练时，每个样本随机选择 `0～10` 个开头动作作为“已经确定、不能修改的动作”。这些动作保持为真实动作；模型只学习如何根据观测和这段已确定动作，生成后面的动作。

可以把训练任务理解为：

```text
已知：前面若干个必须执行的动作
要求：生成与它们连续、同时符合当前观测的后续动作
```

因此普通 Pi0.5 checkpoint 不能直接使用 Trained RTC；checkpoint 必须在训练或微调时启用 `rtc_training_max_delay > 0`。

### 4.2 真机推理时发生什么

仍假设预测延迟和实际延迟都是 `d=6`：

```text
1. 执行队列还剩 30 个动作
                  |
2. 因为 queue_size <= q=30，后台开始下一次推理
                  |
3. 后台读取旧队列；execution_horizon=10 保证最多保留前10步供 RTC 使用
                  |
4. 根据预计的 d=6，旧队列最前面的6个动作被设为新 chunk 的硬前缀
                  |
5. 推理期间机器人继续执行旧队列中的这6个动作，旧队列还剩24个
                  |
6. 模型生成 H=50 的新 chunk：
   位置 0～5 始终被锁定为旧动作，模型主要生成位置 6～49
                  |
7. 推理完成后确认实际也经过了6步，删除新 chunk 的位置 0～5
                  |
8. 丢弃旧队列剩余的24个动作，用新 chunk 的位置6～49替换整个队列
                  |
9. 新执行队列同样共有 H-d = 50-6 = 44 个动作
```

这里“硬前缀”的意义是：新 chunk 的前6个位置在整个生成过程中都不允许偏离旧动作。虽然这6个位置最后会因时间已经过去而被删除，但模型在生成位置6之后的动作时看到了它们，因此能从旧计划自然地继续下去。

Trained RTC 中 `s=10` 的作用与 Guided RTC 不同：

```text
s 只保证系统保存了足够长的旧动作前缀
真正被硬锁定的数量是预计推理延迟 d，而不是全部 s 个动作
它不存在 Guided RTC 那种从 d 到 s 逐渐减弱的软过渡区
```

所以 `max_guidance_weight` 和 `prefix_attention_schedule` 对 Trained RTC 不起作用。

### 4.3 实际延迟比预计延迟更长时

上面的例子假设预计6步、实际也是6步。如果预计按6步生成，但实际用了7步，那么新 chunk 的第6步不是训练时锁定的旧动作，却已经在现实中由旧队列执行过。为避免错误对齐，LeRobot 会丢弃这整个新 chunk，并根据最新延迟重新推理。

如果实际延迟超过 checkpoint 的 `rtc_training_max_delay=10`，也会丢弃。连续5次都无法得到可用 chunk，RTC 线程会停止并报告延迟超出支持范围，而不会悄悄切换成 Guided RTC。

## 5. 两种模式放在一起看

```text
共同点：
  q 决定何时启动后台推理
  推理期间机器人继续执行旧队列
  推理完成后删除新 chunk 中已经过期的前 d_real 步
  最后用 new_chunk[d_real:H] 替换旧队列

Guided RTC：
  普通 flow-matching checkpoint 可以使用
  新 chunk 在前 s 个位置逐渐摆脱旧计划
  RTC 推理额外开销较大

Trained RTC：
  需要专门训练的 Pi0.5 checkpoint
  将前 d 个位置硬锁为旧动作，让模型生成连续后缀
  没有 Guided RTC 的软过渡权重和反向引导开销
```

RTC 本身不会提高底层控制频率。只要执行队列不空，机器人仍然可以每个控制周期获得一个动作；RTC 的作用是让后台推理和动作执行重叠，并处理新旧 chunk 的连续衔接。

关键源码：

- [后台推理触发、延迟测量与队列合并](./lerobot/src/lerobot/rollout/inference/rtc.py)
- [新 chunk 删除过期动作并替换队列](./lerobot/src/lerobot/policies/rtc/action_queue.py)
- [Guided RTC 的旧动作约束](./lerobot/src/lerobot/policies/rtc/modeling_rtc.py)
- [Trained RTC 的训练前缀与硬前缀推理](./lerobot/src/lerobot/policies/pi05/modeling_pi05.py)
- [生成过程中反复锁定硬前缀](./lerobot/src/lerobot/policies/common/flow_matching.py)
- [Trained RTC 配置检查](./lerobot/src/lerobot/rollout/context.py)

## 6. 参数约束与 30 Hz 示例

RTC 论文的可行条件：

```text
d <= s <= H - d
```

因此：

```text
H >= 2*d
```

trained 模式在 LeRobot 中按最大训练延迟进行启动前检查：

```text
D_train <= s <= H - D_train
q >= D_train
```

guided 模式没有同样严格的启动检查，但真机设计仍建议：

```text
d_design = ceil(L_p99 * f)
q >= d_design
d_design <= s <= H - d_design
```

30 Hz 示例：

| 端到端延迟 | `d = ceil(L*30)` | 理论最小 `H = 2*d` |
|---:|---:|---:|
| 100 ms | 3 | 6 |
| 200 ms | 6 | 12 |
| 300 ms | 9 | 18 |

例如 `L_p99=300 ms`：

```text
d_design = 9
H >= 18

若 H=20：
s 可取 9、10 或 11
q 至少取 9
```

数学下限没有抖动余量。真机应使用包含相机、预处理、网络、推理和后处理的 `L_p99`，并额外预留若干动作。

## 7. 选择建议

```text
已有普通 Pi0/Pi0.5/SmolVLA 等 flow checkpoint
    -> 先用 guided RTC

准备训练或微调 Pi0.5，且延迟范围可以预先测量
    -> 使用 trained RTC

延迟经常超过 D_train
    -> 扩大训练延迟分布、降低控制频率，或改回 guided

更看重低推理延迟/低显存
    -> trained 更合适

更看重无需重训和软约束灵活性
    -> guided 更合适
```

## 8. 最小命令

Guided RTC：

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=/path/to/policy \
  --inference.type=rtc \
  --inference.rtc.mode=guided \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --inference.rtc.prefix_attention_schedule=LINEAR \
  --inference.queue_threshold=30 \
  --fps=30 \
  ...
```

训练 Pi0.5 trained RTC checkpoint：

```bash
lerobot-train \
  --dataset.repo_id=user/dataset \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.rtc_training_max_delay=10 \
  ...
```

部署 trained RTC：

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=/path/to/pi05_rtc_checkpoint \
  --inference.type=rtc \
  --inference.rtc.mode=trained \
  --inference.rtc.execution_horizon=10 \
  --inference.queue_threshold=30 \
  --fps=30 \
  ...
```

## 9. 参考

- [LeRobot RTC 文档](https://huggingface.co/docs/lerobot/rtc)
- [原始 Guided RTC 论文](https://arxiv.org/abs/2506.07339)
- [Training-Time RTC 论文](https://arxiv.org/abs/2512.05964)
- [LeRobot RTC 后台执行引擎](./lerobot/src/lerobot/rollout/inference/rtc.py)
- [LeRobot ActionQueue](./lerobot/src/lerobot/policies/rtc/action_queue.py)
- [LeRobot RTC 配置](./lerobot/src/lerobot/policies/rtc/configuration_rtc.py)

## 10. Guided RTC 公式与当前源码梯度流

### 10.1 `A_prev` 的实际长度

以 `q=30、s=10、H=50` 为例：

```text
队列中的旧动作：       30 个
传给 policy 的 A_prev：前 10 个（按 s 截断）
RTC 内部的 A_prev：    补零到 50 个以匹配新 chunk
```

因为位置 `i>=s` 的权重为零，补出的零不参与约束。旧队列第 10 个以后的动作不参与 guidance，但仍用于推理期间持续控制机器人。

### 10.2 从基础 Flow Matching 到 Guided RTC

Flow Matching 约定：

```text
x_t = (1-t)*a + t*epsilon
v_target = epsilon - a
```

因此模型输出 `v_base` 后，可直接估计干净动作：

```text
x_clean = x_t - t*v_base
```

基础推理每个 Euler 步只有：

```text
v_base = v_theta(x_t, observation, t)
x_next = x_t + dt*v_base
dt = -1/N
```

Guided RTC 在两行之间加入：

```text
error      = W * (A_prev - x_clean)
J          = d(x_clean) / d(x_t)
correction = transpose(J) * error

g_raw = (t^2 + (1-t)^2) / (t*(1-t))
g     = min(max_guidance_weight, g_raw)

v_rtc  = v_base - g*correction
x_next = x_t + dt*v_rtc
```

`correction` 是衔接误差对当前带噪 chunk 的反向映射：

```text
correction[j] = sum_k error[k] * d(x_clean[k])/d(x_t[j])
```

源码用 VJP 直接计算，不显式创建 `J`：

```python
correction = torch.autograd.grad(
    outputs=x_clean,
    inputs=x_t,
    grad_outputs=error.detach(),
)[0]
```

它等价于下列衔接损失的负梯度方向：

```text
L = 0.5 * sum_i W_i * ||x_clean[i] - A_prev[i]||^2
correction = -dL/dx_t
```

默认 `LINEAR` 权重为：

```text
i < d       : W_i = 1
d <= i < s  : W_i = (s-i)/(s-d+1)
i >= s      : W_i = 0
```

例如 `d=6、s=10`：

```text
W = [1,1,1,1,1,1, 0.8,0.6,0.4,0.2, 0,...]
```

`W=1` 只是最大权重的软引导，不保证动作严格相等；Guided RTC 没有真正的数值硬锁。

### 10.3 当前源码的梯度流问题

理论上应先为 `x_t` 开启梯度，再计算 `v_base`：

```python
x_t = x_t.clone().detach().requires_grad_(True)
v_base = policy(x_t)
x_clean = x_t - t*v_base
```

此时存在两条梯度路径：

```text
x_t ---------------------> x_clean
x_t -> policy -> v_base -> x_clean

J = I - t*d(v_base)/d(x_t)
correction = (I - t*d(v_base)/d(x_t))^T * error
```

但当前源码顺序是：

```python
x_t = x_t.clone().detach()  # requires_grad=False
v_base = policy(x_t)
x_t.requires_grad_(True)    # 在模型前向之后才开启
x_clean = x_t - t*v_base
```

因此当前计算图缺少 `x_t -> policy -> v_base` 的输入梯度路径：

```text
d(v_base)/d(x_t) = 0       # 仅指当前 autograd 图
J = I
correction = error
```

这意味着当前本地实现没有得到理论上的完整模型 Jacobian 修正；若要实现完整 VJP，应把 `requires_grad_(True)` 移到 `policy(x_t)` 之前。这里只计算相对 `x_t` 的梯度，不会训练或更新模型参数。

对应源码：[`RTCProcessor.denoise_step`](./lerobot/src/lerobot/policies/rtc/modeling_rtc.py) 与 [`euler_integrate`](./lerobot/src/lerobot/policies/common/flow_matching.py)。

## 11. Trained RTC 技术细节与源码实现

固定参数：

```text
H = 50                    # chunk长度
A = 6                     # 每步动作维度
q = 30                    # 队列触发阈值
s = 10                    # execution_horizon
M = 10                    # rtc_training_max_delay
d_cond = 6                # 推理前预计延迟步数
d_real = 6                # 本次实测延迟步数
```

### 11.1 训练：学习从已确定前缀继续生成

每个训练样本独立采样前缀长度：

```text
K ~ UniformInteger(0, M)
```

对 `i=0...H-1`：

```text
i < K  : t_i = 0
i >= K : t_i = t

x_t[i] = t_i*epsilon[i] + (1-t_i)*a[i]
u[i]   = epsilon[i] - a[i]
```

因此前缀是干净动作，后缀是带噪动作：

```text
i < K  : x_t[i] = a[i]
i >= K : x_t[i] = t*epsilon[i] + (1-t)*a[i]
```

只对后缀计算 Flow Matching 损失：

```text
L =
sum_{i=K}^{H-1} sum_{m=0}^{A-1}
    (v_pred[i,m] - u[i,m])^2
/
((H-K)*A)
```

训练目标是：给定当前观测和 `K` 个不可修改的干净动作，生成与它们连续的后续动作。普通 Pi0.5 checkpoint 没有学过这种条件，必须使用 `rtc_training_max_delay > 0` 训练或微调。

### 11.2 推理：每个去噪步都硬锁前缀

队列剩余30个动作时，执行引擎先按 `s=10` 保留：

```text
A_prev = [P0, P1, ..., P9]
```

Pi0.5再按 `d_cond=6` 构造：

```text
hard_prefix[0:6] = [P0, P1, P2, P3, P4, P5]
prefix_mask      = [True]*6 + [False]*44
```

每个 Euler 去噪步执行：

```python
x[0:d_cond] = A_prev[0:d_cond]
time[0:d_cond] = 0

v = policy(x, observation, time)
x = x + dt*v

x[0:d_cond] = A_prev[0:d_cond]
```

所以整个生成过程中始终严格满足：

```text
x[0:d_cond] == A_prev[0:d_cond]
```

这是真正的数值硬约束。Trained RTC 不计算 `error`、Jacobian、`correction` 或 `g(t)`，也不执行推理期反向传播。

### 11.3 推理结束后的队列更新

当 `d_cond=d_real=6`：

```text
推理结果 = [P0,P1,P2,P3,P4,P5, N6,N7,...,N49]
推理期间机器人已执行 = [P0,P1,P2,P3,P4,P5]
new_queue = new_chunk[d_real:H] = [N6,N7,...,N49]
队列长度 = H-d_real = 44
```

旧队列其余动作被新队列替换。

### 11.4 `d_cond` 与 `d_real` 不一致

若预计6步、实际5步：

```text
d_cond = 6
d_real = 5
new_chunk = [P0,P1,P2,P3,P4,P5,N6,...]
new_queue = [P5,N6,...]
```

该 chunk 可以使用：机器人再执行一个已经锁定的旧动作，然后进入新动作。

若预计6步、实际7步：

```text
d_cond = 6
d_real = 7
```

位置6没有被硬锁，但现实中旧队列的 `P6` 已经执行，时间线不再可靠，因此整个新 chunk 被丢弃并重新推理。接受条件为：

```text
d_real <= d_cond
d_real <= M
```

连续5次无法满足就停止 RTC 线程并报错。第一次存在旧 chunk、但没有历史延迟时，源码以 `d_cond=M` 启动。

### 11.5 与基础 Flow Matching 的差异

```text
基础 Flow Matching：
    v = policy(x, observation, t)
    x = x + dt*v

Trained RTC：
    去噪前锁定前缀并将其时间设为0
    执行普通模型前向和Euler更新
    去噪后再次锁定前缀
```

`s` 在 Trained RTC 中只保证缓存足够长的旧动作；真正硬锁的数量是 `d_cond`，不存在 Guided RTC 的软衰减区。当前启动检查为：

```text
policy.type = pi05
M > 0
s >= M
q >= M
s <= H-M
```

对应源码：

- [训练前缀、Flow输入和后缀损失](./lerobot/src/lerobot/policies/pi05/modeling_pi05.py)
- [每个去噪步前后硬锁前缀](./lerobot/src/lerobot/policies/common/flow_matching.py)
- [预计延迟、实测延迟和丢弃判断](./lerobot/src/lerobot/rollout/inference/rtc.py)
- [Trained RTC 启动检查](./lerobot/src/lerobot/rollout/context.py)
