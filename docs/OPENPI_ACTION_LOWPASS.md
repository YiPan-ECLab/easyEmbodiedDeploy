# 机械臂动作的一阶低通平滑

我们不是把模型输出的动作直接发送给机械臂，而是先把它变成“绝对目标”，再对这个目标做一阶低通，最后按固定频率发送。

## 处理流程

```text
模型动作
  → 转成绝对目标
  → 一阶低通
  → 发送 move_js / move_gripper_m
```

### 1. 先统一成绝对目标

机械臂每次需要的是目标位置，而模型可能输出两种动作：

- **绝对动作**：直接把模型输出当作目标位置。
- **增量动作**：把本次增量加到“上一次确定的绝对目标”上，得到新的绝对目标。

两种动作最后都得到同样的 14 维绝对目标：左右两臂各 6 个关节角和 1 个夹爪位置。之后先检查关节和夹爪限位。

低通滤波放在这里之后，滤波的是绝对目标，不是原始增量。这样不会改变增量动作的累计关系。

### 2. 用一阶低通逐步逼近目标

控制线程以固定频率运行。每次取：

- `u`：最新的绝对目标；
- `y_prev`：上一个控制周期已经发送的目标。

计算本次要发送的目标：

```text
alpha = 1 - exp(-1 / (control_hz × tau))
y     = y_prev + alpha × (u - y_prev)
```

其中：

- `control_hz` 是发送频率；默认 `30 Hz`，即每 `33.3 ms` 更新一次；
- `tau` 是平滑时间常数；默认 `0.1 s`；
- `alpha` 决定每次向新目标靠近多少。

默认参数下：

```text
alpha = 1 - exp(-1 / (30 × 0.1)) ≈ 0.2835
```

也就是说，每个控制周期只走完当前误差的约 `28.35%`，不会突然跳到新目标；连续几个周期后会平滑地接近目标。

`tau = 0` 时直接发送最新目标，相当于关闭滤波。`tau` 越大越平滑，但响应越慢；`tau` 越小响应越快，但对目标跳变的抑制越弱。

### 3. 把滤波结果发送给机械臂

关节模式下，每个控制周期发送：

```python
left_arm.move_js(y[0:6])
right_arm.move_js(y[7:13])
left_gripper.move_gripper_m(y[6])
right_gripper.move_gripper_m(y[13])
```

因此，`move_js` 接收到的已经是平滑后的目标；`move_js` 本身不负责这次一阶低通。

## 最简伪代码

```python
filtered = initial_absolute_target.copy()

while running:
    desired = latest_absolute_target

    if tau == 0:
        alpha = 1.0
    else:
        alpha = 1.0 - exp(-1.0 / (control_hz * tau))

    filtered = filtered + alpha * (desired - filtered)
    send_to_piper(filtered)
    sleep(1.0 / control_hz)
```

当前实现对应代码：`scripts/LerobotDeployer/hardware_bridge.py` 中的 `low_pass_absolute_target()` 和 `FollowerCommandLoop`。
