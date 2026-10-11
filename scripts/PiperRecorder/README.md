# pyAgxArm Piper Recorder

双侧 Piper leader/follower 数据采集，操作流程与旧版 `record.py` 一致。代码使用
`pyAgxArm 1.0.0` 的 `PiperFW.V189` 驱动。现场四台臂 `piper-lt`、
`piper-lo`、`piper-rt`、`piper-ro` 的软件版本均为 `S-V1.9-1`，硬件版本
均为 `H-V1.2-1`。

## 运行

```bash
VENV=/home/py/main/easyEmbodiedDeploy/thirdParty/.venvs/piper-home
$VENV/bin/python record.py --check
$VENV/bin/python record.py
# 例如：20 Hz 采样，并把开始动作阈值调到 0.02 rad
$VENV/bin/python record.py --hz 20 --start-joint-delta-rad 0.02
```

默认写入：
`/home/py/main/easyEmbodiedDeploy/datasets/hdf5/pyagxarm_recordings`。可通过
`--output DIR` 修改。

启动后会先检查四路 CAN，连接三台相机和四台机械臂并回零。四台臂均使用关节反馈
闭环运动到六关节零位，leader 到位后恢复零力拖动。启动检查会确认两个
follower 的关节与夹爪反馈，并通过示教器参数查询确认两个 leader 夹爪示教器在线。
回零时，两个 leader 会临时切换到 follower 受控模式；示教臂关节进入位置控制并到达
ZERO 后，再将 leader 和 follower 的四个夹爪全部闭合到 `0 m`，最后恢复 leader
零力拖动。输入任务指令后，
相机完整画面会等比例缩放并用黑边补到 `224x224`，不会裁掉左右视野。程序把
左右 J6 预定位到 -45/+45 度，启动 200 Hz 跟随并默认以 30 Hz 采样。录制时
按回车结束。预定位完成后先建立静止动作基线；在真人动作超过
`--start-joint-delta-rad`（默认 `0.01 rad`）或
`--start-gripper-delta-m`（默认 `0.002 m`）前，只读取诊断数据而不写入 episode，
因此 J6 内旋后的等待静止帧不会进入最终文件。检测到动作后才开始记录 task 帧，
机械臂回零期间仍继续采样，随后选择保存或丢弃。任务阶段如果相邻
采样间隔超过 `--max-task-gap-s`（默认 `0.35 s`），程序会停止本次录制并拒绝保存，
避免把设备阻塞或系统调度造成的长空档混入训练数据；错误信息会同时给出相机读取和
机器人反馈耗时。

任一侧的任一关节或夹爪达到阈值即可开始，允许另一侧全程静止；四台机械臂及三路相机
仍需连接并提供有效数据。task 至少需要两帧，不能用回零帧补足；过短录制会在回零后
明确拒绝保存。关节和夹爪的 NaN/Inf 在限幅前报错，避免被转换成正常边界值。
落盘检查允许相机帧重复使用，但拒绝相机时间戳倒退。

可用采集参数：`--hz` 控制写入采样频率；`--max-task-gap-s` 控制任务段允许的最大
实际间隔；`--start-joint-delta-rad` 和 `--start-gripper-delta-m` 控制从静止等待切换
到正式记录的动作阈值。它们不会改变 200 Hz 的 leader/follower 跟随频率。

跟随线程以 200 Hz 轮询 leader 反馈，只在关节或夹爪实际数值变化时发送
`move_js`/`move_gripper_m`。最新 Piper 固件会保持最新 JS 目标，静止时不需要重复
发送相同控制帧，避免无效 CAN 流量影响反馈采集。

leader 的六关节控制量由 `0x155`、`0x156`、`0x157` 三组 CAN 帧组成。程序只有在
三组时间戳都完成一次刷新后才组装并下发动作，接管瞬间不会把尚未填满的零值半帧
发送给 follower；上一条轨迹遗留的夹爪 `0x159` 帧也会按时间戳丢弃。

夹爪两类 CAN 数据不能混用：受控模式下四个夹爪均通过 `get_gripper_status()` 读取
`0x2A8` 状态并确认回零；恢复 leader 模式后，leader 夹爪示教器在真人操作时发送
`0x159` 控制帧，程序通过 `get_gripper_ctrl_states()` 读取并转发给 follower。
leader 静止时没有周期性的 `0x159`，此时返回 `None` 是正常行为，不表示连接失败。

HDF5 落盘时会处理机械边界附近的编码器误差：关节超出官方边界不超过 `1°`、夹爪
超出 `[0,0.1] m` 不超过 `0.001 m` 时，`state` 和 `action` 会截断为边界值；更大的
越界仍拒绝保存。根节点属性 `boundary_clipped_values` 记录本次截断的数值个数。

## HDF5 控制量

pyAgxArm 是控制 SDK，没有规定官方 HDF5 文件结构。本采集器保留旧采集器的
`data/demo_0` 数据树，数值格式改为 pyAgxArm API 直接使用的物理量。

`data/demo_0/actions` 为 14 维 `float64`：

```text
[left_j1..j6(rad), left_gripper(m), right_j1..j6(rad), right_gripper(m)]
```

这些值是传给 `move_j()` 和 `move_gripper_m()` 的实际参数，没有归一化，也没有
度、毫度或微米单位。夹爪控制力固定为 `1.0 N`，记录在
`data/demo_0.attrs["gripper_force_n"]`。回放单侧 7 维动作时：

```python
arm.move_j(action[:6].tolist())
gripper.move_gripper_m(float(action[6]), force=1.0)
```

观测同样保留 pyAgxArm API 的物理量：关节为 rad，夹爪 width 模式为 m、力为 N；
法兰位姿为 `[x, y, z, roll, pitch, yaw]`，位置为 m，XYZ 外旋欧拉角为 rad。
RGB 图像为 `uint8`。

Piper 关节输入采用 pyAgxArm 的模型限位：J1 `[-2.617994, 2.617994]`、J2
`[0, 3.141593]`、J3 `[-2.967060, 0]`、J4 `[-1.745330, 1.745330]`、
J5 `[-1.221730, 1.221730]`、J6 `[-2.094396, 2.094396]` rad；两台 follower
实测 `max_range_config=0.1 m`，因此夹爪 width 控制范围设为 `[0, 0.1]` m。

## HDF5 结构

```text
/
└── data/demo_0
    ├── actions, left_actions, right_actions
    ├── states, robot_states -> states
    ├── timestamps, scheduled_timestamps, camera_timestamps, action_source_timestamps
    ├── capture_durations, camera_durations, robot_durations
    ├── trajectory_phase, rewards, dones
    └── obs
        ├── agentview, left_eye_in_hand, right_eye_in_hand
        ├── left/right_joint_states
        ├── left/right_ee_states
        ├── left/right_gripper_states
        ├── left/right_gripper_force
        └── joint_states, ee_states, gripper_states
```

所有连续物理量使用 `float64`，图像使用 `uint8`，阶段使用 `int8`，reward
使用 `float32`，done 使用 `bool`。`robot_states` 是 `states` 的 HDF5 硬链接，
不会重复占用空间。

| 数据集 | Shape | 单位与范围 |
| --- | --- | --- |
| `actions` | `(T,14)` | 每臂 J1..J6 为 rad，范围见上方关节限位；夹爪为 `[0,0.1]` m |
| `left_actions`, `right_actions` | `(T,7)` | 对应单臂的 6 个 rad 关节与 1 个 m 夹爪宽度 |
| `states`, `robot_states` | `(T,14)` | 与 `actions` 同排列和单位，值为 follower 反馈 |
| `obs/*_joint_states` | 单臂 `(T,6)`，合并 `(T,12)` | rad，范围受相同关节机械限位约束 |
| `obs/*_ee_states` | 单臂 `(T,6)`，合并 `(T,12)` | `[x,y,z]` 为 m，`[roll,pitch,yaw]` 为 rad |
| `obs/*_gripper_states` | 单臂 `(T,1)`，合并 `(T,2)` | m，配置范围 `[0,0.1]` |
| `obs/left/right_gripper_force` | `(T,1)` | N；新版为 SDK 反馈，迁移的历史文件为 `NaN`（旧采集未保存） |
| `obs/agentview` 等三路图像 | `(T,224,224,3)` | RGB `uint8`，每通道 `[0,255]` |
| `timestamps` | `(T,)` | s，相对 episode 第一帧 |
| `scheduled_timestamps` | `(T,)` | s，采样循环计划时刻，用于诊断调度延迟 |
| `camera_timestamps` | `(T,3)` | s，三路相机相对 episode 第一帧，可有很小负值 |
| `action_source_timestamps` | `(T,2)` | s，左右 leader 最新命令相对 episode 第一帧 |
| `capture_durations` | `(T,)` | s，每帧相机读取加机器人反馈总耗时 |
| `camera_durations`, `robot_durations` | `(T,)` | s，分别记录相机读取和机器人反馈耗时 |
| `trajectory_phase` | `(T,)` | 枚举：`0=task`，`1=home` |
| `rewards` | `(T,)` | 无量纲；Recorder 当前写 0 |
| `dones` | `(T,)` | 布尔值；仅最后一帧为 true |

## Web 与脚踏板

统一入口：在仓库根目录运行 `python3 workbench.py`，选择“Piper 采集”。
独立入口：`thirdParty/.venvs/piper-home/bin/python scripts/PiperRecorder/web_server.py`。

页面直接驱动原 `record.py` 的指令、预定位、等待动作、采集、回零和保存阶段。
回车在指令阶段开始、采集阶段结束、确认阶段保存；丢弃使用页面按钮。
留空沿用指令，`:q` 仍可在指令阶段退出。采样率、启动阈值、gap 判定、动作单位、
task/home 分段和归零轨迹保持原流程。新增 `--web-control` 是内部结构化输入通道，
日常 CLI 仍使用原终端输入。

点击页面使其获得焦点后再踩踏板。阶段切换时丢弃过期输入，长按只接受一次。

页面上方的“三路相机 · 实时画面”显示场景、左腕和右腕相机的当前画面。
连接设备后即可预览，包括等待指令、示教、回零和待保存阶段；断流时隐藏旧画面并自动重试。
预览复用相机已有的 224 × 224 RGB 帧，通过独立 HTTP 线程按需编码 JPEG，
每路最多约 10 次/秒，页面不可见时暂停请求，不改变采样频率、保存图像或控制指令。
升级已有运行实例时，先保存当前轨迹、正常结束会话，再停止并重新打开工作台中的
“Piper 采集”模块，随后重新连接设备。只刷新页面不会升级正在运行的服务进程。

保存失败后可以重试或丢弃，设备异常显示在阶段提示和日志中。
“结束会话并归位”会等待正在进行的归零，丢弃当前未保存 episode 后退出。
Web 服务退出也会请求该清理流程。

## 主臂 J2 零点异常

现场已在左、右 leader 分别观察到同一种故障：J2 到达真实机械下限时，控制器仍报告
约 `+4–5°`；继续命令 `0°` 后驱动进入 `0xA4`（过流、驱动错误、堵转），电流约
`-11.7 A`、估算力矩约 `-55 N·m`。向正方向低速移动时电机、驱动器和减速器均能正常
工作，重新设置 J2 零点后完整采集和回零流程恢复，因此 `piper-lt/piper-rt 使能失败`
是零位参考异常的结果，不是 HDF5 或动作换算导致。正常采集路径不会调用
`calibrate_joint()`。

该问题目前只确认到“leader J2 的控制器参考与真实机械位置分离”。旧版 Piper 固件曾有
[主从模式切换导致零位/参考重建和 Flash 参数损坏](https://github.com/agilexrobotics/pyAgxArm/discussions/58)
的问题；另有上游报告称外力过载后可能出现
[反馈位置与物理位置静默分离](https://github.com/agilexrobotics/piper_sdk/issues/120)。
本机为 `S-V1.9-1`，尚无厂商资料证明其中任一报告就是本机故障的唯一原因；若再次发生，
应保留当次日志并检查 J2 编码器/耦合件、线束、控制器参数存储和供电瞬态。

网页“主臂 J2 零点诊断与恢复”提供现场兜底。必须先结束采集会话，再选择左右主臂进行
诊断。只有同时满足以下条件才开放恢复：J2 报告 `+2–10°`、其余五关节在零位
`±2°` 内、J2 静止且失能、状态码精确为 `0xA4`。操作员还必须现场确认 J2 已停在
真实机械下限零位并再次确认。后端随后执行 J2 失能、清错、当前位置标零、重新使能，
并要求读回位置在 `±0.5°` 内且状态码为 `0x40`。流程不发送任何位置运动命令；不符合
已验证故障签名时会拒绝自动标零，避免把任意姿态误写成永久零点。
