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
```

默认写入：
`/home/py/main/easyEmbodiedDeploy/datasets/pyagxarm_recordings`。可通过
`--output DIR` 修改。

启动后会先检查四路 CAN，连接三台相机和四台机械臂并回零。输入任务指令后，
相机完整画面会等比例缩放并用黑边补到 `224x224`，不会裁掉左右视野。程序把
左右 J6 预定位到 -45/+45 度，启动 200 Hz 跟随并以 10 Hz 采样。录制时
按回车结束，机械臂回零期间仍继续采样，随后选择保存或丢弃。

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
    ├── timestamps, camera_timestamps, action_source_timestamps
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
| `camera_timestamps` | `(T,3)` | s，三路相机相对 episode 第一帧，可有很小负值 |
| `action_source_timestamps` | `(T,2)` | s，左右 leader 最新命令相对 episode 第一帧 |
| `trajectory_phase` | `(T,)` | 枚举：`0=task`，`1=home` |
| `rewards` | `(T,)` | 无量纲；Recorder 当前写 0 |
| `dones` | `(T,)` | 布尔值；仅最后一帧为 true |
