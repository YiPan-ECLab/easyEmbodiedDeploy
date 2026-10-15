# 多硬件、多 Policy 的最小扩展设计

记录日期：2026-10-09。

状态：未来扩展方案，尚未实施。本文整理当前源码分析与建议边界，不要求现在重构系统。后续实施前应重新核对对应模块，避免直接套用已经过时的实现细节。

## 1. 目标与总体结论

当前系统面向双臂 Piper 与 PI05。LeRobot/Torch 和 OpenPI/JAX 是两套模型运行后端，当前两者均围绕 PI05，尚不等于支持多 policy。

建议沿用现有采集、转换、训练、部署模块，增加轻量描述对象与适配器，把硬件和模型假设从公共流程中移回各自实现。无需先重写工作台、合并虚拟环境或建设复杂插件框架。

希望最终达到：

- 新增硬件主要增加设备适配器、硬件配置和对应测试。
- 新增 policy 主要增加训练/推理适配器、配置映射和对应测试。
- 数据处理、动作语义和部署队列可复用，避免每增加一个硬件与 policy 组合就复制整套流程。
- 保持现有 Piper + PI05 默认行为与旧数据兼容。

只增加部署端的硬件或模型加载入口，改动可以较小；完整支持新硬件的采集、转换、训练和部署，还必须参数化数据维度、相机配置及动作语义。统一接口也不意味着任意 checkpoint 能直接控制任意硬件，具体组合仍需匹配输入和输出语义。

## 2. 当前源码中的可复用基础与耦合点

已经存在的可复用基础：硬件独立进程与 ZeroMQ RPC、Torch/JAX 环境隔离、训练任务进程管理、动作解码、同步/异步动作队列、HDF5 与 LeRobot 数据链路，以及独立的几何计算代码。

| 模块与源码入口 | 当前绑定 | 建议的最小调整 |
|---|---|---|
| [机器人控制](../scripts/PiperRecorder/piper_recorder/robot.py)、[采集流程](../scripts/PiperRecorder/record.py) | 双臂、每臂六关节、主从示教、Piper SDK | 包装现有实现，流程通过适配器调用 |
| [相机采集](../scripts/PiperRecorder/piper_recorder/camera.py) | Orbbec、固定三路相机角色 | 相机驱动与机械臂配置分开，角色和数量来自配置 |
| [HDF5 写入](../scripts/PiperRecorder/piper_recorder/dataset.py) | 固定左右臂字段、轴序和单位约定 | 保留存储流程，从数据描述生成字段 |
| [硬件桥](../scripts/LerobotDeployer/hardware_bridge.py)、[桥客户端](../scripts/LerobotDeployer/bridge_client.py) | 固定 14 维、Piper 配置；桥服务调用采集控制类的私有方法 | 保留 RPC，设备访问下沉到适配器，并提供设备描述 |
| [数据转换](../scripts/LerobotDataProcessor/processor.py) | 固定轴名、三路相机、Piper 限位和物理量检查 | 根据数据描述与硬件描述构造 features 和检查项 |
| [轨迹浏览](../scripts/PiperViewer/piper_viewer/server.py)、[数据分析](../scripts/DataAnalysis/analyze_lerobot.py) | 固定 14 维、左右臂与相机配置，页面也有固定通道假设 | 由字段描述生成读取、曲线与报告 |
| [部分融合](../src/easy_embodied_deploy/temporal_fusion.py) | 抽帧主体按索引工作，但入口要求现有录制格式 | 沿用同步选帧逻辑，格式识别交给数据读取适配层 |
| [Torch 训练](../scripts/LerobotTrainer/trainer.py)、[模型发现](../scripts/LerobotTrainer/web_server.py) | 固定 `policy.type=pi05`，筛选 PI05，包含专用训练参数 | 按 policy 分发参数解析、模型发现和训练命令生成 |
| [JAX 模型配置](../scripts/OpenpiTrainer/openpi_config.py) | `pi05=True`、50 步、内部 32 维、外部 14 维 | 收进 PI05/OpenPI 适配器 |
| [JAX 数据读取](../scripts/OpenpiTrainer/dataset_v3.py) | 固定 state/action 宽度与相机列表 | 由数据描述和 policy 输入要求共同配置 |
| [Torch 部署](../scripts/LerobotDeployer/LerobotDeployer.py)、[JAX 运行时](../scripts/OpenpiDeployer/runtime.py) | 固定观测与 chunk 形状；Torch 部分模式复用 OpenPI 目录内的队列 | 公共调度与模型采样实现分离 |
| [动作语义](../src/easy_embodied_deploy/target_runtime.py)、[位姿编解码](../src/easy_embodied_deploy/pose_actions.py) | 固定左右臂、夹爪索引与尺寸 | 保留算法，按运动链和动作描述取字段 |
| [位姿数据与统计](../src/easy_embodied_deploy/pose_dataset.py) | chunk 读取、统计及契约绑定 50 步 | horizon 由 policy 决定，按对应 horizon 生成 chunk 与统计 |
| [相机标定](../src/easy_embodied_deploy/calibration.py) | 直接创建 Orbbec/Piper，固定连接和相机角色 | 替换设备接入部分，沿用标定数学与操作流程 |
| [工作台](../src/easy_embodied_deploy/workbench.py)、[设备占用](../src/easy_embodied_deploy/resources.py) | 固定启动入口和环境，设备锁默认以 Piper 命名 | 保留进程管理；启动配置与设备资源标识逐步参数化 |

目录管理、通用 HTTP、GPU 查询和进程日志等能力可继续使用。以后更换入口脚本名称时，再同步维护目录管理中的运行任务识别，无需提前重构。

## 3. 三个轻量描述对象

初期使用 dataclass 与可序列化 JSON 即可。描述对象不导入 PyTorch、JAX 或硬件 SDK。

| 描述对象 | 负责的信息 | 不应承担的职责 |
|---|---|---|
| `RobotSpec` | 运动链、关节名称/类型/单位、限位、末端执行器、坐标系、支持的控制空间 | 模型架构、训练参数 |
| `DatasetSpec` | state/action 轴序及单位、动作语义、相机键、位姿字段、目标帧偏移、格式版本 | 当前连接设备、模型内部 padding |
| `PolicySpec` | policy 类型、后端、输入要求、输出维度、horizon、训练及推理能力 | CAN 名称、设备 SDK、真实采集频率约束 |

需要明确的边界：

- state 维度和 action 维度分别描述，不假设两者相等。例如七关节机械臂的关节反馈和末端位姿动作可能具有不同维度。
- 物理动作维度和模型内部维度分开。当前 OpenPI 的 32 维 padding 属于模型适配器。
- 224 图像尺寸属于当前模型输入配置，不是相机驱动的固定要求。调整尺寸的处理应在训练和部署时保持一致。
- 相同维度不能替代轴序、单位、控制空间和坐标系的说明。
- 各通道分别描述绝对值/增量语义，不能默认所有夹爪或所有关节动作都采用同一种表示。

旧格式没有这些描述时，由兼容读取器补出当前 Piper 默认描述。新录制在 HDF5 中保存描述，转换后在 LeRobot 元数据中保留所需信息。checkpoint 保存训练时使用的描述与映射快照，部署不依赖之后可能变化的全局默认配置。

## 4. 硬件接入边界

最小机器人接口示意：

```python
describe()
connect()
observe()
send_target(target)
stop()
close()
```

`observe()` 返回反馈物理量及时间戳；组合设备层可以将其与相机图像组成一次观测。`send_target()` 接收具有明确控制空间和字段含义的目标，例如关节位置或末端位姿。

首先创建 `PiperAdapter` 包装现有 `DualPiper`、控制和回零逻辑。公共采集/部署流程不再直接访问 `.pairs`、`.follower` 或 `_move_to()`。Piper 的回零、预备姿态、控制模式和 SDK 单位转换继续由该适配器负责，不要求其他设备模仿这些操作。

另有两个独立接入点：

- **相机适配器**：提供取帧、时间戳、配置，以及标定需要的内参信息。同一种机械臂可以组合不同相机。
- **示教来源适配器**：当前包装主臂跟随，以后可增加手柄等来源。示教目标与从臂实际反馈仍分别记录，不把 feedback 当作 command 替代。

保留当前硬件进程与 ZeroMQ 传输，增加设备描述查询。客户端可以提供新的通用名称并兼容旧 `PiperBridge` 名称，避免一次性改动所有调用。新增硬件使用自己的环境和配置，不把 SDK 装进训练环境。

设备资源标识应描述实际占用的设备，避免不同硬件全部共用一个名为 Piper 的全局锁；这一调整可以在第二种硬件落地时完成。

## 5. Policy 与后端边界

配置分别表达硬件配置、policy 类型与后端，例如：

```yaml
robot_profile: piper_dual
policy_type: pi05
backend: lerobot
```

PI05 使用另一套实现时改变 `backend`；新增其他 policy 时改变 `policy_type`。初期使用显式注册表与延迟导入即可，无需动态插件安装或自动发现框架。

### 训练侧

统一任务生命周期：解析配置、检查输入、构造启动命令、启动/停止进程、记录日志及训练信息。

模型构造、优化器、LoRA、专用 loss、checkpoint 保存与恢复继续交给对应适配器。保留现有 LeRobot/OpenPI Trainer 和各自虚拟环境，先替换其中固定的配置生成分支。

PI05 的 `expert_only`、LoRA 参数、训练时 RTC 和位姿 loss 不作为所有 policy 的必选参数。公共页面保留任务、数据和运行资源等通用项，模型专用项随适配器能力显示。

### 部署侧

最小调用接口示意：

```python
load(checkpoint)
reset()
predict(observation) -> action_chunk
close()
```

适配器负责相机映射、模型输入构造、归一化/反归一化、内部 padding 和模型状态管理。对外返回真实动作通道对应的物理单位值；单步 policy 可返回长度为 1 的 chunk。

RTC、训练时 RTC、temporal ensemble 等作为独立能力声明，不假设所有 policy 都支持。队列和模型内部状态不能重复管理。

本仓库自带 ACT 的实现已有 `predict_action_chunk()`，可考虑用作第二个 policy 的验证对象。若启用其 temporal ensemble，适配器必须保留对应执行语义，不能直接套用 PI05 的 chunk 消费方式。首阶段可以只接通普通同步训练/部署路径。

## 6. 动作语义与队列边界

建议部署数据流：

```text
硬件观测 → policy 适配器 → 物理单位的预测动作
                                  ↓
                         ActionCodec 解码完整 chunk
                                  ↓
                           可执行目标队列
                                  ↓ policy_hz
                            硬件适配器
                                  ↓ control_hz
                              设备控制
```

`ActionCodec` 负责绝对关节、逐步增量关节、绝对位姿、相机坐标系下的位姿变化，以及相应的锚点与通道语义。沿用现有 `TargetSampler` 和 `PoseCodec` 的算法，逐步改成按描述访问运动链和通道。

必须保留的现有语义：

- 增量动作先完整解码，再由目标队列处理延迟和已消费步骤，避免丢弃增量前缀后损失累计位移。
- 相机位姿 chunk 使用明确的观测锚点，不能把所有位姿动作都当成逐步累加增量。
- RTC 的剩余目标重新编码由语义层处理，不在 SDK 驱动中实现。
- 数据生成与部署解码使用一致的目标帧偏移、旋转表示和坐标系约定；不在迁移时顺便改变现有绝对关节与位姿标签含义。

当前位于 `OpenpiDeployer/runtime.py` 的可复用队列可以逐步移动到公共包，原路径保留导入兼容。模型加载、PI05 特有 RTC 参数和观测转换留在后端适配器里，不必同时重写所有推理引擎。

## 7. 数据与部分融合保持独立

保留当前 HDF5 → 部分融合 → LeRobot 流程，只参数化字段解释：

- state/action 分别按实际维度和轴序读取。
- 相机按列表遍历，不要求固定三路。
- 位姿与末端执行器按运动链描述读取，不固定使用 `[6, 13]` 等索引。
- 硬件限位来自 `RobotSpec`，标签含义来自 `DatasetSpec`。
- 浏览器曲线、分析报告和模型输入映射使用相同描述，避免各模块维护一份独立常量。

位姿数据目前绑定 50 步，是多 policy 扩展时需要优先处理的隐藏耦合。同一份位姿序列应能按不同 horizon 生成 chunk；相应统计需按数据版本、动作语义与 horizon 计算或缓存，不能直接复用固定 50 步的统计。更换 horizon 不应要求重新录制数据或重新编码视频。

时间与序列长度的职责保持分离：

| 信息 | 职责 |
|---|---|
| 采集频率 / `mixed` | 描述数据来源和采样性质 |
| 真实时间戳 | 保留物理时间与来源追溯，允许融合后的非均匀间隔 |
| `video_fps` | 视频存储和帧定位的数值时钟 |
| horizon | policy 使用或预测多少个序列步骤 |
| `policy_hz` | 部署多久从动作队列消费一个目标 |
| `control_hz` | 硬件控制循环发送或处理目标的频率 |

训练按最终序列组织 chunk。不得因为引入统一接口，重新增加均匀采样、采集频率一致、固定双臂/三相机，或数据 FPS 必须等于部署频率等限制。具体 policy 若需要特殊输入，由其适配器明确处理，不能成为所有训练数据的通用限制。

## 8. 最小实施顺序与验收

### 第一步：包装现有实现

增加描述对象、简单注册表、Piper 适配器和 PI05 的两套后端适配器；现有入口与默认配置继续可用。先不改目录组织、合并网页或合并环境。

验收：现有 Piper + PI05 的采集、融合、五种动作模式转换、训练读取及部署解码保持一致，旧数据和 checkpoint 的兼容路径可用。

这一阶段只是建立边界，不能据此宣称已经支持其他真实硬件或 policy。

### 第二步：接通第二个 policy

在现有 Piper 上接入一个不同 policy，先实现训练与同步部署，验证不同 horizon、输入映射、动作输出、归一化和模型状态重置。RTC 等能力按实际支持情况开放。

验收：完成真实的数据读取、训练/保存/恢复、checkpoint 加载和离线部署流程；不通过伪装成 PI05 或强制 50 步来适配。

### 第三步：接通第二种硬件

参数化采集、转换、浏览、分析和标定入口的描述读取，再增加具体硬件适配器与配置。

先用不同维度、不同相机数量的模拟设备验证通用流程，例如单臂或不同关节数的配置；随后进行对应硬件的实际验收。模拟测试通过不代表真实硬件已支持。

验收重点：轴序、单位、state/action 不同维度、相机角色映射、示教目标与反馈区分、控制空间转换，以及既有 `mixed` 数据不被重新采样或拒绝。

## 9. 暂不需要做的工作

当前没有明确扩展对象时，保留此设计备查即可。无需提前创建大量空接口、重写训练循环、统一所有依赖环境，或强迫所有 policy 支持 RTC/LoRA、所有硬件支持相同示教和回零流程。

真正开始扩展时，应先选定第二个 policy 或第二种硬件，用具体差异检验接口边界，再提取已被两种实现共同需要的能力。
