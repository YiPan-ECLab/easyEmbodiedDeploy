# easyEmbodiedDeploy

Piper 双臂采集、数据浏览与转换、质量分析、LeRobot / OpenPI 训练和部署的本地工作台。

![统一工作台](docs/images/workbench.png)

## 启动统一 Web

在仓库根目录执行：

```bash
python3 workbench.py
```

打开 **http://127.0.0.1:8088**。可用 `--port 8090` 更换端口，`--port 0` 自动选择空闲端口。入口仅使用 Python 标准库，各模块在各自的虚拟环境里按需启动。

| 模块 | 功能 | 环境 |
| --- | --- | --- |
| 模型与数据目录 | models / datasets 浏览、新建、重命名、复制路径、回收站与恢复 | piper-home |
| Piper 采集 | 四臂示教、脚踏板 / 回车、保存与丢弃 | piper-home |
| 相机标定 | 自定义棋盘 / ChArUco、内参、左右腕与全局双 base 外参、导出 | piper-home |
| 轨迹浏览 | HDF5 三路图像、时间轴、关节曲线 | piper-home |
| 数据转换 | 多目录、task 筛选、尾部裁剪、降采样、absolute / delta | lerobot-pi05 |
| 质量分析 | HDF5 / LeRobot 全量检查、异常 episode 清理与备份 | lerobot-pi05 |
| LeRobot 训练 | PI05、LoRA / 全参数、RTC、恢复、GPU 和日志 | lerobot-pi05 |
| OpenPI 训练 | JAX、LoRA / 全参数、RTC、FSDP、恢复 | openpi-jax |
| LeRobot 部署 | 同步 / RTC、硬件桥、dry-run、诊断 | lerobot-pi05 |
| OpenPI 部署 | 同步 / RTC、硬件桥、离线数据推理、诊断 | openpi-jax |

打开页面只启动网页服务。连接机械臂、采集、训练和部署均由模块内的按钮启动。切换模块保留页面状态；“服务日志”中可以查看启动故障或停止模块及其任务。终端 Ctrl+C 会等待各模块清理；正在回零时可能需要一段时间。

现有独立脚本与 Web 入口继续可用。首次部署的环境、GPU/CUDA、Orbbec 与 CAN 配置见 [thirdParty/README.md](thirdParty/README.md)。Web 默认仅监听本机；远程使用可转发 SSH 端口。

## 脚踏板采集

1. 打开“Piper 采集”，确认输出目录，点击“连接设备并回零”。
2. 输入任务指令，踩踏板 / 回车开始；留空沿用上一条指令。
3. 等待 J6 预定位，随后照常示教。检测到动作前的静止帧不保存。
4. 踩踏板 / 回车结束 task；自动回零期间仍记录 home 帧。
5. 再踩一次保存，或点击“丢弃本条”；随后进入下一条任务。

网页必须获得键盘焦点，页内会显示踏板就绪提示。阶段令牌与按键去重避免长按或重复请求跨阶段触发。“结束会话并归位”会结束当前记录、等待回零并丢弃尚未保存的数据。关闭浏览器标签不会自动终止正在运行的会话，可重新打开工作台继续操作。

采集与部署桥接共用设备锁，不允许同时占用设备。机械臂连接、反馈、预定位和回零直接沿用已验证的原版硬件实现，详见[硬件对照复测](docs/RECORDER_HARDWARE_RECHECK.md)。

## 验证

新增模块的使用和约定见 [相机标定说明](docs/CAMERA_CALIBRATION.md) 与 [位姿动作数据流](docs/POSE_ACTION_DATAFLOW.md)。标定板默认 **4×6 内角点、53.5 mm**，可在网页修改；实际为 5×7 格。位姿转换、两套训练和部署现支持各自 base 绝对位姿、各自腕部 camera delta 和全局 camera delta；state 固定复用 14 维关节反馈，action 为 14 维 xyz / 三维旋转向量 / 夹爪。绝对位姿使用左右自身 base，不依赖相机标定。

标定页面默认直接使用实际 RGB 视频配置的出厂内参；打开相机即可查看 K、畸变和来源，新建记录时自动保存，无需点击“应用内参”。“相机 → 左法兰”是 `T_left_flange_left_camera`，操作顺序和 base / 法兰区别见上述标定说明。

目录管理只操作仓库 `models/` 和 `datasets/`。删除默认移入对应根目录的回收站；恢复不覆盖同名项目；永久删除需输入完整名称。符号链接可管理链接本身，但不进入链接目标。运行任务中请勿移动其模型和数据；页面会阻止能识别到的进程路径引用，但无法替代所有外部程序的文件占用管理。

OpenPI 的控制循环指标现持续记录每个 tick 的推理延迟、累计饿死次数、限幅次数、队列和桥接频率。网页诊断路径留空时，每次运行独立保存到 `outputs/openpi_deployer/`，结束后保留同名 `.summary.json`。自定义路径仍可使用。推理中位数为最近最多 1000 个 chunk 的统计，排除 warmup；离线运行没有机械臂实测频率。

已安装三套环境后，运行项目的 CPU 回归矩阵：

```bash
python3 tests/run_checks.py
```

工作台启动后，还可执行真实 HTTP 数据链路和浏览器测试（均不启动机械臂）：

```bash
thirdParty/.venvs/lerobot-pi05/bin/python tests/http_smoke.py
uv run --no-project --with playwright python tests/browser_smoke.py
uv run --no-project --with playwright python tests/browser_resources.py --url http://127.0.0.1:8088
uv run --no-project --with playwright python tests/browser_theme.py --url http://127.0.0.1:8088
```

浏览器测试需要安装 Playwright Chromium；也可通过 `--browser /path/to/chromium` 指定已有浏览器。非默认端口用 `--url http://127.0.0.1:8090`。

具体修复、验证结果和剩余现场验收见 [仓库审查记录](docs/REPOSITORY_REVIEW.md)；各模块具体改动见 [模块改动清单](docs/MODULE_CHANGES.md)。

2026-09-16 目录管理、相机标定与 OpenPI 指标修复的 **117 项回归、浏览器和真机只读测试**详见 [当日验收记录](docs/RESOURCES_CALIBRATION_ACCEPTANCE.md)。

此前的统一界面与出厂内参工作见 [界面与内参验收](docs/UI_INTRINSICS_ACCEPTANCE.md)。最新位姿扩展、相对关节修复、GPU / 网页 / 真机测试见 [位姿扩展验收](docs/POSE_ACTION_ACCEPTANCE.md)；使用前按 [标定说明](docs/CAMERA_CALIBRATION.md) 和 [位姿数据链路](docs/POSE_ACTION_DATAFLOW.md) 配置所选动作模式。

Git 仅保存源码与环境锁。`datasets/`、`models/`、虚拟环境、测试产生的数据和权重均不会上传，需要在目标机器上单独复制或重新生成。
