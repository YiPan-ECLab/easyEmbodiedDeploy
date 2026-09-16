# 目录管理、相机标定与 OpenPI 指标验收

日期：2026-09-16。基线：`8916cd6`。本轮改动明细见 [模块改动清单](MODULE_CHANGES.md#2026-09-16目录管理相机标定和-openpi-指标)。

## 已通过的验证

| 验证 | 结果 |
| --- | --- |
| CPU 回归矩阵 | **117 项通过**：Recorder 9、LeRobot Trainer 13、LeRobot Deployer 9、OpenPI Trainer 21、OpenPI Deployer 23、运行边界 18、新资源与标定 14、数据链路 10 |
| 原统一 Web 浏览器回归 | 十个模块正常加载，无浏览器 JS 异常；踏板重复按键、保存与丢弃协议通过 |
| 新目录管理 Chromium 流程 | 新建 → 重命名 → 复制路径 → 回收站删除 → 恢复 → 再删除 → 永久删除通过；测试只操作 UUID 命名的临时目录 |
| 新标定 Chromium 流程 | 默认参数正确、自定义棋盘尺寸和 ChArUco 切换、真实 HTTP 求解合成真值、导出部分标定包、手机 390 px 无横向溢出；测试记录自动清理 |
| 标定 HTTP 流程 | 自定义记录、角点检测、静止确认、SDK 来源 / 手动位姿样本保存、图像回读、相机身份不匹配拒绝、移除样本通过 |
| 标定数学 | 两个腕部 / 两个全局方向均与已知变换一致，矩阵误差小于 2e-6；内参恢复、鱼眼 PnP、固定轴 XYZ 欧拉角、逆变换、方形棋盘四向重排、多轴退化拒绝、样本不足拒绝通过 |
| 标定质量与文件契约 | 板参数快照、增删样本后求解失效、未求解不能导出、部分标定列出缺项、保存原图和角点通过 |
| 文件系统边界 | 根目录不可删除 / 重命名，路径越界拒绝，符号链接不遍历，链接删除不影响目标，恢复不覆盖文件，识别中的任务引用阻止修改 |
| 原 HTTP 数据链路 | Viewer、HDF5 质量分析、LeRobot 转换、LeRobot 质量分析、两个训练校验接口和跨源控制请求拒绝通过 |
| OpenPI 指标回归 | 连续 303 tick 的累计计数在仅展示最近 240 行时仍正确；最终 summary 尚未出现时已有推理中位数、最近耗时与频率；独立日志命名、持久汇总、观测响应带出指标且客户端不增加请求通过 |
| OpenPI 真正 GPU 推理 | 本机真实 `openpi_pi05_10hz_full_trainedrtc_absolute/20000`，100 步离线 trained RTC，退出码 0；每个 tick 均写入推理指标，累计计数与落盘 summary 一致 |
| 实际部署记录展示 | 只读验证使用者实际部署时生成的 JSONL；新页面显示推理约 247 ms 中位数、累计饿死 0、实测发送约 150 Hz；修复后有数据的曲线不再覆盖“等待数据”字样 |
| 最终桥接指标链路真机验证 | 在独立 6107 端口启动原桥接流程，只发送一个与当前观测完全相同的保持目标（目标改变量 0）；5 次原 observe 响应均携带统计，最终 31 个发送周期、实测 29.998 Hz；停止后双臂和夹爪按原流程归零完成 |
| 代码检查 | 新模块 Ruff 未使用符号检查、`git diff --check` 通过；原采集器和 SDK 文件与基线一致；共享桥接仅增加响应字段及客户端缓存 |

本机 CPU 测试日志：`/tmp/easy-embodied-checks-a6vgk6vj/`。浏览器截图：`/tmp/easy-resources-browser/`。GPU 离线推理日志：`outputs/openpi_deployer/20260916-154301-45446df2.jsonl` 及同名 `.summary.json`；本次中位数约 242 ms、100 个有效动作 tick、运行饿死 0。该离线验证没有打开硬件桥，没有发送机械臂动作。

离线模型动作触发了既有关节范围裁剪，该次数已正确记录；此次验证只确认推理和诊断链路，不据此判断模型能完成真实任务，也不作为位姿模型测试。

## 真实相机与只读位姿

已逐台打开和关闭实际相机，读取 640×480 RGB 帧和选中视频 profile 的内参：

| 角色 | 序列号 | 实际传感器 | 畸变模型 |
| --- | --- | --- | --- |
| 全局 | CP2AB530005P | COLOR_SENSOR | BROWN_CONRADY_K6 |
| 左腕 | CV2L360000KH | COLOR_SENSOR | BROWN_CONRADY_K6 |
| 右腕 | CV2L3600003K | COLOR_SENSOR | BROWN_CONRADY_K6 |

两腕设备在当前配置中都走采集器已有的普通 RGB 回退；没有强行使用不存在的左右独立彩色传感器。畸变顺序按 OpenCV 有理模型转换，内参来源随记录保存。

左右从臂均通过新模块成功执行 `connect → get_flange_pose → disconnect`。测试只读法兰位姿，未发送使能、模式切换、预定位、运动、回零指令。原 `PiperRecorder` 和 SDK 文件没有改动。共享桥接仅在 observe 响应中附带已有统计；其硬件控制逻辑没有修改。

上述只读测试指标定模块。另对最终共享桥接的响应字段执行了表中所列的短程保持目标测试，未改变目标位置；原桥接启动和退出中的回零流程均完成。该测试不使用额外后台统计请求，不改变原看门狗心跳行为。原始统计摘录保存在本机 `/tmp/easy-bridge-metrics-check.json`。

## 仍需实物操作的三项

1. 摆放真实板面，核对默认 4×6 内角点与 53.5 mm 单格，并确认角点 0 在各帧对应同一物理内角点。也可换板后输入新参数。
2. 按 [操作说明](CAMERA_CALIBRATION.md) 的固定方式，分别完成左腕、右腕、全局对左 base、全局对右 base 的静止样本采集。每项建议 15–25 个姿态。
3. 用未参加拟合的新姿态或独立测量点检验导出外参精度，确认可用于你的实际工作范围。

当前相机视野中没有标定板，无法自动替操作者固定板面和改变板 / 法兰姿态。因此真实外参精度尚未验收；合成真值测试不能替代这一部分。

## 位姿链路交付边界

[位姿动作数据流方案](POSE_ACTION_DATAFLOW.md) 已列出绝对 base 位姿、Qwen 公式 (5) 的 camera delta、当前腕部参考系冻结、rotation6D → rotvec、重采样、action label 来源、夹爪、checkpoint 契约和未来执行端解码。

绝对模式无需相机标定。需要审阅的选择是：旧数据使用从臂未来实测轨迹还是原动作 FK；absolute 用 rotvec 还是 Euler；位姿状态外是否保留关节信息。**没有提前修改数据转换、训练或机械臂执行模式。**

## 复现命令

在另一个终端启动测试工作台 `python3 workbench.py --port 8098`，然后执行：

```bash
python3 tests/run_checks.py
thirdParty/.venvs/lerobot-pi05/bin/python tests/http_smoke.py --url http://127.0.0.1:8098
uv run --no-project --with playwright python tests/browser_smoke.py --url http://127.0.0.1:8098
uv run --no-project --with playwright python tests/browser_resources.py --url http://127.0.0.1:8098
```

浏览器可用 `--browser /path/to/chromium` 指定已有安装。自动浏览器和 HTTP 测试不会打开真实相机或控制机械臂；新资源测试只创建、清理自己的临时目录和合成标定记录。
