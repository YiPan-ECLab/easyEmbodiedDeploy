# 各模块改动清单

比较基准为改造前的 `e4145bf`。本清单描述最终保留的改动，不把已撤回的硬件改动列为优化。
硬件回归的原因、撤回范围和真实 A/B 结果见 [Recorder 硬件复测](RECORDER_HARDWARE_RECHECK.md)。

## 1. PiperRecorder：采集

- 新增独立采集 Web 和统一工作台入口：配置输出目录、采样参数，连接设备并执行原脚本初始回零，显示阶段、帧数、日志和已保存文件。
- `record.py` 增加可选 `--web-control` 通道：只把原终端输入和阶段输出接到网页。指令 → J6 预定位 → 动作触发 task → Enter 结束 → 回零采样 home → Enter 保存 / n 丢弃的顺序不变。
- 浏览器长按 Enter、重复请求和旧页面请求不能跨阶段触发；留空沿用上一条指令，保存失败可重试；结束会话沿原流程回零并丢弃未保存数据。
- 采集与部署共用进程锁，避免同一套设备同时被两个本项目进程占用。锁本身不发送硬件指令。
- 默认输出目录改为仓库相对位置。HDF5 额外记录实际开始 baseline 和原触发阈值，供分析读取；原图像、action/state 单位、task/home 排列不变。
- **全部撤回 robot.py 的修改，文件与 `e4145bf` 完全一致。** 没有新增 CAN 状态门槛、连续反馈要求、SDK 控制判断、运动轨迹或回零同步策略。

主要文件：`scripts/PiperRecorder/record.py`、`piper_recorder/control.py`、`piper_recorder/dataset.py`、`src/easy_embodied_deploy/recorder_console.py`、`web/recorder/`。

## 2. PiperViewer：轨迹浏览

- 页面资源和 API 改为支持统一工作台子路径，同时保留原独立页面。
- 修复文件名含字面量 `%20` 等字符时的二次 URL 解码，防止读取错误路径。
- NumPy NaN 等值转为合法 JSON 空值，避免前端整体解析失败。
- 使用公共 HTTP 边界处理和实际端口显示，改善断开连接等情况下的服务行为。
- 三路图像、逐帧时间轴、动作曲线及 HDF5 读取语义不变。

主要文件：`scripts/PiperViewer/piper_viewer/server.py`、`static/app.js`、`static/index.html`。

## 3. LerobotDataProcessor：转换

- 修复显式输入 0 被当作“未填写”而替换成默认值；帧率等整数参数拒绝布尔值、非整数和无穷值。
- 转换前、episode 后和最终验证后检查取消状态，防止取消后继续创建或提交输出；退出服务会等待工作线程结束。
- `ffprobe` 增加调用超时，避免异常视频使验证无限等待。
- 统计验收在方差空间考虑官方 float32 流式统计的舍入误差，减少近常量轴的 std 假失败；不重写官方统计量。
- 页面支持工作台子路径和请求异常处理。
- 原 task 选择、尾部静止裁剪、固定网格降采样、home 排除、absolute/delta 定义及官方写入器不变。

主要文件：`scripts/LerobotDataProcessor/app.py`、`processor.py`、`static/`，以及公共 `statistics.py`。

## 4. DataAnalysis：质量分析

- 新采集文件使用保存的实际 baseline/阈值分析起始动作，旧文件仍按原格式兼容处理。
- 源数据校验错误归入对应 episode，避免总报告失败而 episode 行看起来正常。
- 多目录 source override 遇到同名文件时拒绝歧义，防止误关联另一条示教。
- 异常结构、缺失元数据及 NaN 返回可解析的失败报告，避免页面被非标准 JSON 中断。
- 删除异常 episode 时重新核对报告任务编号；失败会退出 deleting 状态；拒绝删空整个数据集。
- 保留原数据和 `processing.json` 的完整备份，支持人工回溯。退出时取消并等待分析线程。
- 原时序/物理间隔分析、源数据逐值核对和官方 episode 编辑流程不变。

主要文件：`scripts/DataAnalysis/analysis_app.py`、`analyze_hdf5.py`、`analyze_lerobot.py`、`static/`。

## 5. LerobotTrainer：LeRobot 训练

- 修复验证期间点击停止仍可能启动训练的竞态；服务关闭后拒绝新启动。
- 修复停止状态被后续日志覆盖、结束后计时继续增长；清理子进程和日志管道。
- 原训练器创建输出目录后即写 running manifest；中途 checkpoint 也能找到显式动作语义，完成后更新状态。
- 恢复前检查显式 action mode 契约，改进整数/布尔参数校验。
- 两套训练复用只读 GPU 查询，缺失遥测显示为空，避免 N/A 导致页面失败。
- 页面支持统一工作台子路径与轮询异常恢复。
- 原生 LeRobot PI05、LoRA/全参数、归一化、优化器、RTC 训练和 checkpoint 保存/恢复算法不变。

主要文件：`scripts/LerobotTrainer/trainer.py`、`web_server.py`、`web/`。

## 6. OpenpiTrainer：OpenPI 训练

- 包含与 LeRobot 训练相同的进程取消、状态、计时、资源清理和 GPU 显示修复。
- 从本机 manifest 所在目录发现 checkpoint，修复迁移清单中旧 `/root/...` 路径导致整个页面 bootstrap 失败。
- 归一化资产缓存按 stats 内容分目录，避免同 repo_id、不同数据集覆盖同一缓存；checkpoint 内的 asset_id 保持原兼容方式。
- LeRobot v3 适配逐行核对 14 维、有限数值、episode 索引覆盖/重叠、任务与时间戳，防止错误 reshape 和跨 episode padding。
- 恢复前检查显式 action mode；页面支持子路径与请求异常处理。
- JAX/OpenPI 原生模型、损失、RTC、FSDP、优化器与 checkpoint 算法不变。

主要文件：`scripts/OpenpiTrainer/trainer.py`、`web_server.py`、`dataset_v3.py`、`openpi_config.py`、`web/`。

## 7. LerobotDeployer：LeRobot 部署

- 修复完整启动或单独启动过程中取消后仍生成子进程的竞态；关闭后的服务不再接受启动。
- 明确本控制台的进程所有权：要求使用本页启动且地址匹配的硬件桥，识别两种外部部署进程，避免误接管。
- 模型加载失败后清理本页桥接进程；停止时给原硬件退出流程足够时间，避免重复信号打断已有回零。
- 启动前核对显式训练 action mode 与 trained RTC 范围；没有 manifest 的旧模型仍允许原显式选择。
- 网页增加原 CLI 已支持的 dry-run，显示诊断；补充数值参数和 RPC 超时校验。
- 硬件桥增加设备进程锁与 bind 失败时的 Socket 清理；**预定位和回零清理注册顺序已恢复原版**。
- delta 累加、absolute 直通、限位、绝对目标低通、30 Hz 控制、原生 RTC 队列及 SDK 动作调用不变。

主要文件：`scripts/LerobotDeployer/web_server.py`、`LerobotDeployer.py`、`bridge_client.py`、`hardware_bridge.py`、`web/`。

## 8. OpenpiDeployer：OpenPI 部署

- 包含与 LeRobot 部署相同的启动取消、进程所有权、加载失败清理及退出等待修复。
- 网页增加原 CLI 的离线数据输入入口；离线运行直接使用数据集，不启动硬件桥。
- 布尔参数严格解析，改善页面轮询和异常反馈，支持工作台子路径。
- 同步推理阻塞后更新调度截止时间，避免用过期 deadline 追赶并连发积压 tick；处理方式与 LeRobot 同步循环一致。
- OpenPI 原生模型、归一化、guided/trained RTC 算法及动作转换不变；共用恢复后的原硬件实现。

主要文件：`scripts/OpenpiDeployer/web_server.py`、`OpenpiDeployer.py`、`web/`。

## 统一工作台与公共代码

- 新入口 `python3 workbench.py`，默认 `http://127.0.0.1:8088`。八模块按各自环境按需启动，自动分配内部端口，通过 `/modules/.../` 集成。
- 新增模块卡片、导航、状态、服务日志、重试和停止；切换模块保留 iframe 状态，支持桌面和手机布局。
- 打开网页不连接或移动机械臂；连接、采集、训练及部署必须在模块内启动。关闭浏览器不会终止后台任务。
- 共用 JSON 读取、同源控制请求检查、请求体上限、断开连接处理和严格 JSON 序列化。
- 共用进程锁、checkpoint 动作契约、GPU 查询和统计验收工具；不升级 SDK、环境锁或第三方算法。
- 增加本机回归、HTTP 数据链路、Chromium 页面与踏板协议测试。GitHub Actions 只运行无需硬件/权重的语法和进程生命周期检查，完整数据/GPU 测试在本机运行。

主要文件：`workbench.py`、`src/easy_embodied_deploy/`、`web/workbench/`、`tests/`、`.github/workflows/checks.yml`。

## 验收边界

本机回归矩阵为 100 项通过，原有 59 项保留；真实硬件对照和恢复后 Web 的连接、预定位、跟随、结束与回零已通过。
完整人工示教、真实踏板的焦点习惯以及模型完成物理任务的效果，需要使用者实际操作；自动测试不会伪造这些结论。
本次短程模型验证用于检查数据/训练/checkpoint/推理衔接，不代表一两步训练已经收敛。

本次 Web 链路补测结果将记录在 [统一 Web 验收记录](WEB_ACCEPTANCE.md)。

## 2026-09-16：目录管理、相机标定和 OpenPI 指标

以下是基于 `8916cd6` 的本轮增量；上文是此前已经完成的仓库改动。

| 模块 | 本轮具体改动 |
| --- | --- |
| DirectoryManager（新增） | 浏览 models / datasets；新建文件夹；文件和目录重命名；复制绝对路径、当前目录相对路径；名称筛选、上一级、磁盘剩余空间；删除到回收站、恢复、输入名称后永久删除；防止越界、覆盖和跟随符号链接；阻止可识别的任务路径引用 |
| CameraCalibration（新增） | 三台 RGB 相机的序列号配置和预览；读取实际视频 profile 的 SDK 内参；棋盘 / ChArUco 自定义参数及可打印 SVG；静止采样、角点方向检查、只读从臂法兰位姿或手动位姿；样本持久化、查看、移除；内参重标定及应用；左右腕部与全局双 base 四项外参；质量摘要和单样本误差；按安装版本组合导出、复制路径和 JSON 下载 |
| OpenpiDeployer | 每个 tick 持续记录推理延迟和累计计数；饿死计数不再受 240 行显示窗口影响；通过已有观测响应携带桥接累计统计，不增加 RPC 心跳或 SDK 反馈读取；显示实时推理中位数、最近 / 最大耗时、累计饿死、全程最低队列、限幅次数和实测控制频率；修复有曲线时仍显示“等待数据”的样式；默认每次运行独立持久日志及结束汇总 |
| 共享桥接响应 | observe 响应新增已有 command_loop 统计字段，客户端缓存该字段供显示；不改变模型观测字典、动作、反馈采样、看门狗或归位逻辑 |
| 统一工作台 | 八模块增至十模块；新增资源分组、目录管理与标定入口；模块数量动态显示；继续按需启动与代理，并共享新页面样式 / HTTP 基础代码 |
| 文档 | 新增标定操作说明、位姿动作数据流待审方案、本轮验收记录；README 增加入口和参数说明；本文件列出增量 |
| 回归测试 | 新增文件系统边界、HTTP 采样、几何真值、鱼眼、板配置、内参和导出测试；新增 OpenPI 实时指标及累计计数测试；新增两个新模块的浏览器交互回归；原浏览器测试更新为十模块 |

新后端位于 `src/easy_embodied_deploy/{files,files_console,calibration,calibration_math,calibration_console,module_http}.py`，两个脚本目录仅提供启动入口。新前端位于 `web/files/`、`web/calibration/` 和 `web/shared/`。

默认棋盘参数是 **4×6 内角点、53.5 mm 单格**，可修改，参数随记录保存。绝对位姿设计采用左右各自 base，**不需要相机标定**；camera delta 的方案依据 Qwen-RobotManip 公式 (5)，明确冻结当前相机参考系和 6D → rotvec 表示。

位姿数据转换、位姿模型训练和位姿动作执行均尚未加入。采集器、数据转换器、两个训练器、LeRobot 策略部署器及第三方 SDK 在本轮没有修改。共享硬件桥只在已有响应中附带统计字段，客户端只缓存该字段；控制循环和底层硬件调用没有变动。OpenPI 的策略、RTC 排队、动作转换、限幅和归位算法也没有变动。

完整测试证据与现场验收边界见 [本轮验收记录](RESOURCES_CALIBRATION_ACCEPTANCE.md)。


## 2026-09-16：统一界面、出厂内参与简化位姿说明

本次增量基于 `f4c8099`。位姿功能按最新要求暂停，未新增位姿转换、训练或部署代码。下表为本次最终保留的修改。

| 模块 | 本次修改 |
| --- | --- |
| 相机标定 | 增加“当前使用的内参”卡片，求解前显示实际来源、序列号、视频分辨率、K、D；打开相机显示出厂值，选中历史记录显示记录中的值；说明新建时自动保存的位置；手工重标定移入高级选项，只显示同相机、同传感器和分辨率的记录，没有可用选择时禁用按钮；取消选择记录时清空旧样本与内参显示；增加网页使用顺序和明确矩阵方向 |
| PiperViewer | 顶栏、路径输入、按钮和配色与工作台统一；视频上的深色字幕遮罩保留可读性；播放、曲线和文件读取逻辑不变 |
| 数据转换 | 浅色卡片、绿色主按钮、状态色、表单与日志统一；调整字号、圆角、窄屏布局；修正 fieldset 标题占位，让采样频率字段排列清楚 |
| 数据质量分析 | 标题、状态区、检查与报告卡片、表格、异常提示统一；表单、分析、删除和导出接口不变 |
| LeRobot / OpenPI 训练 | 统一中文页标题、表单、训练预设、GPU 卡片、日志、弹窗和图表配色；提高小字号和窄屏可读性；预设、校验、命令生成、启动 / 停止及训练逻辑不变 |
| LeRobot / OpenPI 部署 | 统一部署配置、流程按钮、设备状态、日志和控制循环图表；保持停止按钮红色；指标、RTC 和控制逻辑不变 |
| 目录管理 | 沿用已有浅色布局；共享样式使复选框与主题绿一致；文件操作行为不变 |
| PiperRecorder / 统一工作台 | 保留原布局、脚踏板习惯和启动方式；本次没有修改相关逻辑 |
| 文档 | 位姿文档缩减为数据来源、两类动作、Qwen 公式 (5)、6D 到 3D rotvec、训练 / 执行边界；去掉 32 维输入设计；标定说明增加出厂内参保存位置、法兰与 base 的区别、四项标定顺序及后续衔接；README 和验收记录同步 |
| 测试 | 新增十模块展开控件后的桌面 / 平板 / 手机布局回归、训练预设和矩阵方向切换；扩充内参求解前展示、清除历史选择、实际指标展示测试；修正浏览器测试在严格 CSP 和离线无硬件频率场景下的断言 |

本次未修改 `src/`、第三方 SDK 或 `scripts/` 下的 Python 后端。四个训练 / 部署页面的 JavaScript 仅变更图表颜色；七个原模块的 DOM ID 保持一致。详细测试证据和实物验收边界见 [界面与内参验收](UI_INTRINSICS_ACCEPTANCE.md)。

退出示教后的补测已覆盖原桥保持 / J6 小步响应、两套部署真实相机推理和退出回零、采集页 Enter 真机流程及十模块布局。补测期间仅修正 `tests/browser_theme.py` 的模块冷启动等待，未修改产品或硬件控制代码；0.5° 请求未见响应及 2° 测试的实际误差均如实记入验收文档。
