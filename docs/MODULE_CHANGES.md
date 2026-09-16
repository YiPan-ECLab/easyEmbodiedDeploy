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
