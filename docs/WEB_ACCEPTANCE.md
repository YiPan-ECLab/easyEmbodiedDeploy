# 统一 Web 验收记录

日期：2026-09-16。本次复测在恢复原版 Piper 硬件实现之后进行。
工作台：`http://127.0.0.1:8088`；启动命令：仓库根目录 `python3 workbench.py`。

统一样式、出厂内参及退出示教后的最新真机补测见 [界面与内参验收](UI_INTRINSICS_ACCEPTANCE.md#退出示教后的真机补测)。本页保留真实数据转换和两套实训的完整链路记录。

## 已通过的链路

| 入口 | 实测与结果 |
| --- | --- |
| Piper 采集 | 原版 CLI 与恢复后的 Web：四臂/夹爪连接、初始回零、J6 −45°/+45°、跟随启动、结束后回零、正常退出均通过；另在真实浏览器中点击连接并按 Enter，验证长按跨阶段不误结束、再次 Enter 结束与回零、空 task 拒绝保存和结束会话 |
| 轨迹浏览 | 通过统一代理读取真实示教的 440 帧；三路相机第 170 帧均返回有效 JPEG |
| HDF5 分析 | 真实示教 PASS，343 个 task 帧、97 个 home 帧；原 task 尾部 5 帧静止目标被识别 |
| 数据转换 | 同一真实示教分别转为 delta_joint 和 absolute_joint；10 Hz 输出分别为 113、114 帧，官方写入和验证成功 |
| LeRobot 分析 | 两种转换结果均 PASS；action/state 与原源文件的最大误差均为 0；home 帧混入数为 0 |
| LeRobot 训练 | 通过统一 Web 的验证、预览和启动 API，使用上述真实 absolute 数据实际训练 1 步 trained RTC LoRA，loss 0.502、grad_norm 0.576，保存模型/processor/optimizer checkpoint，退出码 0 |
| OpenPI 训练 | 使用同一转换数据，通过统一 Web 实际训练 1 步 trained RTC LoRA，loss 0.1542、grad_norm 1.3436，保存 params/assets/train_state，退出码 0 |
| LeRobot 部署 | 在统一工作台 iframe 内实际填写上述新 checkpoint，勾选 dry-run、点击“启动完整流程”；真实相机观测、3 个有效 trained RTC 动作、诊断和退出回零均通过，模型/桥退出码均为 0 |
| OpenPI 部署 | 从对应 iframe 实际填写新 checkpoint 并点击启动；真实相机观测、3 个有效 trained RTC 动作、诊断和退出回零均通过，模型/桥退出码均为 0 |
| 浏览器与回归 | 八页面及资源加载、桌面/手机布局、Enter 长按去重、保存/丢弃协议通过，无 JS 页面错误；项目回归 100 项通过 |

部署短测使用 Policy 2 Hz、Control 30 Hz、max_steps=3，两个 checkpoint 的训练 max_delay 均为 5。
两套运行诊断均为 `runtime_starved_ticks=0`、`move_js.cycles_sent=0`。
LeRobot 首次队列预填充有 2 个等待 tick，之后有 3 个有效动作；OpenPI 的预填充发生在主循环前。
机械臂仅执行原启动/退出所需的回零和 J6 预定位，没有执行测试模型输出。
这验证了接口衔接和执行链路，不代表长时间 RTC 性能或一两步权重已达到实际任务能力。

此前的同步/guided/trained 推理、训练恢复及原生 RTC 单元测试记录仍见[仓库审查记录](REPOSITORY_REVIEW.md)。
本次并未把离线推理等同于真机任务效果，或把浏览器模拟输入等同于真实脚踏板实物验收。

## 数据和产物

真实输入为仓库已有人工示教：

`datasets/hdf5/撕下快递单并贴到桌子上的快递上/pyagxarm_dual_20260910_153312_407067.hdf5`

只在测试目录创建到该文件的链接；原文件未修改。转换结果、两套 checkpoint、API 报告、浏览器截图和部署诊断均保留于：

`outputs/web-acceptance-20260916/`

此目录由 Git 忽略，不会推送数据或权重。测试 checkpoint 仅用于链路验收。

本机执行日志：

- `/tmp/easy-web-full-acceptance.log`：真实文件回放、分析、两种转换、两套 Web 实训。
- `/tmp/easy-web-deploy-browser.log`：实际浏览器填写配置、两套部署 dry-run 与回零。
- `/tmp/easy-web-acceptance-browser.log`：八模块页面和踏板协议。
- `/tmp/easy-recorder-web-retest.log`：恢复原版后 Web 的硬件采集流程。
- `/tmp/easy-web-recorder-browser.log`：真实浏览器 Enter → 原版机械臂流程 → 结束并归零。
- `/tmp/easy-recorder-recheck-regression.log`：100 项项目回归。

上述日志及临时测试脚本另存于测试产物目录的 `logs/` 和 `harness/`，供本机复核；它们不参与产品运行。

## 留给使用者的实际验收

主臂连接、预定位和回零已经补测通过，不再是阻塞。剩下的是必须由真人完成的任务环节：

1. 在工作台进入 Piper 采集，选择自己的输出目录，点击连接设备并回零。输入任务指令，用真实踏板进入预定位后完成一条短示教，再踩停、等待回零、踩保存。下一条留空沿用指令，并试一次丢弃。
2. 在轨迹浏览打开刚保存的文件，检查三路图像、task/home 和操作内容；运行 HDF5 分析，再分别按需要选择 absolute 或 delta 转换，检查 LeRobot 分析结果。
3. 选择刚转换的数据进行短程训练，检查 checkpoint，再在对应部署页面先勾选 dry-run 验证加载、诊断与停止。真实任务效果使用已有充分训练的权重现场评估。

页面内的路径是可编辑配置；训练输出请选择新目录，部署动作表示须与相应训练数据一致。
真实踏板的键盘焦点、人工示教保存以及任务成功率尚未由自动测试代替验收。
