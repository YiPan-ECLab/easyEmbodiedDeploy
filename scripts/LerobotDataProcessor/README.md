# DataProcessor

从项目根目录启动本地网页：

```bash
thirdParty/.venvs/lerobot-pi05/bin/python scripts/LerobotDataProcessor/DataProcessor.py
```

打开 <http://127.0.0.1:8766>。端口被占用时可传 `--port 0` 自动选择空闲端口，或用 `--port 8891` 显式指定。网页支持输入预检、转换启动与取消、逐 episode 进度、日志、结果复验、命令预览和响应式移动端布局。服务强制隐藏 GPU，只执行 CPU 数据处理。

网页的“输入 HDF5 目录”支持每行填写一个目录，多个目录按填写顺序合并成一个 LeRobot 数据集；命令行可重复传入 `--input-dir` 达到同样效果。网页会先检查每个 HDF5 的时间戳、轨迹阶段、Piper 限位和相机同步，对未融合录制，只有目标 FPS 能整除源 FPS 时才允许处理（融合输入见文末）。默认配置是自动识别源频率、`10 Hz + delta_joint + reject`。流程固定为：

```text
原始 HDF5 → 只选 task phase 并裁掉首尾静止重复帧 → 内存中降采样 →（可选）action[t+1]-action[t] → LeRobot v3
```

PiperRecorder 在任务结束后还会记录 `trajectory_phase=1` 的回零过程。处理器只接受带 `trajectory_phase` 的原始录制，排除这些帧，并按示教主臂的 absolute action 裁掉任务结束至按下回车之间的连续等待帧。这项规则保留用于关节模式。位姿模式预测从臂未来位姿，会同时检查命令、从臂 state 与法兰反馈，保留从臂继续收敛的片段。这样既不会训练模型等待或执行回零，也不会把 task→home 边界变成一个巨大的增量 action。阶段必须是连续的 task(0) 前缀和可选 home(1) 后缀。

无界面方式：

```bash
thirdParty/.venvs/lerobot-pi05/bin/python \
  scripts/LerobotDataProcessor/convert_piper_hdf5.py \
  --input-dir datasets/hdf5/撕下快递单并贴到桌子上的快递上 \
  --input-dir datasets/hdf5/another_recording \
  --fps 10 --action-mode delta_joint --gap-policy reject
```

主要选项：

* `--fps`：输出 LeRobot 的固定帧率。它是模型训练/部署使用的动作时间基准，不读取数据集元数据中的 FPS；必须整除每个输入文件的实际源 FPS。
* `--source-fps`：可选地显式指定所有输入文件的源 FPS，默认从每个文件的 timestamp 中检测。
* `--action-mode absolute_joint|delta_joint`：绝对关节目标，或在任务段降采样后计算 `action[t+1]-action[t]`。旧拼写 `absolute` 仍兼容，但输出报告统一写成 `absolute_joint`。
* `--gap-policy compact|reject`：`reject`（默认）遇到时间空档直接拒绝；`compact` 会把空档压缩到固定 FPS 并在检查报告中警告，仅在明确接受改变物理时间时使用。
* `--limit`：只处理排序后的前 N 个 HDF5，`0` 表示全部。
* `--output-root`、`--repo-id`：指定输出目录和 LeRobot 本地 repo id。输出目录必须不存在，脚本不会覆盖已有数据。
* `--inspect`：只扫描输入并输出计划，不创建数据集。
* `--validate-output PATH`：只对已有 `PATH`（数据集目录或其父目录）做 CPU 结构、统计量、Piper 限位和视频帧数验收；若路径旁没有 `processing.json`，可再给 `--validate-action-mode absolute_joint|delta_joint`。

处理采用流式方式：每个 HDF5 降采样后立即写入 LeRobot，不在输出目录保留中间 HDF5，因此只生成 `lerobot/` 和 `processing.json`。完成后会自动执行同一套 CPU 验收。LeRobot `finalize()` 负责生成官方统计量，处理器不会重写其 q01/q99 语义。

## 位姿动作与相对关节更新

支持 `absolute_base_pose`、`camera_delta_wrist`、`camera_delta_global`；state 保持左右关节 / 夹爪 14 维，位姿 action 是双臂 xyz / 三维 rotvec / 绝对夹爪 14 维。
详见 [完整位姿链路](../../docs/POSE_ACTION_DATAFLOW.md)、[标定操作与训练顺序](../../docs/CAMERA_CALIBRATION.md) 和 [相对关节问题排查](../../docs/JOINT_ACTION_AUDIT.md)。位姿数据必须保留 `meta/pose*` 附属文件，并通过本项目训练入口读取。camera delta 部署需确认相机安装版本；绝对 base 位姿不需要标定。

## 单臂轨迹与转换边界

允许只操作左臂、只操作右臂、只操作夹爪，以及另一侧先静止后参与任务。
静止侧仍须提供有效反馈，输出仍为双臂 14 维；absolute 保留静止目标，delta 对恒定轴输出零。
`action_source_timestamps` 表示最后一次接受命令的时间，静止侧长期不更新是正常的。

| 边界 | 当前规则与原因 |
| --- | --- |
| 开始 / 结束 | 采集任一侧超过启动阈值即开始；task 至少两帧，home 不补足任务长度。转换裁剪、降采样后，delta / pose 仍须至少两帧 |
| 阶段 / 帧率 | 只接受连续 task 前缀及可选 home 后缀；源频率仅从 task 时间戳估计，避免回零段主导短任务的频率 |
| 静止首尾 | 原始和融合序列均裁剪首尾重复帧，保留首次变化前的锚点和最终目标。关节模式比较双臂全部 14 维命令，位姿模式额外检查 state 与双臂法兰反馈；中间停顿和单侧静止保留 |
| 固定采样网格 | 先降采样再差分，不跨 home；最终变化若落在最后一个网格点之后，会在预检中警告，可提高目标 FPS 或在最终动作后稍等再结束 |
| 时间间隔 | 采集默认只拒绝超过 0.35 s 的 task 空档；转换 reject 要求选中帧间隔在目标周期的 1/1.5 到 1.5 倍之间，二者用途不同，因此成功采集不保证严格转换一定通过 |
| 物理量 / 图像 | 14 维轴序、单位、有限值、Piper 限位、三路 RGB 及相机时序必须有效；静止不等于反馈缺失 |
| 统计验收 | LeRobot 累加器使用 float64 计算统计，避免长恒定 / 近恒定轴的 float32 方差误差；Parquet 的 state/action 仍为 float32 |
| 输出 / 取消 | 输出目录必须不存在且与输入分离；失败或取消保留对应状态报告，成功需通过统计、结构和视频验收 |

2026-10 边界检查修复了统计精度、回零段影响源帧率及单帧 task 被 home 帧掩盖的问题。
回归覆盖左右单臂和只动夹爪的完整 HDF5 → LeRobot 转换、长恒定轴、多批次及多 episode 统计。
旧失败目录的统计不会自动更新，应转换到新目录；没有通过修改验收容差来放行错误统计。

后续可选优化：采集时实时提示会被严格转换拒绝的短空档；用少量启动前缓存保留跨越启动阈值的首段动作；
长录制的尾部裁剪可按块反向读取 HDF5，减少逐帧访问。这些分别涉及操作提示、起始帧语义和性能，
不是单臂轨迹失败的必要修复。本次保留原有启动阈值和关节模式的尾部裁剪语义。

## 人工部分融合输入

在工作台「轨迹浏览」完成片段标注并导出融合 HDF5 后，可直接将导出目录作为本模块输入。采样频率标记为 `mixed`，图像、状态和绝对控制量已经按人工选择的索引同步保留。

对这种输入，转换不再按源/目标 FPS 二次取样，但仍先排除 HOME，再裁剪首尾静止重复帧。目标频率仅用于 MP4 编码和 Parquet 的逻辑帧索引时间轴；LeRobot `info.json` 的 `fps` 为 `"mixed"`，数值编码频率独立保存为 `video_fps`。关节差分和三种位姿未来目标以融合后的相邻帧重新计算，位姿 sidecar 保留真实、非均匀的时间戳。原始未融合 HDF5 同样执行首尾裁剪，再按原固定网格取样。

原始录制和融合录制可一起输入。只要包含融合 episode，输出数据集的采样频率即标为 `mixed`。训练不要求真实采集间隔均匀，部署 `policy_hz` 独立配置。

处理分工：Recorder 负责启动阈值及 task/home 阶段标记；Viewer 融合导出时排除 HOME，按标注抽帧并保留所选序列的首尾；DataProcessor 统一执行首尾静止裁剪，并再次按阶段标记排除 HOME，然后重算动作。DataAnalysis 只检查和复算，不修改 HDF5。
