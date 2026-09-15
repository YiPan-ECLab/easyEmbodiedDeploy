# DataProcessor

从项目根目录启动本地网页：

```bash
thirdParty/.venvs/lerobot-pi05/bin/python scripts/LerobotDataProcessor/DataProcessor.py
```

打开 <http://127.0.0.1:8766>。端口被占用时可传 `--port 0` 自动选择空闲端口，或用 `--port 8891` 显式指定。网页支持输入预检、转换启动与取消、逐 episode 进度、日志、结果复验、命令预览和响应式移动端布局。服务强制隐藏 GPU，只执行 CPU 数据处理。

网页的“输入 HDF5 目录”支持每行填写一个目录，多个目录按填写顺序合并成一个 LeRobot 数据集；命令行可重复传入 `--input-dir` 达到同样效果。网页会先检查每个 HDF5 的时间戳、轨迹阶段、Piper 限位和相机同步，只有目标 FPS 能整除每个文件的源 FPS 时才允许处理。默认配置是自动识别源频率、`10 Hz + delta_joint + reject`。流程固定为：

```text
原始 HDF5 → 只选 task phase 并裁掉尾部静止帧 → 内存中降采样 →（可选）action[t+1]-action[t] → LeRobot v3
```

PiperRecorder 在任务结束后还会记录 `trajectory_phase=1` 的回零过程。处理器只接受带 `trajectory_phase` 的原始录制，排除这些帧，并按示教主臂的 absolute action 裁掉任务结束至按下回车之间的连续等待帧。从臂可能在真人松手后继续收敛，因此不能把 follower state 是否完全静止作为尾帧判据。这样既不会训练模型等待或执行回零，也不会把 task→home 边界变成一个巨大的增量 action。阶段必须是连续的 task(0) 前缀和可选 home(1) 后缀。

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
