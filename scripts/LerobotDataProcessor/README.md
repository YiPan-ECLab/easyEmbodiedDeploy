# DataProcessor

从项目根目录启动本地网页：

```bash
thirdParty/.venvs/lerobot-pi05/bin/python scripts/LerobotDataProcessor/DataProcessor.py
```

打开 <http://127.0.0.1:8766>。网页会先检查每个 HDF5 的时间戳，只有目标 FPS 能整除每个文件的源 FPS 时才允许处理。流程固定为：

```text
原始 HDF5 → 只选 task phase → 内存中降采样 →（可选）action[t+1]-action[t] → LeRobot v3
```

PiperRecorder 在任务结束后还会记录 `trajectory_phase=1` 的回零过程。处理器会排除这些帧，只保留 `trajectory_phase=0` 的任务演示；这样既不会训练模型执行回零，也不会把 task→home 边界变成一个巨大的增量 action。没有 `trajectory_phase` 的旧数据仍按全部帧处理。

无界面方式：

```bash
thirdParty/.venvs/lerobot-pi05/bin/python \
  scripts/LerobotDataProcessor/convert_piper_hdf5.py \
  --input-dir datasets/hdf5/撕下快递单并贴到桌子上的快递上 \
  --fps 10 --action-mode delta_joint
```

处理采用流式方式：每个 HDF5 降采样后立即写入 LeRobot，不在输出目录保留中间 HDF5，因此只生成 `lerobot/` 和 `processing.json`。输出目录已存在时脚本会拒绝覆盖。
