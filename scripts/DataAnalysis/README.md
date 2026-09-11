# DataAnalysis

两个只读、CPU-only 的数据质量检查工具，以及共用的本地 Web 前端。失败时退出码为 `1`，适合手工验收或 CI。

启动 Web 前端：

```bash
thirdParty/.venvs/lerobot-pi05/bin/python scripts/DataAnalysis/DataAnalysis.py
```

打开 <http://127.0.0.1:8767>。端口被占用时使用 `--port 0` 自动选择空闲端口，或用 `--port 8892` 显式指定。界面可切换 HDF5/LeRobot、启动或停止检查、筛选异常、查看逐文件或逐 episode 指标，并导出完整 JSON 报告。分析子进程强制使用 CPU，不会占用 GPU。

检查 Piper Recorder HDF5 文件或整个目录：

```bash
CUDA_VISIBLE_DEVICES='' thirdParty/.venvs/piper-home/bin/python \
  scripts/DataAnalysis/analyze_hdf5.py datasets/hdf5/your_dataset \
  --expected-fps 30 --json /tmp/hdf5_quality.json
```

检查 DataProcessor 生成的 LeRobot bundle：

```bash
CUDA_VISIBLE_DEVICES='' thirdParty/.venvs/lerobot-pi05/bin/python \
  scripts/DataAnalysis/analyze_lerobot.py datasets/lerobot/your_dataset \
  --json /tmp/lerobot_quality.json
```

LeRobot 工具默认从旁边的 `processing.json` 找到源 HDF5，并逐 episode 重算
固定网格降采样、示教动作尾帧裁剪以及 absolute/delta 动作，同时核对 Parquet、任务与 episode 元数据、官方统计量和三路视频。源目录移动后可用 `--source-dir` 指定。
