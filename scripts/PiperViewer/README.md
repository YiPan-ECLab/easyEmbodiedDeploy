# Piper HDF5 Viewer

本地双臂 HDF5 浏览与部分融合工具。支持目录扫描、三相机同步播放、动作与反馈曲线、逐帧数值比较，以及采集后的片段抽帧融合。始终保留原始录制。

## 启动

```bash
/home/py/main/easyEmbodiedDeploy/thirdParty/.venvs/piper-home/bin/python \
  /home/py/main/easyEmbodiedDeploy/scripts/PiperViewer/viewer.py
```

指定初始目录或端口：

```bash
/home/py/main/easyEmbodiedDeploy/thirdParty/.venvs/piper-home/bin/python \
  /home/py/main/easyEmbodiedDeploy/scripts/PiperViewer/viewer.py \
  --directory /path/to/hdf5_directory --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。页面顶部可以随时输入另一个 HDF5 目录。

Viewer 不修改源文件，只接受当前 Recorder 的 `pyagxarm_direct_v1` 结构。`actions` 是 pyagxarm 直接控制量：关节为 `rad`，夹爪宽度为 `m`；文件中不包含归一化动作。

## 部分融合

在工作台打开「轨迹浏览」，选择一条已采集完成的 episode，在三路画面下方操作：

1. 拖动时间轴或输入帧号定位。点击「当前帧设为起点/终点」，也可用 `[` / `]` 快捷键。
2. 设置片段倍率，点击「添加片段」。例如倍率 3，在该段每 3 个基础步长保留一帧。可添加多个片段、编辑或删除。
3. 点击「预览融合」，用播放或左右方向键检查。时间轴、帧号和时间读数仍指向原始录制，播放与单步只经过保留帧。
4. 点击「导出融合 HDF5」。默认导出到原目录旁的 `<目录名>_fusion`，每次生成新文件；完成后可直接打开检查。
5. 在「数据转换」选择导出目录，照常选择关节或位姿动作模式。融合文件不再被二次降采样；转换模块仍会裁剪首尾静止重复帧并排除回零段。

基础步长默认 1：未标注的复杂片段逐帧保留。例如原采集 30 Hz、希望普通片段约每 3 帧取一次、简单片段约每 9 帧取一次，设基础步长 3、简单片段倍率 3。首帧、末帧和每个片段的边界始终保留，因此边界可能产生较短步长。界面帧号从 1 开始，起止均包含；片段不能重叠但可共用边界。HOME 不参与融合。融合模块保留选定序列的首尾，首尾静止重复帧由后续转换模块裁剪；中间停顿由人工决定是否抽稀。

这是选择完整同步帧并串成新序列，不平均图像、关节或位姿。图像、state、绝对 action、末端位姿都使用相同索引。增量关节 action 在转换时用新序列重新差分；位姿未来目标及固定锚点 chunk 也按新序列生成。

标注添加/编辑/删除后自动保存为源文件旁的 `*.data_demo_0.fusion.json`；基础步长和预览速度更改也会保存。重新打开源文件可以继续编辑。原文件改变时不会静默套用旧标注。导出文件可再次融合，保留完整来源链。

## 时间语义

- 融合 HDF5 根和 demo 的 `fps` 为文本 `mixed`；demo 的 `record_hz` 同样标为 `mixed`，原值保留在 `source_record_hz`。
- `timestamps`、相机及动作源时间戳保留实际采集时间，允许非均匀间隔；`temporal_provenance` 保存原始帧号、直接输入帧号和原始时间戳，demo 的 `temporal_fusion` 属性保存标注及来源。
- `playback_fps` 仅用于 Viewer 预览，不是采样频率。它决定预览多久取一次保留帧。
- 转换后 LeRobot 的 `meta/info.json` 使用 `fps: "mixed"`，`video_fps` 单独描述 MP4/Parquet 的逻辑索引时钟。两个训练入口已经适配：只按融合后的序列取 chunk，不按实际时间重新插值或限制跨度。
- 部署 `policy_hz` 独立配置。能否从这些示教中学到预期快慢行为，需要实际训练和部署实验验证。

`fps: "mixed"` 是本仓库对 LeRobot 元数据的扩展；外部未适配的原版 LeRobot 读取器可能要求数值 fps。仓库内的 Torch/JAX 训练和数据分析均支持这一扩展。
