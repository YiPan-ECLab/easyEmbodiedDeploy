# Piper HDF5 Viewer

本地、只读的双臂 HDF5 数据浏览器。它支持目录扫描、三相机同步播放、动作与反馈曲线、逐帧数值比较，以及 HDF5 数据集结构检查。

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
