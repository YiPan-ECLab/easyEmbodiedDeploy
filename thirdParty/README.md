# easyEmbodiedDeploy 环境迁移说明

本文档用于把整个 `easyEmbodiedDeploy` 源码迁移到另一台 x86_64 Linux GPU
机器，并重建当前的三套 uv 环境。`datasets/`、`models/` 和
`thirdParty/.venvs/` 不进入 Git；数据集与模型权重需要单独复制。

## 当前基线与 GPU 兼容性

环境快照日期为 2026-09-09，当前机器是 Ubuntu 22.04.5 x86_64，NVIDIA
驱动 580.173.02。

| 环境 | 用途 | Python | GPU/CUDA 依赖 |
| --- | --- | --- | --- |
| `.venvs/piper-home` | Piper/Orbbec 采集、Viewer、硬件桥 | 3.10.12 | 无 CUDA 依赖；可在无 NVIDIA GPU 的机器运行 |
| `.venvs/lerobot-pi05` | 数据转换、PI05 训练和推理 | 3.12.14 | PyTorch 2.11.0+cu128，自带 CUDA 12.8 runtime |
| `.venvs/openpi-jax` | OpenPI π0.5（JAX）训练和推理 | 3.11.16 | JAX/JAXLIB 0.5.3，按 OpenPI `uv.lock` 安装 CUDA 12 wheel |

`lerobot-pi05` 当前关键版本为 Torch `2.11.0+cu128`、Torchvision
`0.26.0+cu128`、CUDA runtime `12.8`、cuDNN `9.19.0`、LeRobot `0.6.2`。
当前 wheel 原生包含 `sm_80` 与 `sm_120`，因此：

- NVIDIA A800（Ampere，`sm_80`）可以直接运行，使用 R570
  `570.86.10` 或更新驱动。
- NVIDIA RTX PRO 6000 Blackwell（`sm_120`）可以直接运行，使用 R575
  `575.51.03` 或更新驱动；推荐直接使用 R580 当前生产驱动。

无需安装系统级 CUDA Toolkit；PyTorch wheel 已包含运行时库。只有编译自定义
CUDA 扩展时才建议另装 CUDA Toolkit 12.8。驱动可以比 CUDA 12.8 随附版本更新，
例如当前的 580.173.02。

官方依据：[CUDA 12.8 Release Notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)、
[NVIDIA Blackwell 驱动表](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/deployment-considerations.html)、
[PyTorch cu128 安装索引](https://pytorch.org/get-started/previous-versions/)。

## 仓库内容与依赖记录

| 路径 | 作用 | 重建依据 |
| --- | --- | --- |
| `lerobot/` | 项目实际使用的 LeRobot 0.6.2 源码 | 源码自带 `uv.lock`；本机实装版本另见 runtime lock |
| `pyAgxArm/` | AgileX 机械臂 SDK 1.0.0 | Git submodule，固定到提交 `a13cd89fe17347dcced1f8c1d26ab2ed0c8a0d1b` |
| `pyorbbecsdk/` | Orbbec SDK 源码参考 | 实际运行使用 PyPI wheel `pyorbbecsdk2==2.1.2` |
| `openpi/` | OpenPI JAX 训练与推理源码 | `openpi/uv.lock` 是该环境的依赖锁定文件 |
| `lerobot-openpi/` | OpenPI 数据依赖的旧 LeRobot | 固定提交 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| `requirements-piper-home.txt` | 硬件环境的项目直接依赖 | 便于人工阅读 |
| `requirements-lerobot-project.txt` | LeRobot 之外的项目直接依赖 | 便于人工阅读 |
| `environment-locks/lerobot-pytorch-cu128.lock.txt` | Torch 与 Torchvision 的 cu128 wheel | PyTorch 官方 cu128 索引 |
| `environment-locks/*.runtime-lock.txt` | Piper/LeRobot 环境的精确版本 | `setup_envs.sh` 的安装入口 |
| `environment-locks/*.freeze.txt` | 包括环境工具在内的完整审计快照 | 仅用于审计和排障 |
| `setup_envs.sh` | 自动创建与检查两套环境 | 使用 runtime lock 和本地源码 |
| `setup_openpi_env.sh` | 创建 OpenPI 第三套环境 | 使用 OpenPI lock，并校验旧 LeRobot 提交 |

LeRobot 源码自带的 `uv.lock` 保留了上游依赖解析；为了严格复现本机后来安装的
补充包和补丁版本，重建脚本使用单独的 `lerobot-pi05.runtime-lock.txt`。Torch
和 NVIDIA cu12 包从官方 PyTorch cu128 索引安装。OpenPI 环境则完全按
`openpi/pyproject.toml` 与 `openpi/uv.lock` 安装，当前锁定 JAX/JAXLIB 0.5.3、
Flax 0.10.2、Torch 2.7.1 和 Python 3.11.16；不要把 OpenPI 的依赖装进
`lerobot-pi05`。不要手工逐行安装 freeze 文件。Orbbec 源码镜像中的多平台预编译
库、生成后的 HTML 文档和示例 ONNX 权重不上传；它们不参与本项目运行，安装脚本会
从 PyPI 获取带原生库的固定版本 wheel。

## 新机器首次部署

### 1. 系统准备

推荐使用 Ubuntu 22.04 x86_64，以便和当前机器保持一致。先安装 NVIDIA 驱动并
确认：

```bash
nvidia-smi
```

驱动版本按上表选择，并确保 `nvidia-smi` 能正确显示目标 GPU。安装通用系统工具：

```bash
sudo apt update
sudo apt install -y git curl ffmpeg can-utils libgl1 libglib2.0-0 libusb-1.0-0
```

安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

### 2. 克隆源码

首次克隆必须带 submodule：

```bash
mkdir -p /home/py/main
cd /home/py/main
git clone --recurse-submodules https://github.com/YiPan-ECLab/easyEmbodiedDeploy.git easyEmbodiedDeploy
cd easyEmbodiedDeploy
```

如果普通 clone 时忘了 submodule：

```bash
git submodule update --init --recursive
```

当前 Recorder 与 Viewer 的默认输出路径、部分 README 示例使用
`/home/py/main/easyEmbodiedDeploy`，因此部署到同一路径最省事。通过脚本参数指定
数据和模型路径时，其余主要入口可以从任意克隆目录运行。

### 3. 创建运行环境

```bash
cd /home/py/main/easyEmbodiedDeploy
bash thirdParty/setup_envs.sh
```

脚本会安装 Python 3.10.12 与 3.12.14，创建：

```text
thirdParty/.venvs/piper-home
thirdParty/.venvs/lerobot-pi05
```

已有环境时，检查而不修改：

```bash
bash thirdParty/setup_envs.sh --check
```

需要从零替换两套环境时：

```bash
bash thirdParty/setup_envs.sh --recreate
```

如 shell 已 source ROS，先执行 `unset PYTHONPATH AMENT_PREFIX_PATH
COLCON_PREFIX_PATH`。否则 ROS Humble 的 Python 3.10 包可能污染 Python 3.12
环境。

OpenPI 使用独立的 Python 3.11/JAX 环境，不会修改上面两套环境。创建或检查它：

```bash
# 创建（首次安装）
bash thirdParty/setup_openpi_env.sh

# 只检查，不下载、不重装、不占用 GPU
bash thirdParty/setup_openpi_env.sh --check

# 环境损坏或需要完全重建时使用
bash thirdParty/setup_openpi_env.sh --recreate
```

脚本执行以下固定步骤：

1. 用 Python 3.11.16 创建 `thirdParty/.venvs/openpi-jax`；
2. 按 `thirdParty/openpi/uv.lock` 安装 OpenPI、JAX CUDA 依赖和
   `openpi-client`；
3. 克隆并校验 `thirdParty/lerobot-openpi` 的提交
   `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，以 editable 方式安装它，避免
   使用其他 LeRobot 版本；
4. 清理 ROS 的 Python 环境变量，并用 CPU 完成导入和版本检查。

`thirdParty/lerobot-openpi` 是安装脚本生成的依赖目录，已加入 `.gitignore`，不会
提交到本仓库；重新克隆仓库后再次运行脚本即可恢复。

安装完成后查看版本和设备：

```bash
thirdParty/.venvs/openpi-jax/bin/python - <<'PY'
import jax
import torch

print("JAX", jax.__version__, jax.devices())
print("Torch", torch.__version__, "CUDA", torch.cuda.is_available())
PY
```

需要验证某一张 GPU 时显式指定单卡，例如：

```bash
CUDA_VISIBLE_DEVICES=0 thirdParty/.venvs/openpi-jax/bin/python - <<'PY'
import jax
print(jax.devices())
PY
```

训练和部署入口见 [`scripts/OpenpiTrainer/README.md`](../scripts/OpenpiTrainer/README.md)
和 [`scripts/OpenpiDeployer/README.md`](../scripts/OpenpiDeployer/README.md)。

### 4. 相机与 CAN 的一次性系统配置

Orbbec 相机在 Linux 下需要 udev 规则：

```bash
cd /home/py/main/easyEmbodiedDeploy/thirdParty/pyorbbecsdk
sudo python3 scripts/env_setup/setup_env.py
```

重新插拔相机后，再检查三个相机是否可见。Piper 使用 SocketCAN，CAN 接口名和
bitrate 按现场接线配置；本项目使用 `piper-lt`、`piper-lo`、`piper-rt`、
`piper-ro`。具体采集前检查见 `scripts/PiperRecorder/README.md`。

### 5. 单独复制数据集和模型

Git clone 后 `datasets/` 和 `models/` 为空或不存在。把原机器对应目录用 rsync
或其他大文件存储单独复制：

```bash
rsync -a --info=progress2 OLD_HOST:/home/py/main/easyEmbodiedDeploy/datasets/ datasets/
rsync -a --info=progress2 OLD_HOST:/home/py/main/easyEmbodiedDeploy/models/ models/
```

### 6. 验证与运行入口

环境检查：

```bash
bash thirdParty/setup_envs.sh --check
```

采集器只做连接前检查（会访问相机和四路 CAN）：

```bash
thirdParty/.venvs/piper-home/bin/python scripts/PiperRecorder/record.py --check
```

启动采集：

```bash
thirdParty/.venvs/piper-home/bin/python scripts/PiperRecorder/record.py
```

启动 HDF5 Viewer：

```bash
thirdParty/.venvs/piper-home/bin/python scripts/PiperViewer/viewer.py
```

启动数据处理网页：

```bash
thirdParty/.venvs/lerobot-pi05/bin/python scripts/LerobotDataProcessor/DataProcessor.py
```

启动训练网页：

```bash
scripts/LerobotTrainer/setup.sh
```

真机部署分为硬件桥和 GPU 推理两个进程，完整命令与安全流程见
`scripts/LerobotDeployer/README.md`。

## 常见问题

**`torch.cuda.is_available()` 为 false**：先看 `nvidia-smi`。如果该命令失败，
先修复 NVIDIA 驱动；如果低于对应 GPU 的上表版本，升级驱动后重启。系统是否
安装 `/usr/local/cuda` 不决定当前 wheel 能否运行。

**提示 `no kernel image is available`**：运行
`bash thirdParty/setup_envs.sh --check` 查看 compiled architectures。当前锁定 wheel
应包含 A800 的 `sm_80` 和 RTX PRO 6000 的 `sm_120`。

**相机无权限或找不到设备**：重新运行 Orbbec `setup_env.py`，重新插拔 USB，
并确认相机没有被其他进程占用。

**pyAgxArm 无法连接**：确认 CAN 接口已 UP、接口名与 Recorder 头部配置一致，
再运行 `record.py --check`。pyAgxArm 和相机环境本身不要求 GPU。
