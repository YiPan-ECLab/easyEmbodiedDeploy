
# OpenpiTrainer 最小迁移流程

以下流程针对一台全新的 x86_64 Linux GPU 服务器，工作目录统一使用
`/root/autodl-tmp/easyEmbodiedDeploy`。只使用 OpenpiTrainer/OpenPI 时，不需要初始化顶层
`thirdParty/pyAgxArm` submodule。

## 1. 获取代码

```bash
cd /root/autodl-tmp
git clone https://github.com/YiPan-ECLab/easyEmbodiedDeploy.git easyEmbodiedDeploy
cd easyEmbodiedDeploy
```

如果代码已经在服务器上，直接进入仓库即可。

## 2. 配置下载源

这三类下载分别处理：

- Python 包：清华 PyPI 镜像，供 `uv sync` 使用。
- uv 管理的 CPython：阿里云的 python-build-standalone 镜像。
- GitHub、GCS checkpoint：使用本地代理的反向 SSH 隧道。GCS 是 Google Cloud Storage，不能用 PyPI 镜像替代。

本机另开一个终端，保持隧道进程运行（本机 Clash Verge 当前 HTTP 代理端口为 `7890`）：

```bash
ssh -p 53589 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 17890:127.0.0.1:7890 \
  root@connect.westb.seetacloud.com -N
```

服务器终端执行：

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_PYTHON_INSTALL_MIRROR=https://mirrors.aliyun.com/github/releases/astral-sh/python-build-standalone
export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=http://127.0.0.1:17890
export ALL_PROXY=http://127.0.0.1:17890
export UV_HTTP_TIMEOUT=120
```

如果服务器有可用的直连网络，可以只保留两个镜像变量；GitHub 克隆和 GCS checkpoint 仍建议走代理。

## 3. 创建 OpenPI/JAX 环境

```bash
cd /root/autodl-tmp/easyEmbodiedDeploy
python3 -m pip install --upgrade uv \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 30 --retries 3
GIT_LFS_SKIP_SMUDGE=1 bash thirdParty/setup_openpi_env.sh
bash thirdParty/setup_openpi_env.sh --check
```

脚本会创建隔离环境 `thirdParty/.venvs/openpi-jax`，安装锁定的 OpenPI/JAX 依赖，并自动克隆它需要的固定版本 `thirdParty/lerobot-openpi`。不要把这些依赖混装进其他 LeRobot 环境。

## 4. 下载官方 pi05 JAX 基础 checkpoint

```bash
cd /root/autodl-tmp/easyEmbodiedDeploy
export OPENPI_DATA_HOME=/root/autodl-tmp/easyEmbodiedDeploy/models/base_models
mkdir -p "$OPENPI_DATA_HOME"
thirdParty/.venvs/openpi-jax/bin/python - <<'PY'
from openpi.shared import download

print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base", token="anon"))
PY
```

该 checkpoint 大约 11.6 GiB，目录应为：

```text
/root/autodl-tmp/easyEmbodiedDeploy/models/base_models/openpi-assets/checkpoints/pi05_base
```

下载 GCS 时只需要上面的代理变量；PyPI 镜像不影响 GCS 传输。下载中断后重新执行同一命令即可续传/复用已有文件。

## 5. 训练 OpenpiTrainer

最小的 trained RTC 训练命令如下，数据集路径替换为自己的 consolidated parquet/video 数据集：

```bash
cd /root/autodl-tmp/easyEmbodiedDeploy
thirdParty/.venvs/openpi-jax/bin/python scripts/OpenpiTrainer/trainer.py \
  --dataset-root datasets/lerobot/<DATASET_NAME> \
  --pretrained models/base_models/openpi-assets/checkpoints/pi05_base \
  --output-dir models/finetuned_models/openpi_pi05_piper \
  --job-name openpi_pi05_piper \
  --action-mode delta_joint \
  --normalization-mode quantiles \
  --finetune-mode lora --training-scope expert_only \
  --rtc-mode trained --rtc-max-delay 5 \
  --batch-size 1 --steps 30000 --device cuda --gpu-ids 0
```

普通 checkpoint 训练使用 `--rtc-mode normal --rtc-max-delay 0`。想使用 trained RTC 推理时，必须加载带 `rtc_max_delay > 0` 的训练产物；官方 `pi05_base` 本身不是 trained RTC checkpoint。

训练网页入口：

```bash
scripts/OpenpiTrainer/setup.sh
```

## 6. trained RTC 部署参数

```bash
thirdParty/.venvs/openpi-jax/bin/python scripts/OpenpiDeployer/OpenpiDeployer.py \
  --checkpoint models/finetuned_models/openpi_pi05_piper/<STEP> \
  --inference-mode rtc --rtc-mode trained \
  --rtc-horizon 10 --rtc-queue-threshold 30
```

若训练时 `rtc_max_delay=D`，部署必须满足：

```text
D <= rtc_horizon <= 50-D
D <= rtc_queue_threshold < 50
```

`guided` 与 `trained` 是互斥采样路径：guided 使用普通 flow-matching checkpoint 和推理期 guidance；trained 使用专门训练的 checkpoint 和 train-time hard-prefix 采样，不能在一次采样中同时启用。

## 7. 最小验收

```bash
cd /root/autodl-tmp/easyEmbodiedDeploy
bash thirdParty/setup_openpi_env.sh --check
thirdParty/.venvs/openpi-jax/bin/python -m unittest discover \
  -s scripts/OpenpiTrainer/tests -v
```

### trained RTC 数据链路索引

- 训练参数解析、`rtc_mode`/`rtc_max_delay`：`scripts/OpenpiTrainer/trainer.py:224-246`
- Pi0.5 模型配置写入 train-time delay：`scripts/OpenpiTrainer/openpi_config.py:193-207`
- 训练期随机干净前缀、只计算后缀 loss：`thirdParty/openpi/src/openpi/models/pi0.py:203-243`
- 部署器设置 `OPENPI_RTC_SAMPLE_MODE`、创建 RTC engine：`scripts/OpenpiDeployer/OpenpiDeployer.py:237-301`
- checkpoint manifest 约束与 trained RTC 参数校验：`scripts/OpenpiDeployer/runtime.py:81-117`
- 异步队列快照、推理延迟、丢弃/替换 chunk：`scripts/OpenpiDeployer/runtime.py:364-475`
- policy 将前一 chunk 按训练输入变换后传给 JAX：`thirdParty/openpi/src/openpi/policies/policy.py:89-157`
- `train_time` 分支的 hard-prefix 采样：`thirdParty/openpi/src/openpi/models/pi0_realtime.py:234-262`
