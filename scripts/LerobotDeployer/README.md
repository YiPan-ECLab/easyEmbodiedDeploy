# LerobotDeployer

PI05 和 Piper/Orbbec 的依赖分别安装在两个虚拟环境中，因此部署使用两个进程：

- `hardware_bridge.py` 在 `piper-home` 环境连接真机。
- `LerobotDeployer.py` 在 `lerobot-pi05` 环境加载 checkpoint 并推理。
- ZeroMQ 协议的每条请求都独立应答；任一推理进程异常退出后不会再发生请求阶段错位。

## 1. 启动硬件桥

```bash
cd /home/py/main/easyEmbodiedDeploy
PYTHONPATH=scripts/PiperRecorder \
thirdParty/.venvs/piper-home/bin/python \
scripts/LerobotDeployer/hardware_bridge.py \
--endpoint tcp://127.0.0.1:6007
```

硬件桥连接相机和两台 follower，不连接两台 leader。follower 关节回 ZERO、夹爪闭合到 `0 m` 后，分别将左右 J6 移到 -45°/+45°。停止时两个 follower 的关节和夹爪再次回 ZERO。

`--control-hz` 是桥启动后的初始 `move_js` 发送频率，默认 30 Hz。推理进程连接后会用自己的 `--hz` 更新该值。`--smoothing-time-constant` 是绝对目标低通滤波时间常数，默认 `0.1 s`；设为 0 才会关闭滤波。

## 2. 首次进行无动作推理检查

此命令读取真机观测并生成一个 action，但不向机械臂发送 action：

```bash
cd /home/py/main/easyEmbodiedDeploy
PYTHONPATH=scripts/LerobotDeployer \
thirdParty/.venvs/lerobot-pi05/bin/python \
scripts/LerobotDeployer/LerobotDeployer.py \
--checkpoint models/finetuned_models/pi05_run_lora_test/checkpoints/005000/pretrained_model \
--dataset-root datasets/lerobot/Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta/lerobot \
--action-mode delta_joint \
--inference-mode sync \
--bridge-endpoint tcp://127.0.0.1:6007 \
--task "Remove the shipping label and place it on the package on the table" \
--device cuda \
--hz 30 \
--max-steps 1 \
--dry-run
```

该命令结束时会通知硬件桥停止，并让 follower 回 ZERO。下一次部署前需要重新启动硬件桥。

## 3. 真机同步推理

移除 `--dry-run`，并先用有限步数测试：

```bash
PYTHONPATH=scripts/LerobotDeployer \
thirdParty/.venvs/lerobot-pi05/bin/python \
scripts/LerobotDeployer/LerobotDeployer.py \
--checkpoint models/finetuned_models/pi05_run_lora_test/checkpoints/005000/pretrained_model \
--dataset-root datasets/lerobot/Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta/lerobot \
--action-mode delta_joint \
--inference-mode sync \
--bridge-endpoint tcp://127.0.0.1:6007 \
--task "Remove the shipping label and place it on the package on the table" \
--device cuda \
--hz 30 \
--max-steps 10
```

确认有限步数正常后，可用 `--max-steps 0` 持续运行。

## 4. RTC 推理

当前 checkpoint 没有训练 RTC prefix，必须使用 `guided`：

```bash
PYTHONPATH=scripts/LerobotDeployer \
thirdParty/.venvs/lerobot-pi05/bin/python \
scripts/LerobotDeployer/LerobotDeployer.py \
--checkpoint models/finetuned_models/pi05_run_lora_test/checkpoints/005000/pretrained_model \
--dataset-root datasets/lerobot/Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta/lerobot \
--action-mode delta_joint \
--inference-mode rtc \
--rtc-mode guided \
--rtc-horizon 20 \
--rtc-queue-threshold 40 \
--bridge-endpoint tcp://127.0.0.1:6007 \
--task "Remove the shipping label and place it on the package on the table" \
--device cuda \
--hz 30 \
--smoothing-time-constant 0.1 \
--max-steps 10
```

RTC 直接使用当前 LeRobot 源码中的 `RTCInferenceEngine`、`ActionQueue` 和 PI05 RTC processor。`--max-steps` 只统计真正取得并处理的 action，不统计 RTC 尚未产出 action 时的轮询。

## 两种频率

`--hz` 默认 30 Hz，表示每个 follower 接收 `pyAgxArm.move_js` 目标的频率。桥在独立线程中维持该频率，因此同步模型推理阻塞时也不会改变底层发送周期。

`--policy-hz` 表示从策略 action chunk 消费新 action 的频率，默认读取数据集 `fps`，本数据集为 10 Hz。增量 action 的一步具有训练数据的时间尺度，所以通常不应改变 `--policy-hz`；底层频率可独立设为 20、30 或其他值。

`move_js` 不提供底层平滑或轨迹规划。桥接进程先把模型输出转换并裁剪为绝对目标，再以一阶低通滤波生成每个控制 tick 的 `move_js` 目标。这样不会错误地重复累计或插值 delta action。桥直接从 pyAgxArm 的 `ROBOT_JOINT_LIMIT_PRESET_RAD["piper"]` 读取每个关节的上下限；左右夹爪均严格裁剪到 `[0, 0.1] m`。

六个关节的范围依次为：J1 `[-2.617994, 2.617994]`、J2 `[0, 3.141593]`、J3 `[-2.967060, 0]`、J4 `[-1.745330, 1.745330]`、J5 `[-1.221730, 1.221730]`、J6 `[-2.094396, 2.094396]` rad。

## 主要参数

- `--action-mode delta_joint|absolute_joint`：必须与数据集的动作表示一致。增量动作以最后一次下发目标为累计基准。
- `--hz`：每个 follower 接收 `move_js` 目标的频率，范围 `(0, 200]`，默认 30。
- `--smoothing-time-constant`：绝对目标低通时间常数，范围 `[0, 2] s`，默认 `0.1 s`。
- `--policy-hz`：策略 action 消费频率；默认数据集 fps。
- `--rtc-horizon`：guided/trained RTC 的衔接时域，默认 20。
- `--rtc-queue-threshold`：剩余动作达到该值时开始异步补队，默认 40。
- `--diagnostics-jsonl`：可选逐 tick 诊断文件；退出时终端也会打印延迟、空队列和真实 `move_js` 频率摘要。
- `--watchdog-timeout`：推理进程失联后硬件桥停止并回 ZERO 的时间，默认 30 秒。
- `--max-steps`：真正处理的策略 action 数；0 表示持续运行。
- `--seed`：PI05 采样随机种子，默认 42，用于复现同一部署诊断。
- `--dry-run`：执行完整观测与模型推理，但不发送 action。

按 Ctrl-C 后，推理端停止 inference engine，通知硬件桥停止控制，并让两台 follower 回 ZERO。
