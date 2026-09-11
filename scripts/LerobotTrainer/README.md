# LeRobot PI05 Train Lab

这是 PI05 训练控制台，使用独立的 wrapper、Web 前端和 LeRobot 进程。未指定端口时，启动脚本会从 `8891` 开始自动寻找空闲端口，并在终端打印最终访问地址。

## 启动

```bash
cd /home/py/main/easyEmbodiedDeploy
scripts/LerobotTrainer/setup.sh
```

也可以显式锁定端口；若端口已被占用，服务会直接报错而不会干扰占用者：

```bash
scripts/LerobotTrainer/setup.sh --port 9000
```

显式要求自动选择端口：

```bash
scripts/LerobotTrainer/setup.sh --port auto
```

自动扫描默认从 `8891` 开始，也可以通过环境变量修改起点：

```bash
LEROBOT_TRAINER_PORT_START=9000 scripts/LerobotTrainer/setup.sh
```

需要从其他机器访问时显式指定监听地址；它仍会自动挑选端口：

```bash
scripts/LerobotTrainer/setup.sh --host 0.0.0.0
```

页面会显示每张物理 GPU 的计算进程。只要所选 GPU 已有计算进程，服务端就拒绝启动新训练，因此不会误占正在训练的 GPU1。页面的停止按钮只向本页面启动的新进程组发送信号。

## 支持的训练方式

- LoRA：使用 LeRobot PI05 自带的 adapter 目标，即动作专家注意力 q/v 与状态、动作投影。可以从 base/full checkpoint 新建 LoRA，也可以继续训练已有 LoRA adapter。
- 直接权重微调：可选择只训练动作专家、冻结视觉编码器，或更新完整模型。已有 LoRA adapter 不能伪装成完整模型用于该模式。
- Normal / trained RTC：normal 的 `rtc_training_max_delay` 为 0；trained 会按页面设置注入最大动作延迟，且必须小于模型的 `chunk_size`。
- 单卡和多卡：单卡直接运行 `lerobot-train`；多卡按当前仓库源码要求使用 `torchrun`，物理编号由 `CUDA_VISIBLE_DEVICES` 精确映射。
- 断点续训：选择 checkpoint 的 `train_config.json`。续训使用 checkpoint 中保存的训练配置，页面只覆盖运行设备，避免表单默认值污染原配置。
- W&B online/offline/关闭、bfloat16/float32、动作步数、保存和日志间隔、DataLoader workers 等均可配置。

当前仓库内的 LeRobot 在 `TrainPipelineConfig` 和 dataset factory 中明确禁用了 `MultiLeRobotDataset`，因此页面选择一个已经转换好的本地数据集。若后续要混合多个数据集，应先产出一个合并且重新计算统计量的数据集，或先升级并验证 LeRobot 的多数据集实现。

## 训练前检查

“深度验证数据”会在独立 LeRobot Python 进程中检查：

- `observation.state` 和 `action` 均为有限的 `float32[14]`，轴顺序严格为左右 Piper 的 J1–J6 与 gripper。
- `agentview`、`left_eye_in_hand`、`right_eye_in_hand` 均可解码为 `[3,224,224]`。
- 所选归一化模式需要的 q01/q99、min/max 或 mean/std 是有限的 14 维向量，且边界顺序有效。
- `processing.json` 的 delta/absolute 动作语义与页面选择一致。

Dry run 同样执行深度验证并打印最终命令，但不加载模型、不运行训练，也不创建输出目录。

## 命令行

```bash
thirdParty/.venvs/lerobot-pi05/bin/python \
  scripts/LerobotTrainer/trainer.py \
  --dataset-root datasets/lerobot/Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta \
  --pretrained models/base_models/pi05_base \
  --output-dir models/finetuned_models/pi05_new_run \
  --job-name pi05_piper_lora \
  --action-mode delta_joint \
  --normalization-mode quantiles \
  --finetune-mode lora \
  --rtc-mode normal \
  --batch-size 8 --steps 30000 --save-freq 5000 --log-freq 200 \
  --device cuda --gpu-ids 0 \
  --wandb-enabled --wandb-mode online
```

只验证数据：

```bash
thirdParty/.venvs/lerobot-pi05/bin/python scripts/LerobotTrainer/trainer.py \
  --dataset-root datasets/lerobot/Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta \
  --pretrained models/base_models/pi05_base \
  --output-dir /tmp/not_created \
  --action-mode delta_joint --device cpu --validate-only
```

训练成功后会在输出目录及各 checkpoint 的 `pretrained_model` 中写入精简的 `training_manifest.json`。模型、processor 和反归一化状态仍由 LeRobot 原生 checkpoint 保存，清单不复制大段统计量。
