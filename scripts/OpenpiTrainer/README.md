# OpenPI JAX Trainer

该模块是当前本地 `LerobotTrainer` 的 JAX/OpenPI 适配层，只接受双臂 Piper 的固定契约：左右各 6 个弧度关节和 1 个米制夹爪，共 14 维；相机固定为 `agentview`、`left_eye_in_hand`、`right_eye_in_hand` 三路 224×224 RGB。

实现保持 OpenPI 原有训练入口和 Orbax checkpoint 格式。`dataset_v3.py` 只读适配当前 LeRobot v3 consolidated parquet/video 数据，避免把主仓库降级到 OpenPI 所需的旧 LeRobot。隔离环境位于 `thirdParty/.venvs/openpi-jax`，旧 LeRobot 固定在提交 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。

```bash
thirdParty/setup_openpi_env.sh
scripts/OpenpiTrainer/setup.sh
```

默认从 8893 起自动选择 Web 端口。页面提供数据深度验证、LoRA/非 LoRA、expert-only/冻结视觉/全模型范围、normal/trained RTC、quantile/min-max/mean-std 归一化、梯度检查点、单卡/JAX FSDP、多种 W&B 模式、保存开关和断点恢复。`trained RTC` 会对动作 chunk 随机采样 `[0, rtc_max_delay]` 的已知干净前缀，只计算并重标定后缀 flow-matching loss。

命令行示例：

```bash
thirdParty/.venvs/openpi-jax/bin/python scripts/OpenpiTrainer/trainer.py \
  --dataset-root datasets/lerobot/Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta \
  --pretrained models/base_models/openpi-assets/checkpoints/pi05_base \
  --output-dir models/finetuned_models/openpi_pi05_piper \
  --job-name openpi_pi05_piper \
  --action-mode delta_joint \
  --normalization-mode quantiles \
  --finetune-mode lora --training-scope expert_only \
  --rtc-mode trained --rtc-max-delay 5 \
  --batch-size 1 --steps 30000 --device cuda --gpu-ids 1
```

恢复训练时传入输出根目录中的清单：

```bash
thirdParty/.venvs/openpi-jax/bin/python scripts/OpenpiTrainer/trainer.py \
  --dataset-root datasets/lerobot/Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta \
  --resume-config models/finetuned_models/openpi_pi05_piper/training_manifest.json \
  --device cuda --gpu-ids 1
```

输出目录包含数字 step 子目录，其中 `params/` 用于推理，`train_state/` 用于恢复，`assets/` 保存与 checkpoint 一致的归一化统计。`training_manifest.json` 记录动作语义、模型拓扑、LoRA 参数和 RTC 能力，Deployer 会强制校验。

验收测试：

```bash
thirdParty/.venvs/openpi-jax/bin/python -m unittest discover -s scripts/OpenpiTrainer/tests -v
```

## 位姿动作与相对关节更新

支持 `absolute_base_pose`、`camera_delta_wrist`、`camera_delta_global`；state 保持左右关节 / 夹爪 14 维，位姿 action 是双臂 xyz / 三维 rotvec / 绝对夹爪 14 维。
详见 [完整位姿链路](../../docs/POSE_ACTION_DATAFLOW.md)、[标定操作与训练顺序](../../docs/CAMERA_CALIBRATION.md) 和 [相对关节问题排查](../../docs/JOINT_ACTION_AUDIT.md)。位姿数据必须保留 `meta/pose*` 附属文件，并通过本项目训练入口读取。camera delta 部署需确认相机安装版本；绝对 base 位姿不需要标定。
