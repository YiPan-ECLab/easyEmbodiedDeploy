# LerobotTrainer

## Web UI

```bash
thirdParty/.venvs/lerobot-pi05/bin/python scripts/LerobotTrainer/trainer_app.py
```

浏览器访问 `http://127.0.0.1:8890`。默认 W&B 为启用且 online；使用前先执行 `thirdParty/.venvs/lerobot-pi05/bin/wandb login`。

## 直接脚本训练

```bash
thirdParty/.venvs/lerobot-pi05/bin/python scripts/LerobotTrainer/LerobotTrainer.py \
  --dataset-root datasets/lerobot/Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta \
  --pretrained models/base_models/pi05_base \
  --output-dir models/finetuned_models/pi05_10hz_delta \
  --normalization-mode quantiles --finetune-mode full --rtc-mode normal \
  --batch-size 1 --steps 30000 --save-freq 5000 --gpu-ids 0,1 \
  --wandb-project piper-pi05
```

`--normalization-mode` 支持 `quantiles`、`min_max`、`mean_std`；`--finetune-mode` 支持 `full`、`lora`；`--rtc-mode` 支持 `normal`、`trained`。LoRA 可设置 `--lora-rank 16`，trained RTC 可设置 `--rtc-max-delay 5`。多卡使用 `--gpu-ids 0,2` 这类列表，Trainer 会设置 `CUDA_VISIBLE_DEVICES` 并用 Accelerate 启动。W&B 可用 `--no-wandb` 关闭，或 `--wandb-mode offline` 离线记录。训练成功后 checkpoint 和 `training_adapter.json` 会保存到输出目录。
