# OpenPI JAX Deployer

该模块加载 `OpenpiTrainer` 生成的 Orbax/JAX checkpoint，并复用当前本地 `LerobotDeployer` 的 Piper 硬件桥、14 维动作转换、硬件限位、低通控制、看门狗和停止回零实现。模型进程运行在 `thirdParty/.venvs/openpi-jax`，硬件进程仍运行在 `thirdParty/.venvs/piper-home`，互不污染。

```bash
scripts/OpenpiDeployer/setup.sh --port 8082
```

Web 页面支持同步推理、guided RTC 和 trained RTC，提供 RTC horizon、队列阈值、权重曲线、最大引导权重、flow 采样步数、夹爪引导掩码、JIT warmup、GPU、控制频率、低通、dry-run 与诊断文件。启动完整流程或模型端按钮是明确的动作触发；停止按钮会停止模型端，再让现有硬件桥把两个 follower 及夹爪安全回 ZERO。

真机命令示例（会移动机械臂）：

```bash
# 终端 1：显式启动现有硬件桥，执行 follower 回零和预置
thirdParty/.venvs/piper-home/bin/python scripts/LerobotDeployer/hardware_bridge.py

# 终端 2：guided RTC；命令结束或 Ctrl-C 后调用桥接停止和回零
thirdParty/.venvs/openpi-jax/bin/python scripts/OpenpiDeployer/OpenpiDeployer.py \
  --checkpoint models/finetuned_models/openpi_pi05_piper \
  --action-mode delta_joint --inference-mode rtc --rtc-mode guided \
  --device cuda:1 --max-steps 30
```

不连接硬件的完整离线闭环：

```bash
thirdParty/.venvs/openpi-jax/bin/python scripts/OpenpiDeployer/OpenpiDeployer.py \
  --checkpoint models/finetuned_models/openpi_pi05_piper \
  --inference-mode rtc --rtc-mode trained --device cuda:1 --max-steps 5 \
  --offline-dataset-root datasets/lerobot/Remove_the_shipping_label_and_place_it_on_the_package_on_the_table_10hz_delta \
  --dry-run
```

Deployer 会先在未连接硬件时编译普通采样和所选 RTC 采样图，并在第一控制 tick 前完成 RTC 队列预填充。异步 RTC 在推理开始时读取队列中尚未执行的动作作为 prefix；推理返回后按实测耗时丢弃新 chunk 已经过期的帧，并替换整个剩余队列，避免把新旧 chunk 错误追加造成回抽。trained RTC 只接受清单中 `rtc_max_delay > 0` 的 checkpoint；不可用的新 chunk 会保留旧队列并重试，连续 5 次超出训练支持延迟时停止部署并明确报错。

```bash
thirdParty/.venvs/openpi-jax/bin/python -m unittest discover -s scripts/OpenpiDeployer/tests -v
```
