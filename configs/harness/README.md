# harness 工作流配置

2026-11-02 从 OmniTower 的 `/home/yqt/piper/datasets/workflows` 导出的参数快照。

- `model_library.json`：3 个共享模型的 checkpoint 路径、GPU、控制频率和 RTC 参数。
- `point_library.json`：7 个共享关键点的名称、ID、关节/末端目标和夹爪配置；不包含采集来源、时间戳或原始反馈记录。
- `092b6554b9f540c6851af9127fabe8a0.json`：我的工作流，持续 VLA 与条件接管配置。
- `608bac040c744d9b83a04925dff1857f.json`：workflow_test，8 步顺序流程。
- `78b15ec250c147a7ab0ba40118ba837c.json`：workflow1，空流程。

各 JSON 沿用原 ID，因此流程与共享库可以对应解析。模型文件仅保存路径引用，不含权重；checkpoint 为当前服务器上的绝对路径，其他机器需要调整。当前持续流程的起始模型仍为 `base_model`，未因导出而切换到 shipping 模型。条件窗口为 12～25 秒、持续 0.05 秒，条件为 `(J4+J6 >= 27° AND Z >= 300 mm) OR Z >= 500 mm`。

这些文件用于版本审阅与手动迁移，程序不会自动从此目录加载，也不会改写正在使用的配置。迁移时先备份目标 `datasets/workflows`，再按需复制所需配置文件，并核对目标机器的模型路径与硬件参数。`datasets/` 继续由 `.gitignore` 排除。

本目录不包含历史快照、共享库修订/删除状态、锁文件、运行日志、轨迹、视频、数据集、模型权重、缓存或环境文件。
