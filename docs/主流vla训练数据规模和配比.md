# 主流 VLA 训练阶段、数据规模与算力简表

> 统计截至 2026-09。数据“小时占比”和训练时的“采样占比”不是一回事；下文只在论文明确给出时才写训练采样比例。

## 1. π 系列

| 模型 | 机器人训练阶段 | 每阶段训练内容 | 数据规模与配比 |
|---|---|---|---|
| **π0（3.3B）** | ① 通用预训练；② 目标任务微调 | ① 多机器人连续 action chunk，flow matching；② 用目标任务高质量示教继续训练 | 预训练超过 **10,000 h**；采样约 **90.9% 自采 + 9.1% 开源**。目标任务通常约 1～10 h，复杂任务可超过 100 h；微调混合比例未公开 |
| **π0.5** | ① FAST 预训练 280k steps；② 连续动作后训练 80k steps | ① FAST 离散动作 token、语言、高层子任务、检测等 next-token 训练；② 加入 flow-matching action expert，同时训练 token 与连续 action chunk | 阶段一目标移动机器人 MM 仅占 **2.4%**，其余 **97.6%** 为 ME、CE、HL、WD；阶段二使用 MM、ME、HL、WD、VI 并移除 CE。VI 占移动机器人高层样本约 **11%**；其余细分比例未公开 |
| **π0.6 / π\*0.6** | ① π0.6 基模预训练；② π\*0.6 离线 advantage 训练；③ 目标任务 SFT 与 RECAP 循环 | π0.6 用 KI 同时预测 FAST token、连续动作和高层子任务；π\*0.6 先训 value model，再做 advantage-conditioned policy；部署后反复收集自主 rollout/人工纠正并更新 value 与 policy | 预训练为**数万小时**多机器人示教，加网页视觉语言数据；精确小时数、数据配比和训练算力未公开 |
| **π0.7（约 5B）** | 大规模通用训练；不依赖逐任务后训练即可执行很多任务 | 在 π0.6-MEM 上加入历史视觉、详细语言、质量/速度/错误元数据和视觉 subgoal；同时吸收示教、失败、自主 rollout、人工干预、人类视频、开源机器人数据和网页数据 | 总小时数与数据源采样比例未公开。每个 batch 中 **25%** 样本带视觉 subgoal；这是条件输入比例，不是数据源比例 |

π0 与 π0.5 都继承互联网图文预训练的 VLM。该 VLM 预训练属于前置初始化，不计入上述机器人训练阶段。

## 2. 其他主流模型

| 模型 | 参数量 | 机器人/具身数据 | 公开配比 | 公开训练算力 |
|---|---:|---|---|---|
| **RT-1** | 35M | 约 130k episodes、700+任务，13 台机器人采集 17 个月 | 主要是自采机器人示教；小时数和细分采样比例未公开 | 未公开 |
| **RT-2** | 5B / 12B / 55B | RT-1 机器人数据 + 原始 VLM 网页数据 | PaLI-X：**机器人 50% / 网页 50%**；PaLM-E：**机器人 66% / 网页 34%** | 55B 训练 80k steps；5B 训练 270k steps；硬件规模未公开 |
| **Octo** | 93M | Open X-Embodiment 约 800k trajectories | 按任务、场景多样性重加权；完整权重见论文，未给小时数 | TPU v4-128、batch 2048、300k steps，约 **14 h** |
| **OpenVLA** | 7B | Open X-Embodiment 约 970k trajectories | Bridge 13.3%、Fractal 12.7%、Kuka 12.7%、DROID 初始 10% 等；DROID 在最后 1/3 训练中移除 | **64×A100，14 天，约 21,500 A100-GPU·h** |
| **SmolVLA** | 450M | 481 个社区数据集、22.9k episodes、10.6M frames | 数据集细分采样比例未公开 | 单次预训练使用 4 GPU；论文报告整个项目约 **30k GPU·h**，不是一次最终训练的成本 |
| **GR00T N1** | 2.2B | 总计 **8,375.7 h** | 按原始时长：机器人 3,288.8 h（39.3%）、人类视频 2,517 h（30.1%）、仿真 1,742.6 h（20.8%）、神经生成 827.3 h（9.9%）；后训练 real:neural = 1:1 | 约 **50,000 H100-GPU·h**，单次最多使用 1,024 GPU |
| **GR00T N1.5** | 约 3B | N1 数据、OpenXE、内部 GR-1、仿真、DreamGen、AgiBot-Beta | 官方仅公布配比图，未给可核验的文字百分比 | **1,000×H100、250k steps**；墙钟时间未公开 |
| **GR00T N1.7** | 3B | 约 **32k h** 真实示教与人类第一视角视频 + **8k h** 仿真数据 | 按公开小时约 80% : 20%；实际训练采样比例未公开 | 未公开 |
| **Qwen-RobotManip** | 未明确披露总参数 | **38,100 h**：机器人 11,420 h、人类视频 1,933 h、Human-to-Robot 合成 24,808 h | 按原始时长约 **29.9% / 5.1% / 65.0%**；训练时 manipulation/VLA stream : VL stream = **9:1** | GPU 数量、GPU·h 和训练时间未公开 |

## 3. 自建同体量具身基模需要什么

以下指**继承现有 VLM，只训练 VLA/动作部分和必要的骨干参数**，不包含从零训练 VLM。

| 目标 | 数据建议 | 可参考的集群与时间 |
|---|---|---|
| 100M～500M 开源研究基模 | 2万～100万条多任务轨迹 | 4～16 张高端 GPU，约数天到两周；Octo 的公开上限参考是 TPU v4-128 上 14 h |
| 2B～7B 通用 VLA | 至少数千至数万小时、多本体、多任务数据 | 最低按 **2万～5万 GPU·h** 准备；公开锚点为 OpenVLA 的 21,500 A100-GPU·h 和 GR00T N1 的 50,000 H100-GPU·h |
| π0.6/π0.7/Qwen-RobotManip 级通才模型 | 约 10k～40k h，并包含机器人、仿真/合成、人类视频和 VL 数据 | 官方未披露。工程预算宜按 **128～256×H100、2～6 周/完整训练轮次**估算，并额外预留数据清洗、消融和失败重训成本；这是依据公开同规模模型推算，不是官方数字 |

实际项目成本通常高于一次最终训练：数据清洗、超参数实验、消融和多次失败训练可能再消耗最终训练的数倍算力。对于 Piper 单臂项目，直接微调 π0.5、SmolVLA、GR00T 或 Qwen-RobotManip 基模，通常比从零构建通才基模更合理。

## 4. 主要来源

- [π0](https://www.physicalintelligence.company/download/pi0.pdf)
- [π0.5](https://arxiv.org/html/2504.16054)
- [π\*0.6 / RECAP](https://www.physicalintelligence.company/download/pistar06.pdf)
- [π0.7](https://arxiv.org/html/2604.15483)
- [RT-1](https://arxiv.org/html/2212.06817)
- [RT-2](https://arxiv.org/html/2307.15818)
- [Octo](https://arxiv.org/abs/2405.12213)
- [OpenVLA](https://arxiv.org/html/2406.09246)
- [SmolVLA](https://arxiv.org/html/2506.01844)
- [GR00T N1](https://arxiv.org/html/2503.14734)
- [GR00T N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/)
- [GR00T N1.7](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/)
- [Qwen-RobotManip](https://arxiv.org/html/2606.17846v2)
