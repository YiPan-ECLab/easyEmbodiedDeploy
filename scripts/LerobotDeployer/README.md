# LerobotDeployer

PI05 和 Piper/Orbbec 的依赖位于两个虚拟环境，因此部署分成两个进程：

- `hardware_bridge.py` 在 `piper-home` 环境连接三路相机和两个 follower。
- `LerobotDeployer.py` 在 `lerobot-pi05` 环境加载全量或 LoRA checkpoint 并执行推理。
- 两个进程通过带请求编号和协议版本的 ZeroMQ RPC 通信。

策略控制阶段只使用 `pyAgxArm.move_js`。硬件桥不连接两个 leader；它在启动和退出时将两个 follower 关节及夹爪回 ZERO，并在启动时保留左 J6 `-45°`、右 J6 `+45°` 的内旋预置。

## Web 控制台

使用 setup 脚本启动本地控制台：

```bash
cd /home/py/main/easyEmbodiedDeploy/scripts/LerobotDeployer
./setup.sh
```

浏览器打开 <http://127.0.0.1:8080>。需要从局域网其他设备访问时可以指定监听地址和端口：

```bash
./setup.sh --host 0.0.0.0 --port 8080
```

页面提供以下功能：

- 自动发现仓库中的 checkpoint，也允许直接填写路径。
- 完整配置 RTC、action mode、单卡 GPU、Policy Hz、Control Hz、滤波、最大步数、随机种子和超时。
- 一键依次启动硬件桥和模型端，也可以分别启动两个进程。
- 显示最终命令，配置与终端部署完全一致。
- 实时展示加载阶段、action 进度、RTC queue、控制循环时延、GPU 状态和终端日志。
- 顶部常驻“停止并回零”按钮以及流程卡片中的“安全停止”按钮均可代替 Ctrl-C：先停止 inference engine，再通知桥停止 `move_js` 并将两个 follower 关节和夹爪回 ZERO。

Web 控制台只停止由自身启动的子进程。如果检测到终端或其他用户启动的桥接/部署进程，会显示为外部进程并拒绝接管，避免误停正在运行的真机任务。关闭 setup 服务时，也会安全停止由该页面启动的部署流程。

## 已通过真机测试的默认配置

| 配置 | 默认值 |
| --- | --- |
| checkpoint | `pi05_run_lora_test/checkpoints/005000`（旧 LoRA checkpoint） |
| 动作表示 | `delta_joint` |
| 推理 | `rtc`、`guided` |
| RTC 衔接时域 | `20` 个 action |
| RTC 补队阈值 | 队列剩余 `40` 个 action 时启动异步推理 |
| 策略 action 消费频率 | `10 Hz`，可通过 `--policy-hz` 指定 |
| follower `move_js` 发送频率 | `30 Hz` |
| 绝对目标低通时间常数 | `0.1 s` |
| 随机种子 | `20260909` |
| Web 默认运行长度 | `0`（持续运行；填写正数可限制策略 action 数） |

以上是 Web 控制台的默认值；命令行仍可通过参数覆盖。`--policy-hz` 默认 `10 Hz`，不会读取或跟随数据集 fps，可以显式覆盖。

## 1. 启动硬件桥

```bash
cd /home/py/main/easyEmbodiedDeploy
PYTHONPATH=scripts/PiperRecorder \
thirdParty/.venvs/piper-home/bin/python \
scripts/LerobotDeployer/hardware_bridge.py \
--endpoint tcp://127.0.0.1:6007
```

硬件桥只负责设备和 RPC。控制频率与滤波参数由 Deployer 在开始发送动作前统一配置，桥接端不再保留重复参数。

| 硬件桥参数 | 默认值 | 作用与约束 |
| --- | --- | --- |
| `--endpoint` | `tcp://127.0.0.1:6007` | ZeroMQ 监听地址 |

## 2. 使用实测默认配置部署

下面是上一次丝滑完成的 RTC 真机测试配置，其中策略动作消费频率和底层控制频率都显式写出：

```bash
cd /home/py/main/easyEmbodiedDeploy
PYTHONPATH=scripts/LerobotDeployer \
thirdParty/.venvs/lerobot-pi05/bin/python \
scripts/LerobotDeployer/LerobotDeployer.py \
--checkpoint /home/py/main/easyEmbodiedDeploy/models/finetuned_models/pi05_run_lora_test/checkpoints/005000/pretrained_model \
--action-mode delta_joint \
--inference-mode rtc \
--rtc-mode guided \
--rtc-horizon 20 \
--rtc-queue-threshold 40 \
--bridge-endpoint tcp://127.0.0.1:6007 \
--task "Remove the shipping label and place it on the package on the table" \
--device cuda:0 \
--policy-hz 10 \
--control-hz 30 \
--smoothing-time-constant 0.1 \
--max-steps 0 \
--seed 20260909 \
--rpc-timeout 5 \
--watchdog-timeout 30 \
--diagnostics-jsonl /tmp/piper_rtc_real_tau01_30steps.jsonl
```

### RTC 与平滑关键参数

以下参数已经在上面的示例命令中显式暴露，修改时无需改源码：

| 参数 | 本次实测值 | 具体含义 |
| --- | --- | --- |
| `--rtc-mode guided` | `guided` | 当前旧 checkpoint 没有经过 RTC prefix 训练，因此使用 guided RTC；只有 `rtc_training_max_delay > 0` 的 checkpoint 才能改为 `trained` |
| `--rtc-horizon 20` | `20` | 新旧 action chunk 的衔接时域；新 chunk 会参考队列中未来 20 个 action 来减少 chunk 边界突变 |
| `--rtc-queue-threshold 40` | `40` | 当队列剩余 action 数下降到 40 时立即在后台请求下一批 action；必须小于模型的 `chunk_size` |
| `--policy-hz 10` | `10 Hz` | 每秒从 action queue 取出 10 个新模型动作；提高该值会加快 chunk 消费和动作轨迹执行，与数据集 fps 无绑定关系 |
| `--control-hz 30` | `30 Hz` | 每秒向左右 follower 各调用 30 次 `move_js`，负责在相邻策略动作间发送滤波后的控制目标 |
| `--smoothing-time-constant 0.1` | `0.1 s` | 对转换后的绝对关节与夹爪目标做一阶低通；在 30 Hz 下每个控制周期向最新目标移动约 28.35%，设为 0 会完全关闭滤波 |
| `--seed 20260909` | `20260909` | 固定 PI05 随机采样，方便复现实机诊断结果 |
| `--diagnostics-jsonl PATH` | `/tmp/piper_rtc_real_tau01_30steps.jsonl` | 保存逐 tick 的队列长度、时延、模型 action、绝对目标和限位状态 |

本次实测中单个推理 chunk 的中位延迟约为 `0.479 s`，但队列最低仍有 36 个 action，运行期间没有发生队列饿死。`--rtc-queue-threshold 40` 的作用就是在队列耗尽前提前补充，而 `--smoothing-time-constant 0.1` 负责消除送入 `move_js` 的目标跳变。

由于上述行为参数已经固化为默认值，同一个任务也可以简写为：

```bash
PYTHONPATH=scripts/LerobotDeployer \
thirdParty/.venvs/lerobot-pi05/bin/python \
scripts/LerobotDeployer/LerobotDeployer.py
```

Web 控制台默认使用 `--max-steps 0` 持续运行。需要有限长度测试时，在页面的“最大策略步数”中填写正整数；运行结束后会停止并回 ZERO。

## 3. 全部推理端参数

| 参数 | 默认值 | 作用与约束 |
| --- | --- | --- |
| `--checkpoint` | `models/finetuned_models/pi05_run_lora_test/checkpoints/005000/pretrained_model` | 全量或 LoRA 的 `pretrained_model` 目录 |
| `--action-mode` | `delta_joint` | `delta_joint` 或 `absolute_joint`，必须与 checkpoint 的训练动作语义一致 |
| `--inference-mode` | `rtc` | `rtc` 或 `sync` |
| `--rtc-mode` | `guided` | `guided` 或 `trained`；仅 RTC 使用 |
| `--rtc-horizon` | `20` | 相邻 RTC chunk 的衔接时域，必须为正整数 |
| `--rtc-queue-threshold` | `40` | 队列剩余 action 数达到该值时触发异步补队；必须小于 checkpoint 的 `chunk_size` |
| `--bridge-endpoint` | `tcp://127.0.0.1:6007` | 硬件桥地址 |
| `--task` | 当前撕标签英文任务 | 传给策略的任务指令 |
| `--device` | `cuda:0` | 单卡推理设备；支持 `cuda:0`、`cuda:1`、`cuda:2`、`cuda` 或 `cpu`，Web 页面按实际 GPU 动态列出 |
| `--policy-hz` | `10` | 从 action queue 消费新模型动作的频率，范围 `(0, 200] Hz`；不读取数据集 fps |
| `--control-hz` | `30` | 每个 follower 接收 `move_js` 目标的频率，范围 `(0, 200] Hz` |
| `--smoothing-time-constant` | `0.1` | 绝对目标低通时间常数，范围 `[0, 2] s`；`0` 关闭滤波 |
| `--max-steps` | `0`（Web） | 真正取得并处理的策略 action 数；`0` 持续运行，RTC 预填充等待不计数 |
| `--seed` | `20260909` | PI05 采样随机种子 |
| `--rpc-timeout` | `5` | 单次桥 RPC 超时，单位秒 |
| `--watchdog-timeout` | `30` | 推理端失联后桥停止控制并回 ZERO 的时间，最小 `5 s` |
| `--diagnostics-jsonl` | 不保存 | 可选的逐控制 tick JSONL 诊断文件路径 |

运行以下命令可查看与当前代码一致的完整选项和默认值：

```bash
PYTHONPATH=scripts/LerobotDeployer \
thirdParty/.venvs/lerobot-pi05/bin/python \
scripts/LerobotDeployer/LerobotDeployer.py --help
```

## 4. delta 和 absolute 模型的滤波

两种动作表示共用同一个硬件滤波器，但进入滤波器前的转换不同：

- `delta_joint`：模型 action 以最后一次已下发的绝对目标为基准逐步累加，然后执行硬件限位。
- `absolute_joint`：模型 action 直接作为绝对目标，然后执行相同的硬件限位。
- 两条路径最终都把 14 维绝对目标送入硬件桥；桥以相同的一阶低通滤波生成每个 `move_js` 控制周期的目标。

因此低通不会错误地缩小或重复累计 delta，也不依赖模型训练频率。当前模型使用 `delta_joint`；后续绝对关节模型添加：

```bash
--action-mode absolute_joint
```

旧 checkpoint 没有记录 delta/absolute 语义，因此部署时必须显式选择正确的 `--action-mode`。混合多个数据集训练时，各数据集必须先统一为同一种输出动作语义。

部署不读取训练数据集。输入输出 feature 由 checkpoint `config.json` 与 Piper 固定硬件契约交叉校验；归一化和反归一化参数直接从 checkpoint 内的 processor 配置及 safetensors 权重加载。因此多数据集训练得到的统计量以 checkpoint 保存结果为准。

## 5. 同步推理和 trained RTC

页面默认启动持续运行。需要限制动作数量时直接设置“最大策略步数”，`0` 表示持续运行；运行结束后会停止推理、停止硬件桥并让 follower 回 ZERO。

需要同步推理时添加 `--inference-mode sync`。同步模式忽略 RTC 执行配置。需要使用 trained RTC 时添加 `--rtc-mode trained`，且 checkpoint 必须使用大于 0 的 `rtc_training_max_delay` 进行过 RTC 训练；当前旧 checkpoint 只能使用 `guided`。

## 控制频率与完整动作链路

`--policy-hz` 决定从 action chunk 取出新模型动作的频率，默认 `10 Hz`，且部署时不读取数据集 fps。可以提高该值，但 delta action 的每步位移不会随频率缩小，因此从 10 Hz 提高到 20 Hz 通常会让同一段 chunk 以约两倍速度执行，并改变 RTC 对推理延迟的步数换算。

`--control-hz` 决定硬件桥向每个 follower 调用 `move_js` 的频率。它由独立控制线程维持，可以设为 20、30 Hz 或其他合法值。本次真机验证的组合为 `--policy-hz 10 --control-hz 30`：每 100 ms 消费一个新模型动作，其间每 33.3 ms 发送一个滤波后的 `move_js` 目标。

完整数据链路为：相机/关节反馈 → LeRobot processor → PI05 RTC action queue → checkpoint postprocessor 反归一化 → delta 累加或 absolute 直通 → 关节/夹爪限位 → 绝对目标一阶低通 → 独立 30 Hz `move_js` 控制线程 → 两个 follower。

`move_js` 自身不提供轨迹规划或平滑。时间常数 `0.1 s`、控制频率 `30 Hz` 时，每个控制周期向最新目标移动约 `28.35%`。

桥直接读取 pyAgxArm 的 Piper 关节限位，单位为 rad：J1 `[-2.617994, 2.617994]`、J2 `[0, 3.141593]`、J3 `[-2.967060, 0]`、J4 `[-1.745330, 1.745330]`、J5 `[-1.221730, 1.221730]`、J6 `[-2.094396, 2.094396]`。左右夹爪目标均限制在 `[0, 0.1] m`。

按 Ctrl-C 后，推理端停止 inference engine，读取并打印 RTC 和真实 `move_js` 诊断，通知硬件桥停止控制，然后让两个 follower 的关节和夹爪回 ZERO。
