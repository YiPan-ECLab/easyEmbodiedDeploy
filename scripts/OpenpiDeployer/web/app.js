"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const app = {
  defaults: null,
  checkpoints: [],
  lastLogId: 0,
  previewTimer: null,
  statusTimer: null,
  logTimer: null,
  online: false,
  commandText: "",
};

const fieldIds = [
  "checkpoint", "task", "action_mode", "device",
  "inference_mode", "rtc_mode", "rtc_horizon", "rtc_queue_threshold",
  "rtc_schedule", "rtc_max_guidance_weight", "rtc_mask_grippers",
  "sample_steps", "n_action_steps", "warmup", "dry_run",
  "policy_hz", "control_hz", "smoothing_time_constant", "bridge_endpoint",
  "max_steps", "seed", "rpc_timeout", "watchdog_timeout",
  "diagnostics_jsonl",
];

const numericFields = new Set([
  "rtc_horizon", "rtc_queue_threshold", "policy_hz", "control_hz",
  "smoothing_time_constant", "max_steps", "seed", "rpc_timeout",
  "watchdog_timeout",
  "rtc_max_guidance_weight", "sample_steps", "n_action_steps",
]);
const booleanFields = new Set(["rtc_mask_grippers", "warmup", "dry_run"]);

function formatNumber(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits).replace(/\.0$/, "") : "—";
}

function processStateLabel(state) {
  return {
    stopped: "未启动",
    starting: "启动中",
    ready: "已就绪",
    loading: "加载模型",
    prefilling: "RTC 预填充",
    running: "执行中",
    stopping: "正在停止",
    stopped: "已停止",
    completed: "已完成",
    failed: "运行失败",
  }[state] || state || "未知";
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload;
  try {
    payload = await response.json();
  } catch (_) {
    payload = { error: `HTTP ${response.status}` };
  }
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function toast(message, type = "success") {
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.textContent = message;
  $("#toastStack").append(item);
  setTimeout(() => item.remove(), 4200);
}

function setConnection(online, text) {
  app.online = online;
  const pill = $("#connectionPill");
  pill.classList.toggle("online", online);
  pill.classList.toggle("error", !online);
  $("#connectionText").textContent = text;
}

function populateOptions() {
  const checkpointList = $("#checkpointOptions");
  checkpointList.replaceChildren();
  for (const item of app.checkpoints) {
    const option = document.createElement("option");
    option.value = item.path;
    option.label = `${item.label} · ${item.policy_type} · chunk ${item.chunk_size ?? "?"} · sync ${item.n_action_steps ?? "?"}`;
    checkpointList.append(option);
  }

}

function populateDevices(gpus, preferredDevice = null) {
  const select = $("#device");
  const selected = preferredDevice || select.value;
  select.replaceChildren();
  for (const gpu of gpus || []) {
    const option = document.createElement("option");
    option.value = `cuda:${gpu.index}`;
    option.textContent = `GPU ${gpu.index} · ${gpu.name}`;
    select.append(option);
  }
  const cpu = document.createElement("option");
  cpu.value = "cpu";
  cpu.textContent = "CPU";
  select.append(cpu);
  if ([...select.options].some((option) => option.value === selected)) {
    select.value = selected;
  }
  return select.value;
}

function applyConfig(config) {
  for (const id of fieldIds) {
    if (config[id] !== undefined && $(`#${id}`)) {
      if (booleanFields.has(id)) $(`#${id}`).checked = Boolean(config[id]);
      else $(`#${id}`).value = config[id];
    }
  }
  updateFormState();
  schedulePreview();
}

function selectedCheckpoint() {
  const selected = $("#checkpoint").value.trim().replace(/\/+$/, "").replace(/\/params$/, "");
  return app.checkpoints.find((item) => item.path === selected)
    || app.checkpoints.find((item) => selected && item.path.startsWith(`${selected}/`));
}

function collectConfig() {
  const config = {};
  for (const id of fieldIds) {
    const input = $(`#${id}`);
    config[id] = booleanFields.has(id) ? input.checked : (numericFields.has(id) ? Number(input.value) : input.value.trim());
  }
  return config;
}

function updateFormState() {
  const rtc = $("#inference_mode").value === "rtc";
  for (const field of $$(".rtc-field")) {
    field.classList.toggle("disabled", !rtc);
    for (const input of field.querySelectorAll("input, select")) input.disabled = !rtc;
  }
  const policy = Number($("#policy_hz").value);
  const control = Number($("#control_hz").value);
  const tau = Number($("#smoothing_time_constant").value);
  $("#flowPolicy").textContent = `${formatNumber(policy)} Hz`;
  $("#flowControl").textContent = `${formatNumber(control)} Hz`;
  $("#controlValue").textContent = formatNumber(control);
  $("#frequencyMeta").textContent = `Policy ${formatNumber(policy)} Hz`;
  const alpha = tau === 0 ? 100 : (1 - Math.exp(-1 / (control * tau))) * 100;
  $("#smoothingHint").textContent = Number.isFinite(alpha)
    ? `${formatNumber(control)} Hz 下每步逼近 ${alpha.toFixed(2)}%`
    : "请输入有效频率";

  const checkpoint = selectedCheckpoint();
  if (checkpoint) {
    const configuredHorizon = Number($("#n_action_steps").value);
    const horizon = configuredHorizon > 0
      ? configuredHorizon
      : Number(checkpoint.n_action_steps ?? 10);
    $("#checkpointHint").textContent = `${checkpoint.policy_type} · chunk ${checkpoint.chunk_size ?? "?"} · RTC delay ${checkpoint.rtc_training_max_delay ?? 0}`;
    $("#executionHorizonValue").textContent = `${formatNumber(horizon, 0)} 步`;
    $("#executionHorizonMeta").textContent = $("#inference_mode").value === "sync"
      ? "同步模式：每个 chunk 实际执行"
      : "同步模式 horizon（当前 RTC 不参与调度）";
    $("#executionHorizonHint").textContent = `同步模式每个 chunk 执行 ${formatNumber(horizon, 0)} 步；0 使用训练清单，RTC 不使用此值`;
  } else {
    $("#executionHorizonValue").textContent = "—";
    $("#executionHorizonMeta").textContent = "每个 chunk 实际执行的动作步数";
  }
}

function syncCheckpointContract() {
  const checkpoint = selectedCheckpoint();
  if (!checkpoint) return;
  if (["delta_joint", "absolute_joint"].includes(checkpoint.action_mode)) {
    $("#action_mode").value = checkpoint.action_mode;
  }
  if ($("#rtc_mode").value === "trained" && !(checkpoint.rtc_training_max_delay > 0)) {
    $("#rtc_mode").value = "guided";
  }
  updateFormState();
  schedulePreview();
}

function schedulePreview() {
  clearTimeout(app.previewTimer);
  app.previewTimer = setTimeout(updatePreview, 280);
}

async function updatePreview() {
  try {
    const payload = await request("/api/preview", {
      method: "POST",
      body: JSON.stringify({ config: collectConfig() }),
    });
    app.commandText = `# 硬件桥\n${payload.bridge}\n\n# 模型端\n${payload.deployer}`;
    $("#commandPreview").textContent = app.commandText;
  } catch (error) {
    app.commandText = "";
    $("#commandPreview").textContent = `配置尚未通过检查：${error.message}`;
  }
}

function renderGpu(gpus, error = "") {
  const list = $("#gpuList");
  list.replaceChildren();
  if (!gpus || !gpus.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = error ? `GPU 探测失败：${error}` : "未读取到 NVIDIA GPU";
    list.append(empty);
    return;
  }
  for (const gpu of gpus) {
    const item = document.createElement("div");
    item.className = "gpu-item";
    const usage = gpu.memory_total_mb ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100 : 0;
    item.innerHTML = `
      <div class="gpu-row"><strong>GPU ${gpu.index} · ${escapeHtml(gpu.name)}</strong><span>${gpu.utilization}%</span></div>
      <div class="bar"><i style="width:${Math.min(100, usage)}%"></i></div>
      <div class="gpu-row gpu-memory"><span>${gpu.memory_used_mb} MB</span><span>${gpu.memory_total_mb} MB</span></div>`;
    list.append(item);
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderProcess(metricSelector, stateSelector, metaSelector, process, type) {
  const metric = $(metricSelector);
  metric.classList.toggle("running", process.running || ["ready", "completed"].includes(process.state));
  metric.classList.toggle("failed", process.state === "failed");
  $(stateSelector).textContent = processStateLabel(process.state);
  if (process.running) {
    const minutes = Math.floor(process.uptime_s / 60);
    const seconds = Math.floor(process.uptime_s % 60);
    let text = `PID ${process.pid} · ${minutes}:${String(seconds).padStart(2, "0")}`;
    if (type === "deployer" && process.progress?.current) {
      text = `Action ${process.progress.current}${process.progress.total ? ` / ${process.progress.total}` : ""}`;
    }
    $(metaSelector).textContent = text;
  } else if (process.exit_code !== null) {
    $(metaSelector).textContent = `退出状态 ${process.exit_code}`;
  } else {
    $(metaSelector).textContent = type === "bridge" ? "等待连接" : "PI0.5 · RTC";
  }
}

function renderStatus(status) {
  setConnection(true, "控制台已连接");
  renderProcess("#bridgeMetric", "#bridgeState", "#bridgeMeta", status.bridge, "bridge");
  renderProcess("#modelMetric", "#modelState", "#modelMeta", status.deployer, "deployer");

  const externalBridge = !status.bridge.running && status.bridge_port_open;
  if (externalBridge) {
    $("#bridgeState").textContent = "外部进程";
    $("#bridgeMeta").textContent = "端口 6007 已占用";
    $("#bridgeMetric").classList.add("running");
  }
  const externalDeployer = (status.external_processes || []).find((item) => item.command.includes("OpenpiDeployer.py"));
  if (!status.deployer.running && externalDeployer) {
    $("#modelState").textContent = "外部进程";
    $("#modelMeta").textContent = `PID ${externalDeployer.pid}`;
    $("#modelMetric").classList.add("running");
  }

  const diagnostics = status.diagnostics || {};
  $("#queueValue").textContent = diagnostics.latest_queue ?? "—";
  $("#queueMeta").textContent = diagnostics.latest_queue === null
    ? "等待动作"
    : `最低 ${diagnostics.minimum_queue ?? "—"} · ${diagnostics.action_ticks} ticks`;
  $("#queueCurrent").textContent = diagnostics.latest_queue ?? "—";
  $("#queueMinimum").textContent = diagnostics.minimum_queue ?? "—";
  $("#clippedTicks").textContent = diagnostics.clipped_ticks ?? 0;

  const summary = diagnostics.runtime_summary || {};
  $("#inferenceMedian").textContent = summary.inference_latency_s_median !== undefined
    ? `${(summary.inference_latency_s_median * 1000).toFixed(0)} ms`
    : "—";
  const runtimeStarved = summary.runtime_starved_ticks ?? diagnostics.runtime_starved_ticks;
  const prefillStarved = summary.prefill_starved_ticks ?? diagnostics.prefill_starved_ticks;
  const totalStarved = summary.starved_ticks ?? diagnostics.starved_ticks;
  $("#starvedTicks").textContent = runtimeStarved ?? "—";
  $("#starvedMeta").textContent = `预填充 ${prefillStarved ?? "—"} · 总计 ${totalStarved ?? "—"}`;
  const achieved = summary.move_js?.achieved_control_hz;
  $("#achievedHz").textContent = achieved !== undefined ? `${achieved.toFixed(2)} Hz` : "—";

  const bridgeActive = status.bridge.running && status.bridge.state === "ready";
  const modelActive = status.deployer.running;
  $("#stepBridge").className = bridgeActive ? "active" : status.bridge.state === "failed" ? "failed" : "";
  $("#stepModel").className = modelActive ? "active" : status.deployer.state === "failed" ? "failed" : "";
  $("#stepRun").className = status.deployer.state === "running" ? "active" : status.deployer.state === "failed" ? "failed" : "";

  const ownRunning = status.bridge.running || status.deployer.running;
  const externalRunning = Boolean((status.external_processes || []).length || externalBridge);
  $("#startStackButton").disabled = ownRunning || externalRunning || status.operation === "starting_stack";
  $("#startBridgeButton").disabled = status.bridge.running || externalBridge;
  $("#startModelButton").disabled = status.deployer.running || !status.bridge_port_open || Boolean(externalDeployer);
  $("#stopButton").disabled = !ownRunning;
  $("#globalStopButton").disabled = !ownRunning;

  if (externalRunning && !ownRunning) {
    $("#actionNote").textContent = "检测到外部终端进程。页面不会接管或停止非本页面启动的机械臂进程。";
  } else if (status.operation_error) {
    $("#actionNote").textContent = status.operation_error;
  } else {
    $("#actionNote").textContent = "安全停止等价于 Ctrl-C，并等待 follower 关节与夹爪回 ZERO。";
  }

  renderGpu(status.gpu, status.gpu_error);
  renderCharts(diagnostics.points || []);
}

function drawChart(canvas, values, color, options = {}) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(10, rect.width);
  const height = Math.max(10, rect.height);
  if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!values.length) return;
  const padding = { top: 12, right: 8, bottom: 14, left: 8 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const min = options.min ?? Math.min(...values);
  let max = options.max ?? Math.max(...values);
  if (max <= min) max = min + 1;

  context.strokeStyle = "rgba(255,255,255,.06)";
  context.lineWidth = 1;
  for (let index = 0; index <= 3; index += 1) {
    const y = padding.top + (chartHeight * index) / 3;
    context.beginPath(); context.moveTo(padding.left, y); context.lineTo(width - padding.right, y); context.stroke();
  }

  const points = values.map((value, index) => ({
    x: padding.left + (values.length === 1 ? chartWidth : (index / (values.length - 1)) * chartWidth),
    y: padding.top + chartHeight - ((value - min) / (max - min)) * chartHeight,
  }));
  context.beginPath();
  points.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.lineJoin = "round";
  context.stroke();

  context.lineTo(points.at(-1).x, padding.top + chartHeight);
  context.lineTo(points[0].x, padding.top + chartHeight);
  context.closePath();
  const fill = context.createLinearGradient(0, padding.top, 0, padding.top + chartHeight);
  fill.addColorStop(0, options.fill || "rgba(185,243,106,.2)");
  fill.addColorStop(1, "rgba(185,243,106,0)");
  context.fillStyle = fill;
  context.fill();
}

function renderCharts(points) {
  const queueValues = points.map((point) => point.queue).filter((value) => value !== null && value !== undefined);
  const latencyValues = points.map((point) => point.tick_ms).filter((value) => Number.isFinite(value) && value > 0);
  $("#queueEmpty").hidden = queueValues.length > 0;
  $("#latencyEmpty").hidden = latencyValues.length > 0;
  drawChart($("#queueChart"), queueValues, "#b9f36a", { min: 0, max: Math.max(50, ...queueValues), fill: "rgba(185,243,106,.22)" });
  const latencyMax = latencyValues.length ? Math.max(...latencyValues) : 100;
  drawChart($("#latencyChart"), latencyValues, "#64dbc4", { min: 0, max: Math.max(100, latencyMax), fill: "rgba(100,219,196,.2)" });
}

function appendLogs(entries) {
  if (!entries.length) return;
  $("#terminalPlaceholder")?.remove();
  const terminal = $("#terminal");
  const fragment = document.createDocumentFragment();
  for (const entry of entries) {
    const row = document.createElement("div");
    row.className = `terminal-line ${entry.level || "info"}`;
    const timestamp = document.createElement("time");
    timestamp.textContent = entry.time;
    const source = document.createElement("b");
    source.className = entry.source;
    source.textContent = entry.source;
    const text = document.createElement("span");
    text.textContent = entry.message;
    row.append(timestamp, source, text);
    fragment.append(row);
  }
  terminal.append(fragment);
  while (terminal.children.length > 1800) terminal.firstElementChild?.remove();
  if ($("#autoScroll").checked) terminal.scrollTop = terminal.scrollHeight;
}

async function pollStatus() {
  try {
    const status = await request("/api/status");
    renderStatus(status);
  } catch (error) {
    setConnection(false, "控制台连接中断");
  }
}

async function pollLogs() {
  try {
    const payload = await request(`/api/logs?after=${app.lastLogId}`);
    appendLogs(payload.entries || []);
    app.lastLogId = payload.latest_id ?? app.lastLogId;
  } catch (_) { /* status poll owns connection indicator */ }
}

async function runAction(path, successMessage) {
  const buttons = [$("#startStackButton"), $("#startBridgeButton"), $("#startModelButton")];
  buttons.forEach((button) => { button.disabled = true; });
  try {
    const body = path === "/api/bridge/start"
      ? { endpoint: $("#bridge_endpoint").value.trim() }
      : { config: collectConfig() };
    await request(path, { method: "POST", body: JSON.stringify(body) });
    toast(successMessage);
    await pollStatus();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function safeStop() {
  $("#stopButton").disabled = true;
  $("#globalStopButton").disabled = true;
  try {
    await request("/api/stop", { method: "POST", body: "{}" });
    toast("已发送安全停止，正在等待机械臂回零");
    await pollStatus();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function bootstrap() {
  try {
    const payload = await request("/api/bootstrap");
    // Continuous execution is the only Web-console preset. Keep this
    // explicit so a long-running server process with an older in-memory
    // default cannot restore a bounded run after a page refresh.
    app.defaults = { ...payload.defaults, max_steps: 0 };
    app.checkpoints = payload.checkpoints || [];
    populateOptions();
    const device = populateDevices(payload.status.gpu || [], app.defaults.device);
    applyConfig({ ...app.defaults, device });
    renderStatus(payload.status);
    setConnection(true, "控制台已连接");
  } catch (error) {
    setConnection(false, "控制台不可用");
    toast(error.message, "error");
  }
}

function bindEvents() {
  $("#configForm").addEventListener("input", () => {
    updateFormState();
    schedulePreview();
  });
  $("#checkpoint").addEventListener("change", syncCheckpointContract);
  $("#startStackButton").addEventListener("click", () => runAction("/api/stack/start", "完整启动流程已开始"));
  $("#startBridgeButton").addEventListener("click", () => runAction("/api/bridge/start", "硬件桥正在启动"));
  $("#startModelButton").addEventListener("click", () => runAction("/api/deployment/start", "模型端正在启动"));
  $("#stopButton").addEventListener("click", safeStop);
  $("#globalStopButton").addEventListener("click", safeStop);
  $("#refreshButton").addEventListener("click", async () => {
    await bootstrap();
    toast("配置与状态已刷新");
  });
  $("#clearLogsButton").addEventListener("click", async () => {
    try {
      await request("/api/logs/clear", { method: "POST", body: "{}" });
      $("#terminal").replaceChildren();
      app.lastLogId = 0;
      toast("日志已清空");
    } catch (error) { toast(error.message, "error"); }
  });
  $("#copyCommandButton").addEventListener("click", async () => {
    if (!app.commandText) return toast("当前配置未生成有效命令", "error");
    try {
      await navigator.clipboard.writeText(app.commandText);
      toast("命令已复制");
    } catch (_) {
      toast("浏览器未允许访问剪贴板", "error");
    }
  });
  window.addEventListener("resize", () => pollStatus());
}

bindEvents();
bootstrap().then(() => {
  pollLogs();
  app.statusTimer = setInterval(pollStatus, 1000);
  app.logTimer = setInterval(pollLogs, 650);
});
