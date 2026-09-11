"use strict";

const $ = (id) => document.getElementById(id);
const state = { defaults: null, report: null, status: "idle" };

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

async function request(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.className = "toast"; }, 2500);
}

function kind() { return document.querySelector('input[name="kind"]:checked').value; }

function formPayload() {
  return {
    kind: kind(), path: $("path").value.trim(),
    expected_fps: Number($("expectedFps").value || 30),
    gap_factor: Number($("gapFactor").value || 1.5),
    source_dir: $("sourceDir").value.trim() || null,
  };
}

function selectKind() {
  const hdf5 = kind() === "hdf5";
  $("hdf5Options").classList.toggle("hidden", !hdf5);
  $("sourceField").classList.toggle("hidden", hdf5);
  $("path").value = hdf5 ? state.defaults.hdf5_path : state.defaults.lerobot_path;
  $("metricUnitLabel").textContent = hdf5 ? "HDF5 文件" : "LeRobot Episode";
}

function setConnection(online) {
  $("connection").className = `connection ${online ? "online" : "offline"}`;
  $("connection").querySelector("b").textContent = online ? "服务已连接" : "连接中断";
}

function setRunning(running) {
  ["path", "expectedFps", "gapFactor", "sourceDir"].forEach((id) => { $(id).disabled = running; });
  document.querySelectorAll('input[name="kind"]').forEach((node) => { node.disabled = running; });
  $("start").classList.toggle("hidden", running);
  $("cancel").classList.toggle("hidden", !running);
  $("cancel").disabled = false;
  $("progress").parentElement.classList.toggle("running", running);
}

function setGateState(status) {
  const gates = document.querySelector(".gates");
  gates.classList.remove("pass", "fail");
  if (status === "PASS") gates.classList.add("pass");
  if (status === "FAIL") gates.classList.add("fail");
  const badge = $("reportBadge");
  badge.className = `badge ${status === "PASS" ? "pass" : status === "FAIL" ? "fail" : "neutral"}`;
  badge.textContent = status || "WAIT";
}

function summaryBlocks(source) {
  if (!source) return "";
  return [
    ["动作最大误差", Number(source.max_action_error).toExponential(2)],
    ["状态最大误差", Number(source.max_state_error).toExponential(2)],
    ["输出 Home 帧", source.home_frames_in_output],
    ["最大源间隔", `${Number(source.max_selected_source_dt_ms).toFixed(3)} ms`],
    ["裁剪尾帧", source.trimmed_idle_frames],
  ].map(([name, value]) => `<div><small>${name}</small><b>${value}</b></div>`).join("");
}

function renderHdf5Rows() {
  const query = $("search").value.trim().toLowerCase();
  const issues = $("issues").checked;
  const rows = state.report.files.filter((item) =>
    item.file.toLowerCase().includes(query) && (!issues || !item.valid || item.warnings.length)
  );
  $("tableHead").innerHTML = "<tr><th>结果</th><th>文件</th><th>Task</th><th>Home</th><th>尾部等待</th><th>实测 FPS</th><th>间隔范围</th><th>相机偏差</th><th>边界裁剪</th><th>原因</th></tr>";
  $("tableBody").innerHTML = rows.map((item) => {
    const m = item.metrics;
    const rowClass = item.valid ? (item.warnings.length ? "warning" : "") : "fail";
    const detail = [...item.errors, ...item.warnings].join("；") || "—";
    return `<tr class="${rowClass}"><td>${item.valid ? (item.warnings.length ? "WARN" : "PASS") : "FAIL"}</td>
      <td title="${escapeHtml(item.file)}">${escapeHtml(item.file)}</td><td>${m.task_frames}</td><td>${m.home_frames}</td>
      <td>${m.trailing_idle_frames}</td><td>${Number(m.measured_fps).toFixed(3)}</td>
      <td>${Number(m.task_dt_min_ms).toFixed(2)}–${Number(m.task_dt_max_ms).toFixed(2)} ms</td>
      <td>${Number(m.max_camera_skew_ms).toFixed(1)} ms</td><td>${m.boundary_clipped_values}</td>
      <td title="${escapeHtml(detail)}">${escapeHtml(detail)}</td></tr>`;
  }).join("") || '<tr><td colspan="10">没有匹配项</td></tr>';
}

function renderLerobotRows() {
  const query = $("search").value.trim().toLowerCase();
  const issues = $("issues").checked;
  const rows = state.report.episodes.filter((item) => {
    const hasIssue = item.errors.length > 0;
    return String(item.episode).includes(query) && (!issues || hasIssue);
  });
  $("tableHead").innerHTML = "<tr><th>结果</th><th>Episode</th><th>源文件</th><th>输出帧</th><th>首部静止</th><th>尾部静止</th><th>最大动作步长</th><th>最大源间隔</th><th>Action 误差</th><th>State 误差</th><th>Home</th><th>错误</th></tr>";
  $("tableBody").innerHTML = rows.map((item) => `<tr class="${item.errors.length ? "fail" : ""}">
    <td>${item.errors.length ? "FAIL" : "PASS"}</td><td>${item.episode}</td>
    <td title="${escapeHtml(item.source_file || "—")}">${escapeHtml(item.source_file || "—")}</td><td>${item.rows}</td>
    <td>${item.leading_idle_frames}</td><td>${item.trailing_idle_frames}</td>
    <td>${Number(item.max_action_joint_step_rad).toFixed(4)} rad</td>
    <td>${item.max_source_dt_ms == null ? "—" : `${Number(item.max_source_dt_ms).toFixed(3)} ms`}</td>
    <td>${item.action_error == null ? "—" : Number(item.action_error).toExponential(2)}</td>
    <td>${item.state_error == null ? "—" : Number(item.state_error).toExponential(2)}</td>
    <td>${item.home_frames ?? "—"}</td><td>${escapeHtml(item.errors.join("；") || "—")}</td></tr>`).join("") || '<tr><td colspan="12">没有匹配项</td></tr>';
}

function renderReport(report) {
  state.report = report;
  const hdf5 = Boolean(report.files);
  const summary = report.summary || {};
  $("report").classList.remove("hidden");
  $("reportTitle").textContent = `${hdf5 ? "HDF5" : "LeRobot"} 数据质量报告`;
  $("reportPath").textContent = report.root || report.bundle;
  $("metricStatus").textContent = report.status;
  $("metricStatusMeta").textContent = report.status === "PASS" ? "全部质量门通过" : "存在阻断问题";
  $("metricUnits").textContent = hdf5 ? summary.files : summary.episodes;
  $("metricFrames").textContent = Number(summary.frames || 0).toLocaleString();
  $("metricFramesMeta").textContent = hdf5
    ? `${Number(summary.task_frames || 0).toLocaleString()} task`
    : `${summary.fps} Hz · ${summary.action_mode}`;
  $("metricErrors").textContent = summary.errors || 0;
  $("metricWarnings").textContent = `${summary.warnings || 0} warning`;
  setGateState(report.status);
  const source = $("sourceSummary");
  source.classList.toggle("hidden", !report.source_verification);
  source.innerHTML = summaryBlocks(report.source_verification);
  const itemErrors = hdf5
    ? report.files.flatMap((item) => item.errors.map((value) => `${item.file}: ${value}`)) : [];
  const itemWarnings = hdf5
    ? report.files.flatMap((item) => item.warnings.map((value) => `${item.file}: ${value}`)) : [];
  const messages = [...(report.errors || itemErrors).map((value) => `ERROR  ${value}`),
                    ...(report.warnings || itemWarnings).map((value) => `WARN   ${value}`)];
  $("messages").classList.toggle("hidden", !messages.length);
  $("messages").textContent = messages.join("\n");
  if (hdf5) renderHdf5Rows(); else renderLerobotRows();
}

async function refresh() {
  try {
    const value = await request("/api/status");
    setConnection(true);
    state.status = value.status;
    const running = value.status === "running";
    setRunning(running);
    $("elapsed").textContent = value.elapsed == null ? "—" : `${Number(value.elapsed).toFixed(1)} s`;
    $("runMessage").textContent = value.error || value.events?.at(-1)?.message || (running ? "正在分析…" : "等待开始");
    if (running) {
      $("metricStatus").textContent = "分析中";
      $("metricStatusMeta").textContent = value.kind?.toUpperCase() || "CPU";
      setGateState(null);
    }
    if (value.status === "complete" && value.report) renderReport(value.report);
    if (["failed", "cancelled"].includes(value.status)) {
      $("metricStatus").textContent = value.status === "failed" ? "运行失败" : "已取消";
      $("metricStatusMeta").textContent = value.error || "";
      setGateState("FAIL");
    }
  } catch (error) {
    setConnection(false);
    $("runMessage").textContent = error.message;
  }
}

async function start() {
  try {
    const response = await request("/api/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });
    state.report = null;
    $("report").classList.add("hidden");
    setRunning(true);
    toast(`分析任务 #${response.job_id} 已启动`);
    refresh();
  } catch (error) { toast(error.message, true); }
}

async function cancel() {
  $("cancel").disabled = true;
  try {
    await request("/api/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    $("runMessage").textContent = "正在停止分析…";
  } catch (error) { toast(error.message, true); $("cancel").disabled = false; }
}

function download() {
  if (!state.report) return;
  const blob = new Blob([JSON.stringify(state.report, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${state.report.files ? "hdf5" : "lerobot"}_quality_report.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

document.querySelectorAll('input[name="kind"]').forEach((node) => node.addEventListener("change", selectKind));
$("start").addEventListener("click", start);
$("cancel").addEventListener("click", cancel);
$("download").addEventListener("click", download);
$("search").addEventListener("input", () => state.report && (state.report.files ? renderHdf5Rows() : renderLerobotRows()));
$("issues").addEventListener("change", () => state.report && (state.report.files ? renderHdf5Rows() : renderLerobotRows()));

request("/api/defaults").then((value) => {
  state.defaults = value;
  $("expectedFps").value = value.expected_fps;
  $("gapFactor").value = value.gap_factor;
  selectKind();
  return refresh();
}).catch((error) => { setConnection(false); toast(error.message, true); });
setInterval(refresh, 800);
