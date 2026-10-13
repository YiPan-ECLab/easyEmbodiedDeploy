"use strict";

const $ = (id) => document.getElementById(id);
const ui = {
  plan: null,
  defaults: null,
  lastJob: null,
  lastEventSeq: 0,
  locked: false,
  timingOverrides: {},
  inspectionToken: 0,
};
const statusNames = {
  idle: "空闲", running: "处理中", complete: "已完成",
  failed: "失败", cancelled: "已取消",
};
const stageNames = {
  checking: "检查输入", loading: "加载 LeRobot", workers: "分配 CPU worker", resampling: "准备采样",
  writing: "写入帧", encoding: "编码视频", finalizing: "生成统计量",
  complete: "验收完成", failed: "处理失败", cancelled: "已取消",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.className = "toast"; }, 2600);
}

async function request(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function checked(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value || "";
}

function inputDirectories() {
  return [...new Set($("inputDir").value.split(/\r?\n/).map((value) => value.trim()).filter(Boolean))];
}

function formPayload() {
  return {
    input_dir: inputDirectories()[0] || "",
    input_dirs: inputDirectories(),
    output_root: $("outputRoot").value.trim() || null,
    task: $("task").value.trim(),
    source_fps: $("sourceFps").value ? Number($("sourceFps").value) : null,
    target_fps: Number($("targetFps").value),
    action_mode: checked("actionMode"),
    calibration_bundle: $("calibrationBundle").value.trim(),
    calibration_mount_id: $("calibrationMountId").value.trim(),
    gap_policy: checked("gapPolicy"),
    timing_overrides: {...ui.timingOverrides},
    repo_id: $("repoId").value.trim() || null,
    limit: Number($("limit").value || 0),
  };
}

function shellQuote(value) {
  return `'${String(value).replaceAll("'", `'"'"'`)}'`;
}

function updateCommand() {
  const value = formPayload();
  const cameraPose = value.action_mode.startsWith("camera_delta_");
  const pose = cameraPose || value.action_mode === "absolute_base_pose";
  document.querySelectorAll("[data-camera-pose]").forEach(el => { el.hidden = !cameraPose; });
  const compact = document.querySelector('input[name="gapPolicy"][value="compact"]');
  compact.disabled = ui.locked || pose;
  if (pose) { document.querySelector('input[name="gapPolicy"][value="reject"]').checked = true; value.gap_policy = "reject"; }
  const parts = [
    "CUDA_VISIBLE_DEVICES='' thirdParty/.venvs/lerobot-pi05/bin/python \\",
    "  scripts/LerobotDataProcessor/convert_piper_hdf5.py \\",
  ];
  (value.input_dirs.length ? value.input_dirs : [value.input_dir]).forEach((path) => {
    parts.push(`  --input-dir ${shellQuote(path)} \\`);
  });
  if (value.output_root) parts.push(`  --output-root ${shellQuote(value.output_root)} \\`);
  if (value.repo_id) parts.push(`  --repo-id ${shellQuote(value.repo_id)} \\`);
  if (cameraPose) {
    parts.push(`  --calibration-bundle ${shellQuote(value.calibration_bundle)} \\`);
    parts.push(`  --calibration-mount-id ${shellQuote(value.calibration_mount_id)} \\`);
  }
  parts.push(`  --task ${shellQuote(value.task)} \\`);
  if (value.source_fps) parts.push(`  --source-fps ${value.source_fps} \\`);
  parts.push(`  --fps ${value.target_fps} --action-mode ${value.action_mode} \\`);
  parts.push(`  --gap-policy ${value.gap_policy}${value.limit ? ` --limit ${value.limit}` : ""}`);
  for (const [path, revision] of Object.entries(value.timing_overrides)) {
    parts[parts.length - 1] += " \\";
    parts.push(`  --allow-timing ${shellQuote(path)} ${shellQuote(revision)}`);
  }
  $("commandPreview").textContent = parts.join("\n");
}

function formatNumber(value) {
  return value != null && Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "—";
}

function formatTime(seconds) {
  if (!Number.isFinite(Number(seconds))) return "—";
  const value = Math.max(0, Math.round(Number(seconds)));
  if (value < 60) return `${value}s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}m ${String(value % 60).padStart(2, "0")}s`;
}

function setConnected(connected) {
  $("connection").className = `connection ${connected ? "online" : "offline"}`;
  $("connection").querySelector("b").textContent = connected ? "服务已连接" : "连接中断";
}

function setLocked(locked) {
  ui.locked = locked;
  $("configForm").querySelectorAll("input, textarea").forEach((node) => { node.disabled = locked; });
  $("inspectButton").disabled = locked;
  $("startButton").classList.toggle("hidden", locked);
  $("cancelButton").classList.toggle("hidden", !locked);
  $("cancelButton").disabled = false;
  renderFileTable();
  updateCommand();
}

function invalidatePlan() {
  if (ui.locked) return;
  ui.inspectionToken += 1;
  ui.timingOverrides = {};
  $("inspectButton").disabled = false;
  ui.plan = null;
  $("metricQuality").textContent = "需重新检查";
  $("metricQualityMeta").textContent = "配置已修改";
  $("stepInspect").className = "";
  updateCommand();
}

function renderPlan(plan) {
  ui.plan = plan;
  $("inspectionPanel").classList.remove("hidden");
  $("metricEpisodes").textContent = formatNumber(plan.episodes);
  $("metricEpisodeMeta").textContent = `${plan.files.length} 个 HDF5`;
  $("metricInput").textContent = formatNumber(plan.input_frames);
  $("metricInputMeta").textContent = `${formatNumber(plan.files.reduce((sum, file) => sum + file.source_task_frames, 0))} task`;
  $("metricOutput").textContent = formatNumber(plan.output_frames);
  $("metricOutputMeta").textContent = `${plan.target_fps ?? '—'} Hz · ${plan.action_mode ?? '—'}`;
  $("metricQuality").textContent = plan.invalid_files ? "有待处理错误" : plan.warnings.length ? "有警告" : "检查通过";
  $("metricQualityMeta").textContent = `${plan.invalid_files || 0} 个文件未通过 · ${plan.gaps} 个时序异常`;
  $("stepInspect").className = plan.warnings.length ? "active" : "done";
  $("writtenFrames").textContent = formatNumber(plan.output_frames);
  const warnings = $("warningBox");
  warnings.classList.toggle("hidden", !plan.warnings.length && !plan.errors?.length);
  warnings.textContent = [...(plan.errors || []), ...plan.warnings].join("\n");
  renderFileTable();
}

function renderFileTable() {
  if (!ui.plan) return;
  const query = $("fileSearch").value.trim().toLowerCase();
  const issuesOnly = $("issuesOnly").checked;
  const files = ui.plan.files.filter((file) => {
    const issue = file.errors?.length || file.gaps || file.warnings?.length;
    return file.name.toLowerCase().includes(query) && (!issuesOnly || issue);
  });
  $("fileTable").innerHTML = files.map((file) => {
    const issue = file.errors?.length || file.gaps || file.warnings?.length;
    const index = ui.plan.files.indexOf(file);
    const details = [...(file.errors || []), ...(file.timing_ignored ? ["已人工允许时序异常", ...(file.timing_errors || [])] : []), ...(file.warnings || [])];
    return `<tr class="${issue ? "issue" : ""}">
      <td title="${escapeHtml(file.path || file.name)}">${escapeHtml(file.name)}</td>
      <td>${formatNumber(file.frames)}</td><td>${formatNumber(file.source_task_frames)}</td>
      <td>${formatNumber(file.source_home_frames)}</td><td>${formatNumber(file.trimmed_leading_idle_frames || 0)} / ${formatNumber(file.trimmed_idle_frames)}</td>
      <td>${file.temporal_fusion ? "mixed" : file.measured_fps != null ? Number(file.measured_fps).toFixed(3) : "—"}</td><td>${formatNumber(file.output_frames)}</td>
      <td>${file.max_camera_skew_s != null ? (Number(file.max_camera_skew_s) * 1000).toFixed(1) + " ms" : "—"}</td>
      <td>${file.gaps || "0"}</td>
      <td class="file-issues">${details.length ? details.map(escapeHtml).join("<br>") : "通过"}</td>
      <td class="file-actions"><button data-delete="${index}" ${ui.locked ? "disabled" : ""}>删除轨迹</button>
      ${file.can_ignore_timing || file.timing_ignored ? `<button data-timing="${index}" ${ui.locked ? "disabled" : ""}>${file.timing_ignored ? "撤销时序放行" : "忽略时序错误"}</button>` : ""}</td></tr>`;
  }).join("") || '<tr><td colspan="11">没有匹配的文件</td></tr>';
}

async function handleFileAction(event) {
  if (ui.locked || !ui.plan) return;
  const button = event.target.closest("button");
  if (!button) return;
  const file = ui.plan.files[Number(button.dataset.delete ?? button.dataset.timing)];
  if (!file) return;
  button.disabled = true;
  try {
    if (button.dataset.timing !== undefined) {
      if (file.timing_ignored) delete ui.timingOverrides[file.path];
      else {
        if (!window.confirm(`仅忽略这条轨迹的物理时序错误？\n${file.path}\n${file.errors.join("\n")}\n保留现有选帧方式，不插值。其他数据错误仍会拦截。`)) return;
        ui.timingOverrides[file.path] = file.revision;
      }
      updateCommand();
    } else {
      if (!window.confirm(`将轨迹移入回收站？\n${file.path}\n可以撤销删除。`)) return;
      const result = await request("api/trajectory/delete", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({path: file.path, revision: file.revision})});
      delete ui.timingOverrides[file.path];
      const notice = $("deleteResult");
      notice.classList.remove("hidden"); notice.textContent = `${file.name} 已移入回收站。 `;
      const undo = document.createElement("button"); undo.textContent = "撤销删除";
      undo.onclick = async () => {
        if (ui.locked) return;
        undo.disabled = true;
        try {
          await request("api/trajectory/restore", {method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({path: file.path, id: result.id})});
          notice.classList.add("hidden"); await inspectInput(false);
        } catch (error) { toast(error.message, true); undo.disabled = false; }
      };
      notice.append(undo);
    }
    await inspectInput(false);
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; }
}

async function inspectInput(showMessage = true) {
  const token = ++ui.inspectionToken;
  $("inspectButton").disabled = true;
  $("runNote").textContent = "正在读取全部 HDF5 元数据与质量指标…";
  $("stepInspect").className = "active";
  try {
    const plan = await request("api/inspect", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });
    if (token !== ui.inspectionToken) throw new Error("配置已变化，请重新检查");
    renderPlan(plan);
    $("runNote").textContent = plan.invalid_files ? `有 ${plan.invalid_files} 条轨迹未通过，请查看下方具体问题。` : plan.warnings.length
      ? `检查完成，发现 ${plan.warnings.length} 条警告。`
      : "检查通过，可以开始转换。";
    if (showMessage) toast(`检查完成：${plan.episodes} 条，${plan.invalid_files || 0} 条未通过`, !!plan.invalid_files);
    return plan;
  } catch (error) {
    if (token !== ui.inspectionToken) throw error;
    ui.plan = null;
    $("stepInspect").className = "";
    $("metricQuality").textContent = "检查失败";
    $("metricQualityMeta").textContent = error.message;
    $("runNote").textContent = error.message;
    toast(error.message, true);
    throw error;
  } finally {
    if (token === ui.inspectionToken) $("inspectButton").disabled = ui.locked;
  }
}

function setPipeline(stage, status) {
  const writing = ["loading", "workers", "resampling", "writing", "encoding"].includes(stage);
  $("stepInspect").className = stage === "checking" ? "active" : "done";
  $("stepWrite").className = writing ? "active" : (["finalizing", "complete"].includes(stage) ? "done" : "");
  $("stepValidate").className = stage === "finalizing" ? "active" : (stage === "complete" ? "done" : "");
  if (["failed", "cancelled"].includes(status)) {
    [$("stepInspect"), $("stepWrite"), $("stepValidate")].forEach((node) => node.classList.remove("active"));
  }
}

function appendEvents(events, jobId) {
  if (ui.lastJob !== jobId) {
    ui.lastJob = jobId;
    ui.lastEventSeq = 0;
    $("log").replaceChildren();
  }
  events.filter((event) => event.seq > ui.lastEventSeq).forEach((event) => {
    const line = document.createElement("p");
    const time = document.createElement("time");
    time.textContent = `${formatTime(event.elapsed)} · ${stageNames[event.stage] || event.stage}`;
    line.append(time, document.createTextNode(
      [event.message, event.file, event.frame ? `${event.frame}/${event.frames}` : ""].filter(Boolean).join(" · ")
    ));
    $("log").appendChild(line);
    ui.lastEventSeq = event.seq;
  });
  $("log").scrollTop = $("log").scrollHeight;
}

function renderStatus(state) {
  const running = state.status === "running";
  setLocked(running);
  // Polling a previous job must not replace a newer input review or its actions.
  if (state.plan && (running || ui.lastJob !== state.job_id)) renderPlan(state.plan);
  const badge = $("statusBadge");
  badge.className = `status ${state.status}`;
  badge.lastChild.textContent = statusNames[state.status] || state.status;
  const event = state.events?.at(-1) || {};
  const stage = event.stage || state.status;
  setPipeline(stage, state.status);
  $("phaseLabel").textContent = stageNames[stage] || statusNames[state.status] || "等待开始";
  $("progressMessage").textContent = state.error || event.message || "输入检查完成后可启动转换。";
  const episodes = event.episodes || state.plan?.episodes || 0;
  const episodeProgress = Math.min(event.episode || 0, episodes);
  let progress = episodes ? episodeProgress / episodes : 0;
  if (event.frame && event.frames && episodes) progress = (episodeProgress + event.frame / event.frames) / episodes;
  if (state.status === "complete") progress = 1;
  $("progressBar").style.width = `${Math.max(0, Math.min(100, progress * 100))}%`;
  $("progressMain").textContent = `${episodeProgress} / ${episodes} episode`;
  $("progressFrame").textContent = event.frame ? `${event.frame} / ${event.frames} 帧` : "—";
  $("elapsed").textContent = formatTime(event.elapsed);
  $("writtenFrames").textContent = formatNumber(state.plan?.output_frames);
  $("eta").textContent = progress > 0 && progress < 1 && event.elapsed
    ? formatTime(event.elapsed * (1 - progress) / progress) : "—";
  appendEvents(state.events || [], state.job_id);
  const result = $("resultBox");
  if (state.status === "complete" && state.result) {
    const validation = state.result.validation;
    result.className = `result-box${validation?.valid ? "" : " error"}`;
    result.innerHTML = `<strong>${validation?.valid ? "转换与 CPU 验收通过" : "输出验收失败"}</strong><br>${escapeHtml(state.result.plan.output_root)}<br><button id="validateAgain" class="text-button" type="button">重新验收输出</button>`;
    $("validateAgain").onclick = () => validateOutput(state.result.plan.output_root);
  } else if (["failed", "cancelled"].includes(state.status)) {
    result.className = "result-box error";
    result.textContent = state.error || statusNames[state.status];
  } else {
    result.className = "result-box hidden";
    result.textContent = "";
  }
}

async function refreshStatus() {
  try {
    const state = await request("api/status");
    setConnected(true);
    renderStatus(state);
  } catch (error) {
    setConnected(false);
    $("progressMessage").textContent = error.message;
  }
}

async function startConversion() {
  try {
    // The backend rejects invalid inputs; warnings also include informational
    // notices such as preserving an intentionally fused sequence.
    const plan = await inspectInput(false);
    if (plan.invalid_files) throw new Error("请先处理下方轨迹错误，或人工确认允许对应的时序异常");
    const response = await request("api/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });
    renderPlan(response.plan);
    setLocked(true);
    toast(`转换任务 #${response.job_id} 已启动`);
    await refreshStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

async function cancelConversion() {
  if (!window.confirm("确定取消当前转换？当前 episode 写入完成后会安全停止。")) return;
  $("cancelButton").disabled = true;
  try {
    await request("api/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    $("progressMessage").textContent = "已请求取消，正在等待安全写入点…";
  } catch (error) {
    toast(error.message, true);
    $("cancelButton").disabled = false;
  }
}

async function validateOutput(path) {
  try {
    const result = await request("api/validate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    toast(result.valid ? `验收通过：${formatNumber(result.metrics.rows)} 帧` : result.errors.join("；"), !result.valid);
  } catch (error) { toast(error.message, true); }
}

async function loadDefaults() {
  ui.defaults = await request("api/defaults");
  $("inputDir").value = ui.defaults.input_dir;
  $("outputRoot").value = ui.defaults.output_root || "";
  $("task").value = ui.defaults.task;
  $("sourceFps").value = ui.defaults.source_fps || "";
  $("targetFps").value = ui.defaults.target_fps;
  $("repoId").value = ui.defaults.repo_id || "";
  $("limit").value = ui.defaults.limit;
  document.querySelector(`input[name="actionMode"][value="${ui.defaults.action_mode}"]`).checked = true;
  document.querySelector(`input[name="gapPolicy"][value="${ui.defaults.gap_policy}"]`).checked = true;
  updateCommand();
}

$("configForm").addEventListener("input", invalidatePlan);
$("configForm").addEventListener("change", invalidatePlan);
$("inspectButton").addEventListener("click", () => inspectInput().catch(() => {}));
$("startButton").addEventListener("click", startConversion);
$("cancelButton").addEventListener("click", cancelConversion);
$("refresh").addEventListener("click", refreshStatus);
$("fileSearch").addEventListener("input", renderFileTable);
$("issuesOnly").addEventListener("change", renderFileTable);
$("fileTable").addEventListener("click", handleFileAction);
$("clearLog").addEventListener("click", () => $("log").replaceChildren());
$("copyCommand").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("commandPreview").textContent); toast("命令已复制"); }
  catch { toast("浏览器未允许访问剪贴板", true); }
});

Promise.all([loadDefaults(), refreshStatus()])
  .then(() => setConnected(true))
  .catch((error) => { setConnected(false); toast(error.message, true); });
setInterval(refreshStatus, 1000);
