"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const COLORS = ["#087f6d", "#d17a12", "#2473a6", "#a84363", "#7155a5", "#65722c"];
const CAMERA_LABELS = {
  agentview: "场景相机",
  agentview_rgb: "场景相机",
  left_eye_in_hand: "左腕相机",
  left_eye_in_hand_rgb: "左腕相机",
  right_eye_in_hand: "右腕相机",
  right_eye_in_hand_rgb: "右腕相机",
};

const state = {
  files: [],
  episode: null,
  selectedPath: "",
  frame: 0,
  playing: false,
  speed: 1,
  animationFrame: null,
  frameLoadPending: false,
  playbackStartedAt: 0,
  playbackStartedMediaTime: 0,
  curveSource: "actions",
  side: "left",
  visibleJoints: new Set([0, 1, 2, 3, 4, 5]),
  loadToken: 0,
  playbackToken: 0,
};

async function api(path, params = {}) {
  const url = new URL(path, window.location.href);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, value);
  });
  const response = await fetch(url);
  let payload;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) throw new Error(payload?.error || `请求失败 (${response.status})`);
  return payload;
}

function icons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "--";
  const number = Number(value);
  return Math.abs(number) >= 1000 ? number.toExponential(3) : number.toFixed(digits);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** power).toFixed(power > 1 ? 1 : 0)} ${units[power]}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "--";
  if (seconds < 60) return `${seconds.toFixed(2)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

let toastTimer;
function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("error", error);
  node.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("visible"), 2400);
}

function setScanStatus(message, error = false) {
  const node = $("#scan-status");
  node.textContent = message;
  node.classList.toggle("error", error);
}

async function scanDirectory(directory) {
  stopPlayback();
  setScanStatus("正在扫描...");
  $("#file-list").innerHTML = '<div class="list-message">正在读取 HDF5 文件...</div>';
  try {
    const result = await api("api/files", { directory });
    state.files = result.files;
    $("#directory-input").value = result.directory;
    localStorage.setItem("piperViewerDirectory", result.directory);
    $("#library-summary").textContent = `${result.count} 个文件 · ${result.total_frames.toLocaleString()} 帧 · ${formatBytes(result.total_size)}`;
    setScanStatus(result.errors.length ? `${result.count} 个可用，${result.errors.length} 个读取失败` : `${result.count} 个 episode` , result.errors.length > 0);
    renderFileList();
    if (result.files.length && !state.episode) selectFile(result.files[0].path);
    if (!result.files.length) {
      $("#empty-state h1").textContent = "目录中没有 HDF5 文件";
      $("#empty-state p").textContent = result.directory;
      $("#empty-state").classList.remove("hidden");
      $("#episode-view").classList.add("hidden");
    }
  } catch (error) {
    state.files = [];
    renderFileList();
    setScanStatus(error.message, true);
    toast(error.message, true);
  }
}

function renderFileList() {
  const query = $("#file-search").value.trim().toLowerCase();
  const files = state.files.filter((file) => `${file.relative_path} ${file.instruction}`.toLowerCase().includes(query));
  const list = $("#file-list");
  if (!files.length) {
    list.innerHTML = `<div class="list-message">${state.files.length ? "没有匹配的 episode" : "没有可浏览的文件"}</div>`;
    return;
  }
  list.innerHTML = files.map((file) => `
    <button class="file-row ${file.path === state.selectedPath ? "active" : ""}" data-path="${escapeHtml(file.path)}" role="option" aria-selected="${file.path === state.selectedPath}">
      <span class="file-icon"><i data-lucide="file-axis-3d"></i></span>
      <span class="file-content">
        <span class="file-name" title="${escapeHtml(file.relative_path)}">${escapeHtml(file.relative_path)}</span>
        <span class="file-task">${escapeHtml(file.instruction || "未设置任务描述")}</span>
        <span class="file-meta"><span>${file.frames.toLocaleString()} 帧</span><span>${file.cameras} 相机</span><span>${formatBytes(file.size)}</span></span>
      </span>
    </button>`).join("");
  list.querySelectorAll(".file-row").forEach((row) => row.addEventListener("click", () => selectFile(row.dataset.path)));
  icons();
}

async function selectFile(path, demo = null) {
  stopPlayback();
  state.selectedPath = path;
  state.frame = 0;
  renderFileList();
  const token = ++state.loadToken;
  $("#empty-state").classList.remove("hidden");
  $("#empty-state h1").textContent = "正在读取 episode";
  $("#empty-state p").textContent = path;
  $("#episode-view").classList.add("hidden");
  try {
    const episode = await api("api/episode", { path, demo });
    if (token !== state.loadToken) return;
    state.episode = episode;
    state.curveSource = "actions";
    $("#empty-state").classList.add("hidden");
    $("#episode-view").classList.remove("hidden");
    renderEpisode();
    if (window.innerWidth <= 700) $(".workspace").classList.add("sidebar-collapsed");
  } catch (error) {
    if (token !== state.loadToken) return;
    $("#empty-state h1").textContent = "无法读取 episode";
    $("#empty-state p").textContent = error.message;
    toast(error.message, true);
  }
}

function renderEpisode() {
  const episode = state.episode;
  $("#episode-name").textContent = episode.name;
  $("#episode-path").textContent = episode.path;
  $("#timeline").max = Math.max(0, episode.frames - 1);
  $("#timeline").value = 0;
  $("#frame-input").max = Math.max(1, episode.frames);
  $("#frame-input").value = 1;

  const direct = !episode.normalized && episode.representation.toLowerCase().includes("direct");
  const phase = episode.series.trajectory_phase || [];
  const homeFrames = phase.filter((value) => Number(value) === 1).length;
  $("#episode-metrics").innerHTML = [
    `<span class="metric ${direct ? "direct" : episode.normalized ? "normalized" : ""}"><strong>${direct ? "直接控制量" : episode.normalized ? "归一化" : escapeHtml(episode.representation)}</strong>${escapeHtml(episode.action_units)}</span>`,
    `<span class="metric"><strong>${episode.frames.toLocaleString()}</strong>帧</span>`,
    `<span class="metric"><strong>${formatDuration(episode.duration)}</strong>时长</span>`,
    `<span class="metric"><strong>${episode.cameras.length}</strong>相机</span>`,
    phase.length ? `<span class="metric"><strong>${homeFrames}</strong>HOME 帧</span>` : "",
  ].join("");
  const instruction = $("#instruction");
  instruction.textContent = episode.instruction;
  instruction.classList.toggle("hidden", !episode.instruction);

  const demoWrap = $("#demo-select-wrap");
  demoWrap.classList.toggle("hidden", episode.demos.length <= 1);
  $("#demo-select").innerHTML = episode.demos.map((name) => `<option ${name === episode.demo ? "selected" : ""}>${escapeHtml(name)}</option>`).join("");

  renderCameras();
  renderJointLegend();
  renderSchema();
  $$("#curve-source button").forEach((button) => button.classList.toggle("active", button.dataset.source === "actions"));
  updateFrame(0, true);
  icons();
}

function renderCameras() {
  const grid = $("#camera-grid");
  if (!state.episode.cameras.length) {
    grid.innerHTML = '<div class="no-cameras">此 demo 没有 RGB 图像数据集</div>';
    return;
  }
  grid.innerHTML = state.episode.cameras.map((camera, cameraIndex) => `
    <div class="camera-view">
      <img id="camera-${cameraIndex}" alt="${escapeHtml(CAMERA_LABELS[camera.name] || camera.label)} 第 1 帧">
      <span class="camera-label"><i data-lucide="camera"></i>${escapeHtml(CAMERA_LABELS[camera.name] || camera.label)}</span>
      <span class="camera-resolution">${camera.shape[2]}×${camera.shape[1]}</span>
      <span id="camera-time-${cameraIndex}" class="camera-time">--</span>
    </div>`).join("");
}

function frameImageUrl(camera, frame) {
  const params = new URLSearchParams({ path: state.episode.path, demo: state.episode.demo, camera: camera.name, index: frame });
  return `api/frame?${params}`;
}

function frameTime(index) {
  const timestamps = state.episode?.series.timestamps;
  if (timestamps?.length) return Number(timestamps[index] ?? 0) - Number(timestamps[0] ?? 0);
  if (state.episode?.duration && state.episode.frames > 1) return index * state.episode.duration / (state.episode.frames - 1);
  return index / 20;
}

function updateFrame(index, force = false, imageUrls = null) {
  const episode = state.episode;
  if (!episode?.frames) return;
  index = Math.min(Math.max(Math.round(index), 0), episode.frames - 1);
  if (!force && index === state.frame) return;
  state.frame = index;
  $("#timeline").value = index;
  $("#frame-input").value = index + 1;
  $("#frame-label").textContent = `${(index + 1).toLocaleString()} / ${episode.frames.toLocaleString()}`;
  $("#time-label").textContent = `${frameTime(index).toFixed(3)} s`;
  const phaseValue = Number(episode.series.trajectory_phase?.[index] ?? 0);
  const phaseLabel = $("#phase-label");
  phaseLabel.textContent = phaseValue === 1 ? "HOME" : "TASK";
  phaseLabel.className = `phase ${phaseValue === 1 ? "home" : "task"}`;

  const cameraTimes = episode.series.camera_timestamps?.[index] || [];
  episode.cameras.forEach((camera, cameraIndex) => {
    const image = $(`#camera-${cameraIndex}`);
    image.src = imageUrls?.[cameraIndex] || frameImageUrl(camera, index);
    image.alt = `${CAMERA_LABELS[camera.name] || camera.label} 第 ${index + 1} 帧`;
    $(`#camera-time-${cameraIndex}`).textContent = Number.isFinite(Number(cameraTimes[cameraIndex])) ? `${Number(cameraTimes[cameraIndex]).toFixed(3)} s` : `frame ${index + 1}`;
  });
  renderCharts();
  renderFrameDetails();
}

function stopPlayback() {
  state.playing = false;
  state.playbackToken += 1;
  cancelAnimationFrame(state.animationFrame);
  state.animationFrame = null;
  state.frameLoadPending = false;
  const button = $("#play-toggle");
  if (button) {
    button.innerHTML = '<i data-lucide="play"></i>';
    button.title = "播放";
    button.setAttribute("aria-label", "播放");
    icons();
  }
}

function playbackCycleDuration() {
  const frames = state.episode?.frames || 0;
  if (frames <= 1) return 0.05;
  const lastTime = frameTime(frames - 1);
  const lastInterval = Math.max(0.001, lastTime - frameTime(frames - 2));
  return lastTime + lastInterval;
}

function preloadFrameImages(index) {
  const urls = state.episode.cameras.map((camera) => frameImageUrl(camera, index));
  return Promise.all(urls.map((url) => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(url);
    image.onerror = () => reject(new Error(`图像帧加载失败: ${index + 1}`));
    image.src = url;
  })));
}

function frameAtMediaTime(mediaTime) {
  let low = 0;
  let high = Math.max(0, (state.episode?.frames || 1) - 1);
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (frameTime(middle) <= mediaTime) low = middle;
    else high = middle - 1;
  }
  return low;
}

function playbackTick(now, token) {
  if (!state.playing || !state.episode || token !== state.playbackToken) return;
  const cycleDuration = playbackCycleDuration();
  let mediaTime = state.playbackStartedMediaTime
    + (now - state.playbackStartedAt) * state.speed / 1000;
  if (mediaTime >= cycleDuration) {
    mediaTime %= cycleDuration;
    state.playbackStartedAt = now;
    state.playbackStartedMediaTime = mediaTime;
  }
  const target = frameAtMediaTime(mediaTime);
  if (target !== state.frame && !state.frameLoadPending) {
    state.frameLoadPending = true;
    preloadFrameImages(target)
      .then((imageUrls) => {
        if (state.playing && token === state.playbackToken) {
          updateFrame(target, true, imageUrls);
        }
      })
      .catch((error) => {
        if (state.playing && token === state.playbackToken) {
          stopPlayback();
          toast(error.message, true);
        }
      })
      .finally(() => {
        if (token === state.playbackToken) state.frameLoadPending = false;
      });
  }
  state.animationFrame = requestAnimationFrame((nextNow) => playbackTick(nextNow, token));
}

function togglePlayback() {
  if (!state.episode?.frames) return;
  if (state.playing) {
    stopPlayback();
    return;
  }
  state.playing = true;
  state.playbackToken += 1;
  const token = state.playbackToken;
  state.frameLoadPending = false;
  if (state.frame >= state.episode.frames - 1) updateFrame(0, true);
  state.playbackStartedAt = performance.now();
  state.playbackStartedMediaTime = frameTime(state.frame);
  const button = $("#play-toggle");
  button.innerHTML = '<i data-lucide="pause"></i>';
  button.title = "暂停";
  button.setAttribute("aria-label", button.title);
  icons();
  state.animationFrame = requestAnimationFrame((now) => playbackTick(now, token));
}

function sourceData() {
  const series = state.episode?.series || {};
  if (state.curveSource === "state") {
    const joints = series[`obs__${state.side}_joint_states`];
    const gripper = series[`obs__${state.side}_gripper_states`];
    if (joints) return { joints, gripper: gripper?.map((row) => row[0]) || [], unit: "rad", gripperUnit: "m" };
    const combined = series.states || [];
    const start = state.side === "left" ? 0 : 7;
    return { joints: combined.map((row) => row.slice(start, start + 6)), gripper: combined.map((row) => row[start + 6]), unit: "rad", gripperUnit: "m" };
  }
  const rows = series[state.curveSource] || [];
  const start = state.side === "left" ? 0 : 7;
  return {
    joints: rows.map((row) => row.slice(start, start + 6)),
    gripper: rows.map((row) => row[start + 6]),
    unit: "rad",
    gripperUnit: "m",
  };
}

function renderJointLegend() {
  $("#joint-legend").innerHTML = COLORS.map((color, index) => `
    <label class="joint-toggle"><input type="checkbox" data-joint="${index}" ${state.visibleJoints.has(index) ? "checked" : ""}><span class="swatch" style="background:${color}"></span>J${index + 1}</label>`).join("");
  $("#joint-legend").querySelectorAll("input").forEach((input) => input.addEventListener("change", () => {
    const joint = Number(input.dataset.joint);
    input.checked ? state.visibleJoints.add(joint) : state.visibleJoints.delete(joint);
    renderCharts();
  }));
}

function valueRange(rows, columns) {
  let min = Infinity, max = -Infinity;
  rows.forEach((row) => columns.forEach((column) => {
    const value = Number(Array.isArray(row) ? row[column] : row);
    if (Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); }
  }));
  if (!Number.isFinite(min)) return [-1, 1];
  if (min === max) { const padding = Math.max(Math.abs(min) * 0.1, 0.01); return [min - padding, max + padding]; }
  const padding = (max - min) * 0.08;
  return [min - padding, max + padding];
}

function drawChart(canvas, rows, columns, colors) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return [0, 0];
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const width = rect.width, height = rect.height;
  const pad = { left: 46, right: 10, top: 8, bottom: 22 };
  const plotWidth = Math.max(1, width - pad.left - pad.right);
  const plotHeight = Math.max(1, height - pad.top - pad.bottom);
  ctx.clearRect(0, 0, width, height);
  const [min, max] = valueRange(rows, columns);

  ctx.strokeStyle = "#e3e8e5";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#75817b";
  ctx.font = '9px "SFMono-Regular", Consolas, monospace';
  ctx.textAlign = "right";
  for (let step = 0; step <= 4; step++) {
    const y = pad.top + plotHeight * step / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillText((max - (max - min) * step / 4).toFixed(3), pad.left - 7, y + 3);
  }
  ctx.textAlign = "center";
  for (let step = 0; step <= 4; step++) {
    const x = pad.left + plotWidth * step / 4;
    const frame = Math.round(Math.max(0, rows.length - 1) * step / 4);
    ctx.fillText(String(frame + 1), x, height - 5);
  }

  const xAt = (index) => pad.left + plotWidth * index / Math.max(1, rows.length - 1);
  const yAt = (value) => pad.top + plotHeight * (max - value) / (max - min);
  columns.forEach((column, colorIndex) => {
    ctx.strokeStyle = colors[colorIndex];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    rows.forEach((row, index) => {
      const value = Number(Array.isArray(row) ? row[column] : row);
      if (!Number.isFinite(value)) { started = false; return; }
      const x = xAt(index), y = yAt(value);
      if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
    });
    ctx.stroke();
  });

  const cursorX = xAt(Math.min(state.frame, Math.max(0, rows.length - 1)));
  ctx.strokeStyle = "#17201c";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(cursorX, pad.top); ctx.lineTo(cursorX, pad.top + plotHeight); ctx.stroke();
  ctx.setLineDash([]);
  return [min, max];
}

function renderCharts() {
  if (!state.episode || $("#tab-curves").classList.contains("hidden")) return;
  const source = sourceData();
  const columns = [...state.visibleJoints].sort();
  const jointRange = drawChart($("#joint-chart"), source.joints, columns, columns.map((index) => COLORS[index]));
  const gripperRows = source.gripper.map((value) => [value]);
  const gripperRange = drawChart($("#gripper-chart"), gripperRows, [0], ["#aa3a35"]);
  $("#joint-unit").textContent = source.unit;
  $("#gripper-unit").textContent = source.gripperUnit;
  $("#joint-range").textContent = `${formatNumber(jointRange[0], 3)} ~ ${formatNumber(jointRange[1], 3)}`;
  $("#gripper-range").textContent = `${formatNumber(gripperRange[0], 4)} ~ ${formatNumber(gripperRange[1], 4)}`;
}

function sideValues(side) {
  const series = state.episode.series;
  const index = state.frame;
  const stateJoints = series[`obs__${side}_joint_states`]?.[index] || [];
  const stateGrip = series[`obs__${side}_gripper_states`]?.[index]?.[0];
  const pose = series[`obs__${side}_ee_states`]?.[index] || [];
  const force = series[`obs__${side}_gripper_force`]?.[index]?.[0];
  const action = series.actions?.[index] || [];
  const offset = side === "left" ? 0 : 7;
  return { stateJoints, stateGrip, pose, force, actionJoints: action.slice(offset, offset + 6), actionGrip: action[offset + 6] };
}

function renderFrameDetails() {
  if (!state.episode) return;
  $("#frame-details").innerHTML = ["left", "right"].map((side) => {
    const values = sideValues(side);
    const rows = Array.from({ length: 6 }, (_, index) => {
      const action = values.actionJoints[index];
      const feedback = values.stateJoints[index];
      const delta = Number(action) - Number(feedback);
      const deltaClass = Number.isFinite(delta) ? (delta >= 0 ? "delta-positive" : "delta-negative") : "";
      return `<tr><td>J${index + 1}</td><td>${formatNumber(action)}</td><td>${formatNumber(feedback)}</td><td class="${deltaClass}">${formatNumber(delta)}</td></tr>`;
    }).join("");
    const gripDelta = Number(values.actionGrip) - Number(values.stateGrip);
    const poseNames = ["X / m", "Y / m", "Z / m", "Roll / rad", "Pitch / rad", "Yaw / rad"];
    const pose = poseNames.map((name, index) => `<div class="pose-cell"><span>${name}</span><code>${formatNumber(values.pose[index])}</code></div>`).join("");
    return `<article class="arm-panel">
      <h3>${side === "left" ? "左臂" : "右臂"}</h3>
      <table class="value-table"><thead><tr><th>通道</th><th>动作</th><th>反馈</th><th>差值</th></tr></thead><tbody>${rows}<tr><td>夹爪 / m</td><td>${formatNumber(values.actionGrip)}</td><td>${formatNumber(values.stateGrip)}</td><td>${formatNumber(gripDelta)}</td></tr>${values.force !== undefined ? `<tr><td>夹爪力 / N</td><td>--</td><td>${formatNumber(values.force, 2)}</td><td>--</td></tr>` : ""}</tbody></table>
      ${values.pose.length ? `<div class="pose-block"><strong>末端位姿</strong><div class="pose-grid">${pose}</div></div>` : ""}
    </article>`;
  }).join("");
}

function schemaUnits(item) {
  const attrs = item.attrs || {};
  const keys = ["units", "unit", "joint_unit", "gripper_unit", "representation", "layout", "range"];
  return keys.filter((key) => attrs[key] !== undefined).map((key) => `<div><strong>${escapeHtml(key)}</strong>: ${escapeHtml(Array.isArray(attrs[key]) ? attrs[key].join(", ") : attrs[key])}</div>`).join("") || "--";
}

function renderSchema() {
  const query = $("#schema-search").value.trim().toLowerCase();
  const items = (state.episode?.schema || []).filter((item) => item.path.toLowerCase().includes(query));
  $("#dataset-count").textContent = `${items.length} datasets`;
  $("#schema-body").innerHTML = items.map((item) => `<tr><td>${escapeHtml(item.path)}</td><td>[${item.shape.join(", ")}]</td><td>${escapeHtml(item.dtype)}</td><td class="attr-list">${schemaUnits(item)}</td></tr>`).join("");
}

function switchTab(name) {
  $$(".tabs button").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== `tab-${name}`));
  if (name === "curves") requestAnimationFrame(renderCharts);
  if (name === "frame") renderFrameDetails();
}

function bindEvents() {
  $("#directory-form").addEventListener("submit", (event) => { event.preventDefault(); scanDirectory($("#directory-input").value.trim()); });
  $("#file-search").addEventListener("input", renderFileList);
  $("#schema-search").addEventListener("input", renderSchema);
  $("#sidebar-toggle").addEventListener("click", () => $(".workspace").classList.add("sidebar-collapsed"));
  $("#sidebar-open").addEventListener("click", () => $(".workspace").classList.remove("sidebar-collapsed"));
  $("#previous-frame").addEventListener("click", () => { stopPlayback(); updateFrame(state.frame - 1, true); });
  $("#next-frame").addEventListener("click", () => { stopPlayback(); updateFrame(state.frame + 1, true); });
  $("#play-toggle").addEventListener("click", togglePlayback);
  $("#timeline").addEventListener("input", (event) => { stopPlayback(); updateFrame(Number(event.target.value), true); });
  $("#frame-input").addEventListener("change", (event) => { stopPlayback(); updateFrame(Number(event.target.value) - 1, true); });
  $("#demo-select").addEventListener("change", (event) => selectFile(state.selectedPath, event.target.value));
  $("#copy-path").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(state.episode.path); toast("文件路径已复制"); }
    catch { toast("浏览器未允许访问剪贴板", true); }
  });
  $$("#speed-control button").forEach((button) => button.addEventListener("click", () => {
    if (state.playing) {
      const now = performance.now();
      const elapsed = (now - state.playbackStartedAt) * state.speed / 1000;
      state.playbackStartedMediaTime = (
        state.playbackStartedMediaTime + elapsed
      ) % playbackCycleDuration();
      state.playbackStartedAt = now;
    }
    state.speed = Number(button.dataset.speed);
    $$("#speed-control button").forEach((item) => item.classList.toggle("active", item === button));
  }));
  $$("#curve-source button").forEach((button) => button.addEventListener("click", () => {
    state.curveSource = button.dataset.source;
    $$("#curve-source button").forEach((item) => item.classList.toggle("active", item === button));
    renderCharts();
  }));
  $$("#curve-side button").forEach((button) => button.addEventListener("click", () => {
    state.side = button.dataset.side;
    $$("#curve-side button").forEach((item) => item.classList.toggle("active", item === button));
    renderCharts();
  }));
  $$(".tabs button").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  window.addEventListener("resize", () => requestAnimationFrame(renderCharts));
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (event.code === "Space") { event.preventDefault(); togglePlayback(); }
    if (event.code === "ArrowLeft") { event.preventDefault(); stopPlayback(); updateFrame(state.frame - 1, true); }
    if (event.code === "ArrowRight") { event.preventDefault(); stopPlayback(); updateFrame(state.frame + 1, true); }
  });
}

async function init() {
  icons();
  bindEvents();
  try {
    const config = await api("api/config");
    const directory = localStorage.getItem("piperViewerDirectory") || config.default_directory;
    $("#directory-input").value = directory;
    await scanDirectory(directory);
  } catch (error) {
    setScanStatus(error.message, true);
    toast(error.message, true);
  }
}

init();
