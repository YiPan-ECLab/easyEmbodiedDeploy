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
  imageFrames: new Map(),
  imageLoads: 0,
  prefetchFrames: [],
  fusion: null,
  fusionPreview: false,
  fusionBusy: false,
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
  state.imageFrames.clear();
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
    state.fusion = null;
    state.fusionPreview = false;
    state.curveSource = "actions";
    $("#empty-state").classList.add("hidden");
    $("#episode-view").classList.remove("hidden");
    renderEpisode();
    await loadFusion(token);
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
    episode.demo_attrs.fps === "mixed" ? '<span class="metric"><strong>mixed</strong>采样频率</span>' : "",
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
  if (!$("#tab-frame").classList.contains("hidden")) renderFrameDetails();
}

function stopPlayback() {
  state.playing = false;
  state.playbackToken += 1;
  cancelAnimationFrame(state.animationFrame);
  state.animationFrame = null;
  state.frameLoadPending = false;
  state.prefetchFrames = [];
  const button = $("#play-toggle");
  if (button) {
    button.innerHTML = '<i data-lucide="play"></i>';
    button.title = "播放";
    button.setAttribute("aria-label", "播放");
    icons();
  }
}

function playbackCycleDuration() {
  if (state.fusionPreview) return state.fusion.indices.length / state.fusion.plan.playback_fps;
  const frames = state.episode?.frames || 0;
  if (frames <= 1) return 0.05;
  const lastTime = frameTime(frames - 1);
  const lastInterval = Math.max(0.001, lastTime - frameTime(frames - 2));
  return lastTime + lastInterval;
}

function preloadFrameImages(index) {
  const cached = state.imageFrames.get(index);
  if (cached) return cached.promise;
  const urls = state.episode.cameras.map((camera) => frameImageUrl(camera, index));
  const entry = { images: [], urls: null, promise: null };
  state.imageFrames.set(index, entry);
  state.imageLoads += 1;
  entry.promise = Promise.all(urls.map((url) => new Promise((resolve, reject) => {
    const image = new Image();
    entry.images.push(image); // Retain decoded images until they leave the window.
    image.onload = async () => {
      try { await image.decode(); resolve(url); }
      catch { reject(new Error(`图像帧解码失败: ${index + 1}`)); }
    };
    image.onerror = () => reject(new Error(`图像帧加载失败: ${index + 1}`));
    image.src = url;
  }))).then((loaded) => {
    entry.urls = loaded;
    return loaded;
  }).finally(() => {
    state.imageLoads -= 1;
    pumpFramePrefetch();
  });
  return entry.promise;
}

function pumpFramePrefetch() {
  if (!state.playing) return;
  for (const index of state.prefetchFrames) {
    if (state.imageLoads >= 2) break;
    if (!state.imageFrames.has(index)) {
      // Report errors only when the affected frame is actually requested.
      preloadFrameImages(index).catch(() => {});
    }
  }
}

function prefetchPlaybackFrames(index) {
  const indices = state.fusionPreview ? state.fusion.indices : null;
  const length = indices?.length || state.episode.frames;
  const position = indices ? Math.max(0, indices.indexOf(index)) : index;
  const bytesPerFrame = state.episode.cameras.reduce((sum, camera) =>
    sum + camera.shape[1] * camera.shape[2] * 4, 0);
  // At most 18 camera groups / about 32 MiB decoded. Two groups are prefetched;
  // after a seek, a foreground request may briefly coexist with old requests.
  const count = Math.min(length, 18, Math.max(2, Math.floor(32 * 1024 ** 2 / Math.max(1, bytesPerFrame))));
  state.prefetchFrames = Array.from({ length: count }, (_, offset) =>
    indices ? indices[(position + offset) % length] : (position + offset) % length);
  const wanted = new Set(state.prefetchFrames);
  for (const key of state.imageFrames.keys()) {
    if (!wanted.has(key)) state.imageFrames.delete(key);
  }
  pumpFramePrefetch();
}

function frameAtMediaTime(mediaTime) {
  if (state.fusionPreview) {
    const indices = state.fusion.indices;
    const position = Math.max(0, Math.min(indices.length - 1,
      Math.floor(mediaTime * state.fusion.plan.playback_fps)));
    return indices[position];
  }
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
  if (state.frameLoadPending) {
    state.animationFrame = requestAnimationFrame((nextNow) => playbackTick(nextNow, token));
    return;
  }
  const cycleDuration = playbackCycleDuration();
  // rAF timestamps refer to the start of a rendering frame. A click handler
  // can start playback later in that same frame, so the first delta is negative.
  let mediaTime = state.playbackStartedMediaTime
    + Math.max(0, now - state.playbackStartedAt) * state.speed / 1000;
  if (mediaTime >= cycleDuration) {
    mediaTime %= cycleDuration;
    state.playbackStartedAt = now;
    state.playbackStartedMediaTime = mediaTime;
  }
  const target = frameAtMediaTime(mediaTime);
  prefetchPlaybackFrames(target);
  const ready = state.imageFrames.get(target)?.urls;
  if (target !== state.frame && ready) {
    updateFrame(target, true, ready);
  } else if (target !== state.frame) {
    state.frameLoadPending = true;
    // Buffering must not consume media time and skip a stretch of the episode.
    state.playbackStartedMediaTime = mediaTime;
    preloadFrameImages(target)
      .then((imageUrls) => {
        if (state.playing && token === state.playbackToken) {
          updateFrame(target, true, imageUrls);
          state.playbackStartedAt = performance.now();
        }
      })
      .catch((error) => {
        if (state.playing && token === state.playbackToken) {
          state.imageFrames.delete(target);
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
  state.frameLoadPending = true;
  if (state.frame >= state.episode.frames - 1) updateFrame(0, true);
  state.playbackStartedAt = performance.now();
  state.playbackStartedMediaTime = state.fusionPreview
    ? Math.max(0, state.fusion.indices.findIndex((i) => i >= state.frame)) / state.fusion.plan.playback_fps
    : frameTime(state.frame);
  const button = $("#play-toggle");
  button.innerHTML = '<i data-lucide="pause"></i>';
  button.title = "暂停";
  button.setAttribute("aria-label", button.title);
  icons();
  const first = frameAtMediaTime(state.playbackStartedMediaTime);
  prefetchPlaybackFrames(first);
  preloadFrameImages(first).then((urls) => {
    if (!state.playing || token !== state.playbackToken) return;
    updateFrame(first, true, urls);
    state.playbackStartedAt = performance.now();
    state.frameLoadPending = false;
    state.animationFrame = requestAnimationFrame((now) => playbackTick(now, token));
  }).catch((error) => {
    if (!state.playing || token !== state.playbackToken) return;
    state.imageFrames.delete(first);
    stopPlayback();
    toast(error.message, true);
  });
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
  bindFusion();
  $("#delete-episode").addEventListener("click", deleteEpisode);
  $("#directory-form").addEventListener("submit", (event) => { event.preventDefault(); scanDirectory($("#directory-input").value.trim()); });
  $("#file-search").addEventListener("input", renderFileList);
  $("#schema-search").addEventListener("input", renderSchema);
  $("#sidebar-toggle").addEventListener("click", () => $(".workspace").classList.add("sidebar-collapsed"));
  $("#sidebar-open").addEventListener("click", () => $(".workspace").classList.remove("sidebar-collapsed"));
  $("#previous-frame").addEventListener("click", () => stepFrame(-1));
  $("#next-frame").addEventListener("click", () => stepFrame(1));
  $("#play-toggle").addEventListener("click", togglePlayback);
  $("#timeline").addEventListener("input", (event) => { stopPlayback(); updateFrame(Number(event.target.value), true); });
  $("#frame-input").addEventListener("change", (event) => { stopPlayback(); updateFrame(Number(event.target.value) - 1, true); });
  $("#demo-select").addEventListener("change", (event) => selectFile(state.selectedPath, event.target.value));
  $("#copy-path").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(state.episode.path); toast("文件路径已复制"); }
    catch { toast("浏览器未允许访问剪贴板", true); }
  });
  $$("#speed-control button").forEach((button) => button.addEventListener("click", () => {
    if (state.playing && !state.frameLoadPending) {
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
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (event.code === "Space") { event.preventDefault(); togglePlayback(); }
    if (event.code === "ArrowLeft") { event.preventDefault(); stepFrame(-1); }
    if (event.code === "ArrowRight") { event.preventDefault(); stepFrame(1); }
    if (event.code === "BracketLeft") $("#fusion-mark-start").click();
    if (event.code === "BracketRight") $("#fusion-mark-end").click();
  });
}

async function trajectoryRequest(action, payload) {
  const response = await fetch(`api/trajectory/${action}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "轨迹操作失败");
  return result;
}

async function deleteEpisode() {
  const episode = state.episode;
  if (!episode || state.fusionBusy) return;
  if (!window.confirm(`将整条轨迹移入回收站？\n${episode.path}\n可以撤销删除。`)) return;
  stopPlayback();
  $("#delete-episode").disabled = true;
  try {
    const result = await trajectoryRequest("delete", {path: episode.path, revision: episode.revision});
    ++state.loadToken;
    state.episode = null; state.selectedPath = ""; state.fusion = null;
    state.imageFrames.clear();
    $("#episode-view").classList.add("hidden");
    $("#empty-state").classList.remove("hidden");
    const notice = $("#delete-result");
    notice.classList.remove("hidden");
    notice.textContent = `${episode.name} 已移入回收站。 `;
    const undo = document.createElement("button");
    undo.textContent = "撤销删除";
    undo.onclick = async () => {
      undo.disabled = true;
      try {
        await trajectoryRequest("restore", {path: episode.path, id: result.id});
        notice.classList.add("hidden");
        await scanDirectory($("#directory-input").value.trim());
        await selectFile(episode.path);
      } catch (error) { toast(error.message, true); undo.disabled = false; }
    };
    notice.append(undo);
    await scanDirectory($("#directory-input").value.trim());
  } catch (error) { toast(error.message, true); }
  finally { $("#delete-episode").disabled = false; }
}

function stepFrame(direction) {
  stopPlayback();
  let target = state.frame + direction;
  if (state.fusionPreview) {
    const indices = state.fusion.indices;
    target = direction > 0 ? indices.find((i) => i > state.frame) : indices.findLast((i) => i < state.frame);
    target ??= state.frame;
  }
  updateFrame(target, true);
}

function fusionPlan() {
  return {base_stride: Number($("#fusion-base").value), playback_fps: Number($("#fusion-fps").value),
    segments: state.fusion?.plan.segments || []};
}

async function fusionRequest(action, plan = fusionPlan()) {
  const response = await fetch(`api/fusion/${action}`, {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({path: state.episode.path, demo: state.episode.demo, plan,
      output_directory: $("#fusion-directory").value.trim()})});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "融合请求失败");
  return result;
}

function renderFusion() {
  const fusion = state.fusion;
  if (!fusion) return;
  $("#fusion-base").value = fusion.plan.base_stride;
  $("#fusion-fps").value = fusion.plan.playback_fps;
  $("#fusion-start").max = $("#fusion-end").max = fusion.task_frames;
  $("#fusion-summary").textContent = `TASK ${fusion.task_frames} → ${fusion.frames} 帧，保留 ${(100*fusion.frames/fusion.task_frames).toFixed(1)}%；采样频率 mixed；标注自动保存`;
  $("#fusion-preview").textContent = state.fusionPreview ? "返回原始回放" : "预览融合";
  $("#fusion-preview").classList.toggle("active", state.fusionPreview);
  $("#fusion-segments").innerHTML = fusion.plan.segments.map((segment, i) =>
    `<div class="fusion-segment"><button data-seek="${segment.start}">第 ${segment.start+1}–${segment.end+1} 帧</button><span>×${segment.factor} · 实际步长 ${segment.factor*fusion.plan.base_stride}</span><button data-edit="${i}">编辑</button><button data-remove="${i}">删除</button></div>`).join("") || '<span class="fusion-help">未添加片段。拖动时间轴定位，使用 [ 和 ] 标记起止帧。</span>';
  $("#fusion-strip").innerHTML = fusion.plan.segments.map((segment) =>
    `<span title="第 ${segment.start+1}–${segment.end+1} 帧 ×${segment.factor}" style="left:${100*segment.start/fusion.task_frames}%;width:${100*(segment.end-segment.start+1)/fusion.task_frames}%"></span>`).join("");
  $$("#fusion-segments [data-seek]").forEach((button) => button.onclick = () => { stopPlayback(); updateFrame(Number(button.dataset.seek), true); });
  $$("#fusion-segments [data-remove]").forEach((button) => button.onclick = () => changeFusion((plan) => {
    plan.segments.splice(Number(button.dataset.remove), 1);
  }));
  $$("#fusion-segments [data-edit]").forEach((button) => button.onclick = () => {
    const i = Number(button.dataset.edit), segment = fusion.plan.segments[i];
    $("#fusion-start").value = segment.start + 1;
    $("#fusion-end").value = segment.end + 1;
    $("#fusion-factor").value = segment.factor;
    $("#fusion-add").dataset.edit = String(i);
    $("#fusion-add").textContent = "更新片段";
  });
}

function resetFusionEdit() {
  delete $("#fusion-add").dataset.edit;
  $("#fusion-add").textContent = "添加片段";
}

async function loadFusion(token) {
  $("#fusion-result").textContent = "";
  resetFusionEdit();
  try {
    const result = await api("api/fusion", {path: state.episode.path, demo: state.episode.demo});
    if (token !== state.loadToken) return;
    state.fusion = result;
    state.fusionPreview = state.episode.demo_attrs.fps === "mixed";
    $("#fusion-directory").value = result.output_directory;
    $("#fusion-start").value = 1;
    $("#fusion-end").value = Math.min(2, result.task_frames);
    renderFusion();
  } catch (error) {
    if (token === state.loadToken) $("#fusion-result").textContent = error.message;
  }
}

async function changeFusion(mutate = () => {}) {
  if (!state.fusion || state.fusionBusy) return;
  const token = state.loadToken;
  const plan = structuredClone(fusionPlan());
  mutate(plan);
  state.fusionBusy = true;
  stopPlayback();
  try {
    const result = await fusionRequest("save", plan);
    if (token !== state.loadToken) return;
    state.fusion = result;
    state.fusionPreview = false;
    resetFusionEdit();
    renderFusion();
    $("#fusion-result").textContent = "标注已保存；可随时重新打开继续编辑。";
  } catch (error) {
    if (token === state.loadToken) {
      $("#fusion-result").textContent = error.message;
      toast(error.message, true);
    }
  } finally { state.fusionBusy = false; }
}

function bindFusion() {
  $("#fusion-mark-start").onclick = () => { $("#fusion-start").value = state.frame + 1; };
  $("#fusion-mark-end").onclick = () => { $("#fusion-end").value = state.frame + 1; };
  $("#fusion-base").onchange = $("#fusion-fps").onchange = () => changeFusion();
  $("#fusion-add").onclick = () => changeFusion((plan) => {
    const segment = {start: Number($("#fusion-start").value)-1, end: Number($("#fusion-end").value)-1,
      factor: Number($("#fusion-factor").value)};
    const editing = $("#fusion-add").dataset.edit;
    if (editing !== undefined) plan.segments[Number(editing)] = segment;
    else plan.segments.push(segment);
  });
  $("#fusion-save").onclick = () => changeFusion();
  $("#fusion-preview").onclick = async () => {
    if (!state.fusion || state.fusionBusy) return;
    stopPlayback();
    const token = state.loadToken;
    try {
      if (!state.fusionPreview) {
        const result = await fusionRequest("preview");
        if (token !== state.loadToken) return;
        state.fusion = result;
      }
      state.fusionPreview = !state.fusionPreview;
      updateFrame(state.fusionPreview ? (state.fusion.indices.find((i) => i >= state.frame) ?? 0) : state.frame, true);
      renderFusion();
      $("#fusion-result").textContent = state.fusionPreview ? "正在预览融合序列：播放及前后步进只经过保留帧；时间轴仍显示源帧号，方便继续标注。" : "已返回原始回放。";
    } catch (error) { toast(error.message, true); }
  };
  $("#fusion-export").onclick = async () => {
    if (!state.fusion || state.fusionBusy) return;
    state.fusionBusy = true;
    const token = state.loadToken, button = $("#fusion-export");
    button.disabled = true;
    button.textContent = "正在导出…";
    $("#fusion-result").textContent = "正在同步导出所有图像和轨迹，请等待完成。";
    try {
      const result = await fusionRequest("export");
      if (token !== state.loadToken) return;
      $("#fusion-result").textContent = `已导出 ${result.frames} 帧：${result.path}。在数据转换页选择目录 ${result.output_directory}；目标 FPS 仅设置逻辑时间，不会再次抽帧。`;
      const open = document.createElement("button");
      open.textContent = "打开导出文件";
      open.onclick = () => selectFile(result.path);
      $("#fusion-result").append(document.createElement("br"), open);
    } catch (error) {
      if (token === state.loadToken) $("#fusion-result").textContent = error.message;
    } finally {
      state.fusionBusy = false;
      button.disabled = false;
      button.textContent = "导出融合 HDF5";
    }
  };
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
