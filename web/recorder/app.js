'use strict';
const $ = id => document.getElementById(id);
let state = null,
  online = false,
  sending = false,
  lastToken = null,
  configured = false,
  zeroInspection = null;
const zeroConfirmation = '我已确认J2位于物理零位';
let startPoint={mode:'zero',name:'默认零位',arms:{left:Array(7).fill(0),right:Array(7).fill(0)}},startPoints=[];
function renderStartPoint(){
  $('start-point-mode').value=startPoint.mode;
  $('start-point-name').value=startPoint.name;
  $('start-point-custom').hidden=startPoint.mode!=='custom';
  $('start-point-summary').textContent=startPoint.mode==='custom'?`采集起点 · ${startPoint.name}`:'采集起点 · 默认零位';
  $('start-point-values').textContent=['left','right'].map(side=>{
    const q=startPoint.arms[side];
    return `${side==='left'?'左主臂 + 左从臂':'右主臂 + 右从臂'}\n${q.slice(0,6).map((v,i)=>`J${i+1} ${(v*180/Math.PI).toFixed(3)}°`).join(' / ')}\n夹爪 ${(q[6]*1000).toFixed(3)} mm`;
  }).join('\n\n');
}
function renderStartSources(){
  const select=$('start-point-source');select.replaceChildren();
  startPoints.forEach((p,i)=>select.add(new Option(p.name,String(i))));
}
const phases = {
  idle: ['准备就绪', '从连接设备开始', '连接并回零后，可输入指令开始采集。', 'home'],
  initializing: ['正在连接', '正在初始化设备', '正在连接三台相机与四台机械臂，请等待。', 'home'],
  homing: ['正在回零', '机械臂与夹爪归位中', '回零完成后自动进入任务指令。', 'home'],
  instruction: ['等待指令', '踩一下，开始这一条', '输入新指令，或留空并踩踏板沿用上一条。', 'instruction'],
  preposition: ['预定位', '准备进入示教', '左右 J6 正在到达 −45° / +45°，随后启动跟随。', 'capture'],
  waiting_motion: ['等待动作', '可以开始操作了', '静止帧会被丢弃；检测到真人动作后正式记录。踩踏板结束。', 'capture'],
  recording: ['正在采集', '专注示教，踩踏板结束', '回车结束 task 阶段，随后继续采样机械臂回零过程。', 'capture'],
  homing_review: ['采集结束', '回零过程仍在记录', '请等待回零完成，随后可保存或丢弃本条。', 'review'],
  review: ['待保存', '这一条，要保存吗？', '踩踏板 / 回车保存；点击“丢弃本条”放弃本次采集。', 'review'],
  saving: ['正在保存', '数据正在写入 HDF5', '完成后自动进入下一条任务指令，请等待。', 'review'],
  saved: ['保存完成', '这一条已保存', '即将进入下一条任务。', 'review'],
  exited: ['会话结束', '设备已断开', '可以修改设置后重新连接。', 'home'],
  failed: ['会话异常', '请检查运行日志', '处理设备或配置问题后，可重新启动会话。', 'home']
};

function error(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}
async function api(path, body) {
  const response = await fetch(`api/${path}`, body === undefined ? {} : {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `请求失败 ${response.status}`);
  return result;
}

function config() {
  const values = Object.fromEntries(new FormData($('config')));
  for (const key of Object.keys(values))
    if (key !== 'output') values[key] = Number(values[key]);
  values.start_point=structuredClone(startPoint);
  return values;
}

function renderZeroInspection() {
  const output = $('zero-result');
  if (!zeroInspection) {
    output.textContent = state?.maintenance ? '零点维护正在运行…' : '尚未检查';
    return;
  }
  const before = zeroInspection.before;
  if (!before) {
    output.textContent = zeroInspection.error || '没有收到诊断反馈';
    return;
  }
  const side = zeroInspection.side === 'left' ? '左主臂' : '右主臂';
  const j2 = before.drivers[1];
  const motor = before.motors[1];
  const lines = [
    `${side} (${zeroInspection.arm})`,
    `J2: ${before.joints_deg[1].toFixed(3)}° · 状态码 0x${j2.code.toString(16).toUpperCase().padStart(2, '0')} · ${j2.enabled ? '已使能' : '未使能'}`,
    `电流 ${Number(motor.current_a).toFixed(3)} A · 力矩 ${Number(motor.torque_nm).toFixed(3)} N·m`,
    zeroInspection.eligible ? '匹配已验证故障，可在人工确认物理零位后恢复。' : `拒绝自动恢复：${zeroInspection.reasons.join('；')}`
  ];
  if (zeroInspection.repaired && zeroInspection.after) {
    const after = zeroInspection.after;
    lines.push(`恢复完成：J2 ${after.joints_deg[1].toFixed(3)}° · 状态码 0x${after.drivers[1].code.toString(16).toUpperCase().padStart(2, '0')}`);
  }
  output.textContent = lines.join('\n');
}

function render() {
  if (!state) return;
  const phase = [...(phases[state.phase] || phases.idle)];
  const custom=(state.running?state.config?.start_point:startPoint)?.mode==='custom';
  if(custom){
    for(let i=0;i<3;i++)phase[i]=phase[i].replaceAll('回零','返回起点');
    if(state.phase==='preposition')phase[2]='四台机械臂与夹爪返回自定义起点，对齐后启动跟随，不执行 J6 预摆。';
    if(state.phase==='instruction')phase[2]='四臂保持在起点。填写任务指令并踩踏板后，主臂才进入拖动跟随。';
    if(state.phase==='review')phase[2]='四臂已返回并保持起点。踩踏板保存，或点击“丢弃本条”。';
  }
  const can = online && !sending && !state.pending && !state.stopping;
  const action = state.accepts?.[0];
  $('phase-tag').textContent = state.stopping ? '正在结束会话' : phase[0];
  $('phase-title').textContent = phase[1];
  $('phase-help').textContent = state.error ? `${phase[2]} 上次异常：${state.error}` : phase[2];
  $('frames').textContent = state.frames || 0;
  $('pulse').classList.toggle('live', state.running);
  document.querySelectorAll('[data-step]').forEach(el => el.classList.toggle('active', el.dataset.step === phase[3]));
  $('config-fields').disabled = state.running || state.maintenance;
  $('start').disabled = state.running || state.maintenance || sending || !online;
  $('start').textContent=custom?'连接设备并返回起点':'连接设备并回零';
  $('check').disabled = state.running || state.maintenance || sending || !online;
  $('stop').disabled = !state.running || state.stopping || sending || !online;
  $('instruction').disabled = state.phase !== 'instruction' || !can;
  $('pedal').disabled = !can || !action;
  $('discard').hidden = state.phase !== 'review';
  $('discard').disabled = !can;
  document.querySelectorAll('.zero-inspect').forEach(button => {
    button.disabled = state.running || state.maintenance || sending || !online;
  });
  $('zero-confirm').disabled = state.running || state.maintenance || sending || !zeroInspection?.eligible;
  $('zero-repair').disabled = state.running || state.maintenance || sending || !online || !zeroInspection?.eligible || !$('zero-confirm').checked;
  $('pedal-label').textContent = state.pending ? '操作已发送，请等待' : state.phase === 'instruction' ? '开始本条采集' : state.phase === 'review' ? '保存这一条' : action === 'finish' ? (custom?'结束采集并返回起点':'结束采集并回零') : '请等待当前阶段完成';
  $('connection').textContent = online ? '控制台已连接' : '连接中断，正在重连';
  $('saved-count').textContent = `${state.saved.length} 条`;
  $('saved').replaceChildren();
  if (!state.saved.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = '保存后的文件会显示在这里';
    $('saved').append(li);
  }
  for (const item of [...state.saved].reverse()) {
    const li = document.createElement('li');
    li.textContent = `${item.path} · ${item.frames} 帧`;
    $('saved').append(li);
  }
  const logs = state.logs.join('\n');
  if ($('logs').textContent !== logs) {
    const bottom = $('logs').scrollHeight - $('logs').scrollTop - $('logs').clientHeight < 50;
    $('logs').textContent = logs || '等待会话开始…';
    if (bottom) $('logs').scrollTop = $('logs').scrollHeight;
  }
  const token = `${state.session_id}:${state.token}`;
  if (token !== lastToken) {
    lastToken = token;
    if (state.phase === 'instruction') {
      $('instruction').value = '';
      if (document.hasFocus()) $('instruction').focus({
        preventScroll: true
      });
    }
  }
  renderZeroInspection();
  updateCameraPreview();
}

// One request per camera at a time; no image data enters the control channel.
let previewVisible = false;
const cameraPreviews = [...document.querySelectorAll('[data-camera]')].map(tile => ({
  tile, name: tile.dataset.camera, image: tile.querySelector('img'),
  message: tile.querySelector('.camera-placeholder'), label: tile.querySelector('.camera-status'),
  controller: null, url: null, generation: null, lastGood: 0
}));

function previewUnavailable(camera, message) {
  camera.image.hidden = true;
  camera.message.hidden = false;
  camera.message.textContent = message;
  camera.label.textContent = message;
  camera.tile.classList.remove('live');
  if (camera.url) URL.revokeObjectURL(camera.url);
  camera.url = null;
  camera.lastGood = 0;
}

function updateCameraPreview() {
  const enabled = online && state?.running && state.camera_preview;
  const visible = previewVisible && !document.hidden;
  let message = !online ? '等待控制台连接' : !state?.running ? '连接设备后显示画面'
    : state.camera_preview ? (visible ? '等待相机画面' : '预览暂停')
    : state.camera_preview === undefined ? '更新预览需重启采集模块'
    : state.phase === 'initializing' ? '相机正在连接' : '预览不可用，请查看运行日志';
  for (const camera of cameraPreviews) {
    if (!enabled || !visible || camera.generation !== state.session_id) {
      camera.controller?.abort();
      previewUnavailable(camera, message);
      camera.generation = state?.session_id;
    } else if (camera.lastGood && performance.now() - camera.lastGood > 2000) {
      previewUnavailable(camera, '画面中断，正在重连');
    }
  }
  const count = cameraPreviews.filter(camera => camera.lastGood).length;
  $('camera-preview-status').textContent = enabled && visible ? `实时预览 · ${count} / 3 路` : message;
}

async function refreshCamera(camera) {
  let nextUrl = null;
  if (online && state?.running && state.camera_preview && previewVisible && !document.hidden) {
    const generation = state.session_id;
    const controller = new AbortController();
    camera.controller = controller;
    const timer = setTimeout(() => controller.abort(), 2000);
    try {
      const params = new URLSearchParams({camera: camera.name, session_id: generation});
      const response = await fetch(`api/camera-frame?${params}`, {signal: controller.signal, cache: 'no-store'});
      if (!response.ok) throw new Error('相机画面暂不可用');
      nextUrl = URL.createObjectURL(await response.blob());
      const decoded = new Image();
      decoded.src = nextUrl;
      await decoded.decode();
      if (controller.signal.aborted || state.session_id !== generation || !state.running)
        throw new Error('预览请求已过期');
      const previous = camera.url;
      camera.image.src = nextUrl;
      camera.url = nextUrl;
      nextUrl = null;
      if (previous) URL.revokeObjectURL(previous);
      camera.image.hidden = false;
      camera.message.hidden = true;
      camera.label.textContent = '实时画面';
      camera.tile.classList.add('live');
      camera.lastGood = performance.now();
    } catch {
      if (!controller.signal.aborted && state?.session_id === generation)
        previewUnavailable(camera, '画面中断，正在重连');
    } finally {
      clearTimeout(timer);
      if (nextUrl) URL.revokeObjectURL(nextUrl);
      camera.controller = null;
      updateCameraPreview();
    }
  }
  setTimeout(() => refreshCamera(camera), 100);
}

new IntersectionObserver(entries => {
  previewVisible = entries[0].isIntersecting;
  updateCameraPreview();
}).observe(document.querySelector('.camera-panel'));
document.addEventListener('visibilitychange', updateCameraPreview);
cameraPreviews.forEach(camera => refreshCamera(camera));
async function perform(fn) {
  if (sending) return;
  sending = true;
  error('');
  render();
  try {
    await fn();
    state = await api('status');
  } catch (exc) {
    error(exc.message);
  } finally {
    sending = false;
    render();
  }
}

function pedal(value) {
  if (!state || $('pedal').disabled) return;
  const action = state.accepts[0];
  const text = value === undefined ? (state.phase === 'instruction' ? $('instruction').value : '') : value;
  return perform(() => api('input', {
    session_id: state.session_id,
    token: state.token,
    action,
    value: text
  }));
}
$('config').addEventListener('submit', event => {
  event.preventDefault();
  if (!state?.running) perform(() => api('start', config()));
});
$('pedal').onclick = () => pedal();
$('discard').onclick = () => pedal('n');
$('stop').onclick = () => {
  if (confirm('结束会话并归位？当前未保存的采集将丢弃。')) perform(() => api('stop', {}));
};
$('check').onclick = () => perform(async () => {
  const result = await api('check', {});
  error(result.output);
});
document.querySelectorAll('.zero-inspect').forEach(button => {
  button.onclick = () => perform(async () => {
    $('zero-confirm').checked = false;
    zeroInspection = await api('joint-zero/inspect', {side: button.dataset.side});
    $('zero-panel').open = true;
  });
});
$('zero-confirm').onchange = render;
$('zero-repair').onclick = () => {
  if (!zeroInspection?.eligible || !$('zero-confirm').checked) return;
  const side = zeroInspection.side === 'left' ? '左主臂' : '右主臂';
  if (!confirm(`最后确认：把${side} J2 当前物理位置写成永久零点？\n\n错误姿态标零会造成碰撞。`)) return;
  perform(async () => {
    zeroInspection = await api('joint-zero/repair', {
      side: zeroInspection.side,
      confirmation: zeroConfirmation
    });
    $('zero-confirm').checked = false;
  });
};
$('copy-log').onclick = () => perform(async () => {
  await navigator.clipboard.writeText($('logs').textContent);
});
document.addEventListener('keydown', event => {
  if (event.key !== 'Enter' || event.isComposing || event.ctrlKey || event.altKey || event.metaKey) return;
  if (!state?.running) return;
  if (event.repeat) {
    event.preventDefault();
    return;
  }
  const target = event.target;
  if (target.closest('#config') || target.isContentEditable) return;
  event.preventDefault();
  if (target.closest('#discard')) pedal('n');
  else if (target.closest('#stop')) $('stop').click();
  else pedal();
});

function focus() {
  $('pedal-status').textContent = document.hasFocus() ? '脚踏板已就绪 · 回车跟随当前阶段' : '点击本页后，脚踏板即可操作';
}
window.addEventListener('focus', focus);
window.addEventListener('blur', focus);
$('start-point-mode').onchange=()=>{
  startPoint.mode=$('start-point-mode').value;
  if(startPoint.mode==='zero')startPoint={mode:'zero',name:'默认零位',arms:{left:Array(7).fill(0),right:Array(7).fill(0)}};
  else if(startPoint.name==='默认零位')startPoint.name='自定义起点';
  renderStartPoint();render();
};
$('start-point-name').onchange=()=>{startPoint.name=$('start-point-name').value;};
$('start-point-refresh').onclick=()=>perform(async()=>{startPoints=(await api('start-points')).points;renderStartSources();});
$('start-point-import').onclick=()=>perform(async()=>{
  const point=startPoints[Number($('start-point-source').value)];if(!point)throw new Error('没有可导入的关节记录');
  startPoint={mode:'custom',name:point.name,arms:{...startPoint.arms,...structuredClone(point.arms)}};
  renderStartPoint();$('start-point-message').textContent='已导入，核对左右侧角度后点击保存起点。';
});
for(const side of ['left','right'])$('start-point-'+side).onclick=()=>perform(async()=>{
  const result=await api('start-point/capture',{side});
  startPoint={mode:'custom',name:startPoint.mode==='custom'?startPoint.name:'手动录制起点',arms:{...startPoint.arms,...result.arms}};
  renderStartPoint();$('start-point-message').textContent='已读取，核对角度后点击保存起点；采集前退出机身示教。';
});
$('start-point-save').onclick=()=>perform(async()=>{
  startPoint=await api('start-point/save',{start_point:startPoint});renderStartPoint();
  $('start-point-message').textContent=startPoint.mode==='custom'?'起点已保存；每次开始和结束采集，四臂与夹爪都会返回此处。':'已恢复默认零位；保留开始采集前的 J6 ±45° 预摆。';
});
async function poll() {
  try {
    if (!configured) {
      const data = await api('bootstrap');
      state = data.status;
      const values = state.running ? state.config : data.defaults;
      startPoint=values.start_point||startPoint;startPoints=data.start_points||[];
      renderStartPoint();renderStartSources();
      for (const [key, value] of Object.entries(values))
        if ($('config').elements[key]) $('config').elements[key].value = value;
      configured = true;
    } else state = await api('status');
    online = true;
  } catch (exc) {
    online = false;
    error(`无法连接采集服务：${exc.message}。设备状态未知，请等待重连。`);
  }
  render();
  focus();
  setTimeout(poll, 450);
}
poll();
