'use strict';
const $ = id => document.getElementById(id);
let state = null,
  online = false,
  sending = false,
  lastToken = null,
  configured = false;
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
  return values;
}

function render() {
  if (!state) return;
  const phase = phases[state.phase] || phases.idle;
  const can = online && !sending && !state.pending && !state.stopping;
  const action = state.accepts?.[0];
  $('phase-tag').textContent = state.stopping ? '正在结束会话' : phase[0];
  $('phase-title').textContent = phase[1];
  $('phase-help').textContent = state.error ? `${phase[2]} 上次异常：${state.error}` : phase[2];
  $('frames').textContent = state.frames || 0;
  $('pulse').classList.toggle('live', state.running);
  document.querySelectorAll('[data-step]').forEach(el => el.classList.toggle('active', el.dataset.step === phase[3]));
  $('config-fields').disabled = state.running;
  $('start').disabled = state.running || sending || !online;
  $('stop').disabled = !state.running || state.stopping || sending || !online;
  $('instruction').disabled = state.phase !== 'instruction' || !can;
  $('pedal').disabled = !can || !action;
  $('discard').hidden = state.phase !== 'review';
  $('discard').disabled = !can;
  $('pedal-label').textContent = state.pending ? '操作已发送，请等待' : state.phase === 'instruction' ? '开始本条采集' : state.phase === 'review' ? '保存这一条' : action === 'finish' ? '结束采集并回零' : '请等待当前阶段完成';
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
}
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
async function poll() {
  try {
    if (!configured) {
      const data = await api('bootstrap');
      state = data.status;
      const values = state.running ? state.config : data.defaults;
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
