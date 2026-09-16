'use strict';
const $ = id => document.getElementById(id);
const icons = {
  recorder: '◎',
  viewer: '▣',
  processor: '⇄',
  analysis: '⌁',
  'lerobot-trainer': '▤',
  'openpi-trainer': '◈',
  'lerobot-deployer': '↗',
  'openpi-deployer': '⌘'
};
const stateLabels = {
  ready: '服务已就绪',
  starting: '正在启动',
  stopped: '点击打开',
  failed: '需要检查',
  stopping: '正在清理'
};
let modules = [],
  selected = 'overview',
  created = false,
  opening = new Set(),
  framePids = new Map();
async function api(path, body) {
  const response = await fetch(`/api/${path}`, body === undefined ? {} : {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 ${response.status}`);
  return data;
}

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  setTimeout(() => $('toast').hidden = true, 8000);
}

function make(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
}

function build() {
  let group = '';
  for (const module of modules) {
    if (group !== module.group) {
      group = module.group;
      $('navigation').append(make('div', 'nav-label', group));
    }
    const nav = make('button', 'nav-item');
    nav.dataset.module = module.id;
    nav.title = module.title;
    nav.append(make('span', 'nav-icon', icons[module.id]), document.createTextNode(module.title), make('span', 'service-dot'));
    $('navigation').append(nav);
    const card = make('button', 'module-card');
    card.dataset.module = module.id;
    const top = make('div', 'card-top');
    top.append(make('span', 'card-icon', icons[module.id]), make('span', 'category', module.group));
    const bottom = make('div', 'card-bottom');
    const status = make('span', 'card-status');
    status.append(make('i', 'status-dot'), make('span', 'status-label'));
    bottom.append(status, make('span', 'arrow', '↗'));
    card.append(top, make('h3', '', module.title), make('p', '', module.description), bottom);
    $('cards').append(card);
  }
  created = true;
}

function render() {
  if (!created) build();
  let running = 0;
  for (const module of modules) {
    if (module.state === 'ready') running++;
    const card = document.querySelector(`.module-card[data-module="${module.id}"]`);
    card.querySelector('.status-label').textContent = module.available ? stateLabels[module.state] : '运行环境待创建';
    card.querySelector('.status-dot').className = `status-dot ${module.state==='ready'?'on':module.state==='failed'?'error':''}`;
    document.querySelector(`.nav-item[data-module="${module.id}"] .service-dot`).classList.toggle('on', module.state === 'ready');
  }
  $('running-count').replaceChildren(document.createTextNode(`${running} `), make('small', '', '个已启动'));
  if (selected === 'overview') return;
  const current = modules.find(m => m.id === selected);
  if (!current) return;
  $('logs').textContent = current.logs.join('\n') || '服务尚未启动';
  const ready = current.state === 'ready';
  $('loading').hidden = ready;
  $('retry').hidden = !['stopped', 'failed'].includes(current.state);
  $('loading-title').textContent = current.state === 'failed' ? '模块启动失败' : current.state === 'stopping' ? '正在结束模块任务' : '正在启动模块';
  $('loading-detail').textContent = current.error || '首次加载需要几秒钟，请稍候。';
  $('open-page').hidden = !ready;
  $('reload-page').hidden = !ready;
  let frame = $(`frame-${selected}`);
  if (ready && (!frame || framePids.get(selected) !== current.pid)) {
    if (frame) frame.remove();
    frame = make('iframe');
    frame.id = `frame-${selected}`;
    frame.title = current.title;
    frame.src = current.path;
    frame.addEventListener('load', () => {
      if (selected === current.id) frame.contentWindow.focus();
    });
    $('frames').append(frame);
    framePids.set(selected, current.pid);
  }
  for (const child of $('frames').children) child.hidden = child.id !== `frame-${selected}`;
}
async function openModule(id) {
  selected = id;
  $('overview').hidden = id !== 'overview';
  $('module-view').hidden = id === 'overview';
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('selected', el.dataset.module === id));
  const current = modules.find(m => m.id === id);
  $('page-title').textContent = current?.title || '概览';
  $('service-log').hidden = !current;
  $('open-page').hidden = true;
  $('reload-page').hidden = true;
  if (!current) return;
  $('open-page').href = current.path;
  render();
  if (['stopped', 'failed'].includes(current.state) && !opening.has(id)) {
    opening.add(id);
    try {
      await api('services/start', {
        module: id
      });
      await refresh();
    } catch (exc) {
      toast(exc.message);
      $('loading-detail').textContent = exc.message;
      $('retry').hidden = false;
    } finally {
      opening.delete(id);
    }
  }
}
document.addEventListener('click', event => {
  const button = event.target.closest('[data-module]');
  if (button) location.hash = button.dataset.module;
});
window.addEventListener('hashchange', () => openModule(location.hash.slice(1) || 'overview'));
$('quick-record').onclick = () => location.hash = 'recorder';
$('retry').onclick = () => openModule(selected);
$('reload-page').onclick = () => {
  const frame = $(`frame-${selected}`);
  if (frame) frame.src = frame.src;
};
$('service-log').onclick = () => $('logs-dialog').showModal();
$('close-logs').onclick = () => $('logs-dialog').close();
$('stop-service').onclick = async () => {
  if (!confirm('停止这个模块服务及其启动的任务？')) return;
  try {
    await api('services/stop', {
      module: selected
    });
    $('logs-dialog').close();
    await refresh();
  } catch (exc) {
    toast(exc.message);
  }
};
async function refresh() {
  const data = await api('status');
  modules = data.modules;
  $('connection').textContent = '本地工作台已连接';
  $('connection-dot').style.background = '';
  render();
}
async function poll() {
    try {
      await refresh();
    } catch (exc) {
      $('connection').textContent = '连接中断，正在重连';
      $('connection-dot').style.background = '#c29a67';
    }
    setTimeout(poll, 1500);
  }
  (async () => {
    try {
      await refresh();
      await openModule(location.hash.slice(1) || 'overview');
    } catch (exc) {
      toast(exc.message);
    }
    poll();
  })();
