const $ = id => document.getElementById(id);
let root = 'models',
  path = '',
  listing = null,
  busy = false;
const status = (message, error = false) => {
  $('status').textContent = message;
  $('status').className = error ? 'error' : '';
};
async function api(route, body) {
  const response = await fetch(`api/${route}`, body ? {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  } : {});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}

function disableActions(value) {
  document.querySelectorAll('header button, section button').forEach(b => b.disabled = value);
  if (!value) $('up').disabled = !path;
}
async function run(action) {
  if (busy) return;
  busy = true;
  disableActions(true);
  try {
    await action();
  } catch (e) {
    status(e.message, true);
  } finally {
    busy = false;
    disableActions(false);
  }
}

function size(bytes) {
  if (bytes == null) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let n = 0;
  while (bytes >= 1024 && n < 4) {
    bytes /= 1024;
    n++;
  }
  return `${bytes.toFixed(n?1:0)} ${units[n]}`;
}
async function copy(value) {
  try {
    let copied = false;
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(value);
        copied = true;
      } catch {}
    }
    if (!copied) {
      const area = document.createElement('textarea');
      area.value = value;
      document.body.append(area);
      area.select();
      const ok = document.execCommand('copy');
      area.remove();
      if (!ok) throw new Error('请从路径栏手动复制');
    }
    status(`已复制：${value}`);
  } catch (e) {
    status(e.message, true);
  }
}

function button(label, action, className = '') {
  const b = document.createElement('button');
  b.textContent = label;
  b.className = className;
  b.disabled = busy;
  b.onclick = () => run(action);
  return b;
}

function cell(row, value) {
  const td = document.createElement('td');
  if (value instanceof Node) td.append(value);
  else td.textContent = value;
  row.append(td);
  return td;
}

function promptAction(title, description, initial, needsInput = true) {
  return new Promise(resolve => {
    const dialog = $('dialog');
    $('dialog-title').textContent = title;
    $('dialog-description').textContent = description;
    $('dialog-input').value = initial || '';
    $('dialog-input').hidden = !needsInput;
    const finish = value => {
      dialog.close();
      dialog.oncancel = null;
      $('dialog-form').onsubmit = null;
      resolve(value);
    };
    $('cancel').onclick = () => finish(null);
    dialog.oncancel = e => {
      e.preventDefault();
      finish(null);
    };
    $('dialog-form').onsubmit = e => {
      e.preventDefault();
      finish(needsInput ? $('dialog-input').value : true);
    };
    dialog.showModal();
    if (needsInput) $('dialog-input').select();
  });
}

function render() {
  const rows = listing.items.filter(item => item.name.toLowerCase().includes($('filter').value.toLowerCase()));
  $('items').replaceChildren();
  $('empty').hidden = rows.length > 0;
  for (const item of rows) {
    const row = document.createElement('tr');
    const name = button(`${item.kind==='directory'?'▣':item.kind==='link'?'↗':'▤'}  ${item.name}`, async () => {
      if (item.kind === 'directory') await refresh(item.path);
      else await copy(item.absolute);
    }, 'name');
    cell(row, name);
    cell(row, size(item.size));
    cell(row, new Date(item.modified * 1000).toLocaleString());
    const actions = cell(row, '');
    actions.className = 'row';
    actions.append(button('复制路径', () => copy(item.absolute)), button('重命名', async () => {
      const name = await promptAction('重命名', item.relative, item.name);
      if (name === null || name === item.name) return;
      await api('rename', {
        root,
        path: item.path,
        name
      });
      await refresh();
      status('已重命名');
    }), button('删除', async () => {
      if (await promptAction('移入回收站', `将 ${item.relative} 移入回收站？`, null, false) === null) return;
      await api('delete', {
        root,
        path: item.path
      });
      await refresh();
      status('已移入回收站，可恢复');
    }, 'danger'));
    $('items').append(row);
  }
}
async function refresh(next = path) {
  const data = await api(`list?root=${encodeURIComponent(root)}&path=${encodeURIComponent(next)}`);
  listing = data;
  path = data.path === '.' ? '' : data.path;
  $('current').textContent = data.absolute;
  $('space').textContent = `磁盘可用 ${size(data.free_bytes)}`;
  $('up').disabled = !path;
  render();
  if (!$('trash-panel').hidden) await loadTrash();
}
async function loadTrash() {
  const {
    items
  } = await api(`trash?root=${root}`);
  $('trash-items').replaceChildren();
  for (const item of items) {
    const row = document.createElement('tr');
    cell(row, item.path);
    cell(row, new Date(item.deleted_at * 1000).toLocaleString());
    const actions = cell(row, '');
    actions.className = 'row';
    actions.append(button('恢复', async () => {
      await api('restore', {
        root,
        id: item.id
      });
      await refresh();
      status('已恢复到原位置');
    }), button('永久删除', async () => {
      const confirmation = await promptAction('永久删除', `无法恢复。请输入完整名称「${item.name}」确认。`, '');
      if (confirmation === null) return;
      await api('purge', {
        root,
        id: item.id,
        confirmation
      });
      await refresh();
      status('已永久删除');
    }, 'danger'));
    $('trash-items').append(row);
  }
  if (!items.length) {
    const row = document.createElement('tr');
    const td = cell(row, '回收站为空');
    td.colSpan = 3;
    td.className = 'empty';
    $('trash-items').append(row);
  }
}
document.querySelectorAll('[data-root]').forEach(b => b.onclick = () => run(async () => {
  root = b.dataset.root;
  path = '';
  $('filter').value = '';
  await refresh();
  document.querySelectorAll('[data-root]').forEach(item => item.classList.toggle('selected', item === b));
  status('');
}));
$('refresh').onclick = () => run(() => refresh());
$('up').onclick = () => run(() => refresh(path.split('/').slice(0, -1).join('/')));
$('home').onclick = () => run(() => refresh(''));
$('copy-current').onclick = () => listing && copy(listing.absolute);
$('copy-relative').onclick = () => listing && copy(listing.relative);
$('filter').oninput = () => listing && render();
$('mkdir').onclick = () => run(async () => {
  const name = await promptAction('新建文件夹', '在当前目录创建，不覆盖现有项目。', '');
  if (name === null) return;
  await api('mkdir', {
    root,
    path,
    name
  });
  await refresh();
  status('文件夹已创建');
});
$('trash-toggle').onclick = () => run(async () => {
  $('trash-panel').hidden = !$('trash-panel').hidden;
  if (!$('trash-panel').hidden) await loadTrash();
});
run(() => refresh());
