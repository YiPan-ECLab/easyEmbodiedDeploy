const $ = id => document.getElementById(id);
const titles = {
  left_wrist: '左腕外参',
  right_wrist: '右腕外参',
  global_left: '全局 → 左 base',
  global_right: '全局 → 右 base',
  intrinsics: '相机内参'
};
let defaults, sessions = [],
  session = null,
  rig = {
    open: false
  },
  busy = false,
  previewBusy = false,
  exportPath = '', dualSelection = [];
const purposeNames = {calibration:'求解样本', validation:'独立验证', dual_validation:'双臂观测'};
const status = (message, error = false) => {
  $('status').textContent = message;
  $('status').className = error ? 'error' : '';
};
async function api(route, body) {
  const response = await fetch(`api/${route}`, body === undefined ? {} : {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}
async function run(action) {
  if (busy) return;
  busy = true;
  document.querySelectorAll('button').forEach(b => b.disabled = true);
  try {
    await action();
  } catch (e) {
    status(e.message, true);
  } finally {
    busy = false;
    document.querySelectorAll('button').forEach(b => b.disabled = false);
    updateRig();
  }
}

function board() {
  return {
    kind: $('board-kind').value,
    columns: Number($('columns').value),
    rows: Number($('rows').value),
    square_mm: Number($('square-mm').value),
    marker_ratio: Number($('marker-ratio').value),
    dictionary: $('dictionary').value
  };
}

function updateBoard() {
  const b = board(),
    chess = b.kind === 'chessboard';
  $('cols-label').textContent = chess ? '内角点列数' : '格子列数';
  $('rows-label').textContent = chess ? '内角点行数' : '格子行数';
  $('charuco-options').hidden = chess;
  const c = b.columns + Number(chess),
    r = b.rows + Number(chess);
  $('board-size').textContent = `${c}×${r} 个格子 · 板面 ${(c*b.square_mm).toFixed(1)}×${(r*b.square_mm).toFixed(1)} mm（打印另加白边）`;
  $('board-download').href = `api/board.svg?${new URLSearchParams(b)}`;
}

function procedure(job) {
  return job === 'intrinsics' ? '保持相机设置固定，移动标定板，覆盖不同位置、距离与倾角。' : job.endsWith('wrist') ? '腕部相机固定在对应法兰；标定板固定在环境中。改变机械臂姿态，每次静止后采样，整个记录中不要移动标定板。' : '全局相机固定不动；标定板刚性固定在对应法兰上，随该机械臂改变姿态。分别完成左右 base 两份记录，中间不得移动全局相机或 base。';
}

function changeJob() {
  const job = defaults.jobs[$('job').value];
  if (job.camera) $('camera').value = job.camera;
  $('camera').disabled = !!job.camera || rig.open;
  $('read-pose').disabled = !job.side;
  $('read-pose').checked = !!job.side;
  $('serial').value = defaults.serials[$('camera').value];
  $('procedure').textContent = procedure(session ? session.job : $('job').value);
}

function updateRig() {
  $('rig-status').textContent = rig.open ? `${rig.camera.role} · ${rig.camera.serial} · 已打开` : '相机未打开';
  $('open').disabled = rig.open || busy;
  $('close').disabled = !rig.open || busy;
  $('create').disabled = !rig.open || busy;
  $('capture').disabled = !rig.open || !session || busy || defaults?.validation_version !== 1 || ($('sample-purpose').value !== 'calibration' && !session.result);
  $('detect').disabled = !rig.open || !session || busy;
  $('solve').disabled = !session || busy;
  $('job').disabled = rig.open;
  $('camera').disabled = rig.open || !!defaults?.jobs[$('job').value].camera;
  $('serial').disabled = rig.open;
  $('read-pose').disabled = rig.open || !defaults?.jobs[$('job').value].side;
  $('compare-bases').disabled = busy || !dualSelection.length || !$('dual-left').value || !$('dual-right').value || defaults?.validation_version !== 1;
  $('load-dual').disabled = busy || defaults?.validation_version !== 1;
  renderIntrinsics();
}

function renderIntrinsics() {
  const camera = session?.camera || (rig.open ? rig.camera : null);
  const intrinsics = camera?.intrinsics;
  $('intrinsics-context').textContent = session ? `当前记录：${session.name} · ${camera.serial}` : camera ? `已打开相机：${camera.serial} · 新建记录时自动保存` : '打开相机后自动读取出厂内参；选择历史记录可查看已保存值。';
  $('intrinsics-values').hidden = !intrinsics;
  $('intrinsics-metrics').replaceChildren();
  $('intrinsics-source-label').textContent = !intrinsics ? '无需先做内参重标定' : intrinsics.source === 'orbbec_selected_video_profile' ? '正在使用出厂内参 · SDK 视频配置' : '当前使用重标定 / 记录内参';
  if (intrinsics) {
    for (const [name, value] of [['fx', intrinsics.K[0][0]], ['fy', intrinsics.K[1][1]], ['cx', intrinsics.K[0][2]], ['cy', intrinsics.K[1][2]]]) {
      const tile = document.createElement('div');
      tile.className = 'quality-tile';
      const label = document.createElement('small');
      label.textContent = name + ' / px';
      const number = document.createElement('strong');
      number.textContent = Number(value).toFixed(3);
      tile.append(label, number);
      $('intrinsics-metrics').append(tile);
    }
    $('intrinsics-values').textContent = JSON.stringify({
      resolution: `${camera.width} × ${camera.height}`,
      sensor: camera.sensor,
      source: intrinsics.source,
      model: intrinsics.sdk_model || intrinsics.model,
      K: intrinsics.K,
      D: intrinsics.D
    }, null, 2);
  }
  const selected = $('intrinsic-source').value;
  const compatible = sessions.filter(s => s.solved && s.job === 'intrinsics' && session && ['role', 'serial', 'sensor', 'width', 'height'].every(k => s.camera[k] === session.camera[k]));
  $('intrinsic-source').replaceChildren(option('', compatible.length ? '选择可选的重标定记录' : '无匹配的重标定记录；不影响使用出厂内参'));
  compatible.forEach(s => $('intrinsic-source').append(option(s.id, s.name)));
  if (compatible.some(s => s.id === selected)) $('intrinsic-source').value = selected;
  $('intrinsic-source').disabled = busy || !compatible.length;
  $('apply-intrinsics').disabled = busy || !session || !$('intrinsic-source').value;
  const job = session?.job || $('job').value;
  const transform = defaults?.jobs[job]?.transform;
  $('transform-direction').textContent = transform ? `${transform}：把相机坐标变换到${job.endsWith('wrist') ? '对应法兰' : '对应 base'}坐标。p_目标 = T_目标_相机 × p_相机；反向变换取矩阵逆。` : '内参任务求解 K 和畸变 D，不产生相机到机械臂的外参。';
}

function option(value, label) {
  const o = document.createElement('option');
  o.value = value;
  o.textContent = label;
  return o;
}
async function reload() {
  const data = await api('state');
  sessions = data.sessions;
  rig = data.rig;
  $('sessions').replaceChildren(option('', '选择记录'));
  const selected = new Set([...document.querySelectorAll('#export-list input:checked')].map(i => i.value));
  $('export-list').replaceChildren();
  for (const s of sessions) {
    $('sessions').append(option(s.id, `${s.name} · ${titles[s.job]} · ${s.sample_count} 张`));
    if (s.solved) {
      const label = document.createElement('label');
      label.className = 'inline';
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.value = s.id;
      check.checked = selected.has(s.id);
      label.append(check, document.createTextNode(`${s.name} / ${titles[s.job]} / ${s.mount_id}`));
      $('export-list').append(label);
    }
  }
  if (session) $('sessions').value = session.id;
  updateRig();
}

function renderSession() {
  if (!session) return;
  $('sessions').value = session.id;
  $('procedure').textContent = procedure(session.job);
  const b = session.board;
  $('session-info').textContent = `当前：${session.name} · ${titles[session.job]} · ${b.kind} ${b.columns}×${b.rows} / ${b.square_mm} mm · 安装 ${session.mount_id} · ${session.camera.serial}`;
  $('sample-count').textContent = `${session.samples.length} 个样本`;
  $('corner-rotation').disabled = b.kind !== 'chessboard';
  for(const option of $('corner-rotation').options)option.disabled=Number(option.value)%2===1&&b.columns!==b.rows;
  if(b.kind!=='chessboard'||(b.columns!==b.rows&&Number($('corner-rotation').value)%2))$('corner-rotation').value='0';
  $('manual-pose').disabled = session.job === 'intrinsics';
  $('sample-purpose').querySelector('[value=dual_validation]').disabled=!session.job.endsWith('wrist');
  if($('sample-purpose').value==='dual_validation'&&!session.job.endsWith('wrist'))$('sample-purpose').value='calibration';
  updatePurpose();
  $('sample-list').replaceChildren();
  session.samples.forEach((sample, index) => {
    const tr = document.createElement('tr');
    const values = [String(index + 1), String(sample.pixels.length), session.result?.per_sample_rms_px?.[index]?.toFixed(3) ?? '待求解'];
    for (const value of values) {
      const td = document.createElement('td');
      td.textContent = value;
      tr.append(td);
    }
    const td = document.createElement('td');
    const view = document.createElement('button');
    view.textContent = '查看角点';
    view.onclick = () => {
      $('preview-live').checked = false;
      showPreview(`api/sample?id=${session.id}&sample=${sample.id}`);
    };
    const remove = document.createElement('button');
    remove.textContent = '移除';
    remove.className = 'danger';
    remove.onclick = () => run(async () => {
      session = await api('remove-sample', {
        id: session.id,
        sample: sample.id
      });
      renderSession();
      await reload();
      status('已移除样本，原求解结果已失效');
    });
    td.append(view, remove);
    tr.append(td);
    $('sample-list').append(tr);
  });
  $('result').textContent = JSON.stringify({intrinsics: session.camera.intrinsics, result: session.result || '尚未求解'}, null, 2);
  renderQuality();
  renderValidation();
  updateRig();
}

function showPreview(source) {
  $('preview').hidden = false;
  $('preview-placeholder').hidden = true;
  $('preview').src = source;
}

function renderQuality() {
  const result = session?.result;
  $('quality').replaceChildren();
  if (!result) return;
  const values = session.job === 'intrinsics' ? [
    ['重投影 RMS', `${result.rms_px.toFixed(3)} px`],
    ['板面倾角范围', `${result.normal_spread_deg.toFixed(1)}°`]
  ] : [
    ['求解集平移 RMS', `${result.translation_rms_mm.toFixed(2)} mm`],
    ['求解集旋转 RMS', `${result.rotation_rms_deg.toFixed(2)}°`],
    ['最大重投影误差', `${Math.max(...result.per_sample_rms_px).toFixed(3)} px`]
  ];
  for (const [label, value] of values) {
    const tile = document.createElement('div');
    tile.className = 'quality-tile';
    const name = document.createElement('small');
    name.textContent = label;
    const metric = document.createElement('div');
    metric.className = 'quality-value';
    metric.textContent = value;
    tile.append(name, metric);
    $('quality').append(tile);
  }
  for (const message of result.warnings) {
    const notice = document.createElement('p');
    notice.className = 'notice';
    notice.textContent = message;
    $('quality').append(notice);
  }
}

function updatePurpose() {
  const purpose=$('sample-purpose').value;
  $('scene-field').hidden=purpose!=='dual_validation';
  $('purpose-help').textContent=purpose==='calibration'?'用于求解；增加或移除后需要重新求解。':purpose==='validation'?'保持求解时板与环境（腕部）或板与法兰（全局）的固定关系；换一个新姿态，仅验证、不重新拟合。':'可把板重新固定在两只腕部相机都能看见的位置；左右使用同一场景名称。本观测不与求解时板的位置比较。';
  $('capture').textContent=purpose==='calibration'?'保存求解样本':purpose==='validation'?'保存独立验证样本':'保存双臂检查观测';
  updateRig();
}
function renderValidation() {
  const summary=session?.validation_summary;
  $('validation-summary').textContent=!summary||!summary.sample_count?'当前求解尚未完成单项独立验证。':`${summary.sample_count} 个独立验证样本 · ${summary.status==='passed'?'均在参考阈值内':'存在超差，请检查'} · 最大重投影 ${summary.max_reprojection_rms_px.toFixed(3)} px`+(session.job==='intrinsics'?'':` · 最大偏差 ${summary.max_translation_mm.toFixed(2)} mm / ${summary.max_rotation_deg.toFixed(2)}° · ${summary.novel_pose_count} 个新姿态`);
  if(summary?.stale_sample_count)$('validation-summary').textContent+=`；${summary.stale_sample_count} 个旧验证已失效。`;
  $('validation-list').replaceChildren();
  for(const sample of session?.validation_samples||[]) {
    const r=sample.validation, current=session.result && (session.result.solution_id?r.solution_id===session.result.solution_id:!summary?.stale_sample_count);
    const row=document.createElement('tr');
    for(const value of [`${purposeNames[sample.purpose]||'验证'} ${sample.scene_id||''}`,`${r.reprojection_rms_px.toFixed(3)} px`,r.translation_mm===undefined?'—':`${r.translation_mm.toFixed(2)} mm / ${r.rotation_deg.toFixed(2)}°`,!current?'已失效':r.within_thresholds?'阈值内':'超差']) {
      const cell=document.createElement('td');cell.textContent=value;row.append(cell);
    }
    row.title=(r.warnings||[]).join('；');
    const cell=document.createElement('td'),view=document.createElement('button'),remove=document.createElement('button');
    view.textContent='查看';view.onclick=()=>{$('preview-live').checked=false;showPreview(`api/sample?id=${session.id}&sample=${sample.id}`);status((r.warnings||[]).join('；')||'验证观测未参与求解');};
    remove.textContent='移除';remove.onclick=()=>run(async()=>{session=await api('remove-sample',{id:session.id,sample:sample.id});renderSession();await reload();status('已移除验证样本，求解结果保留');});
    cell.append(view,remove);row.append(cell);$('validation-list').append(row);
  }
  const all=[...(session?.samples||[]),...(session?.validation_samples||[])];
  const latest=all.filter(s=>s.metadata?.quality).sort((a,b)=>(a.metadata.sample_wall_time_s||0)-(b.metadata.sample_wall_time_s||0)).at(-1);
  $('capture-quality').hidden=!latest;
  if(latest){const q=latest.metadata.quality;$('capture-quality').textContent=`最近采样：角点漂移 ${q.corner_drift_rms_px.toFixed(2)} px；`+(q.pose_verified?`法兰变化 ${q.translation_span_mm.toFixed(2)} mm / ${q.rotation_span_deg.toFixed(2)}°，最大接收时间差 ${(q.max_image_pose_skew_s*1000).toFixed(0)} ms。`:'位姿时间与机械臂静止性未自动验证。');}
}
$('sample-purpose').onchange=updatePurpose;
function exportIds(){return [...document.querySelectorAll('#export-list input:checked')].map(i=>i.value);}
$('load-dual').onclick=()=>run(async()=>{
  const ids=exportIds(), rows=await Promise.all(ids.map(id=>api(`session?id=${id}`)));
  const jobs=Object.fromEntries(rows.map(s=>[s.job,s]));
  if(!['left_wrist','right_wrist','global_left','global_right'].every(j=>jobs[j]))throw new Error('请先在导出列表勾选四项外参结果');
  dualSelection=ids;
  for(const side of ['left','right']){
    const record=jobs[`${side}_wrist`],select=$(`dual-${side}`);select.replaceChildren(option('','选择验证观测'));
    for(const sample of record.validation_samples||[])if(sample.scene_id&&(!record.result.solution_id||sample.validation.solution_id===record.result.solution_id))select.append(option(sample.id,`${sample.scene_id} · ${sample.id.slice(0,8)}`));
  }
  status('已载入观测；请选择左右同一固定板场景');
});
for(const side of ['left','right'])$(`dual-${side}`).onchange=updateRig;
$('export-list').onchange=()=>{dualSelection=[];updateRig();};
$('compare-bases').onclick=()=>run(async()=>{
  const r=await api('compare-bases',{ids:dualSelection,left_sample:$('dual-left').value,right_sample:$('dual-right').value,board_fixed:$('dual-fixed').checked});
  $('dual-summary').textContent=`同一标定板在左基座下：${r.translation_mm.toFixed(2)} mm / ${r.rotation_deg.toFixed(2)}°，最大重投影 ${r.max_reprojection_rms_px.toFixed(3)} px · ${r.within_thresholds?'在 5 mm / 2° / 2 px 参考阈值内':'超过参考阈值，请检查标定、图像与板固定情况'}。报告已保存，将随对应标定包导出。`;
  $('dual-report').textContent=JSON.stringify(r,null,2);status('双臂独立一致性检查已完成');
});

function requireSession() {
  if (!session) throw new Error('请先新建或选择标定记录');
}
$('job').onchange = () => { changeJob(); updateRig(); };
$('intrinsic-source').onchange = updateRig;
$('camera').onchange = () => {
  $('serial').value = defaults.serials[$('camera').value];
};
for (const id of ['board-kind', 'columns', 'rows', 'square-mm', 'marker-ratio', 'dictionary']) $(id).oninput = updateBoard;
$('open').onclick = () => run(async () => {
  const job = defaults.jobs[$('job').value];
  const camera = await api('open', {
    camera: $('camera').value,
    serial: $('serial').value,
    side: $('read-pose').checked ? job.side : null
  });
  rig = {
    open: true,
    camera
  };
  $('preview-live').checked = true;
  status('相机已打开。只读位姿连接不会操作机械臂。');
});
$('close').onclick = () => run(async () => {
  await api('close', {});
  rig = {
    open: false
  };
  $('preview').removeAttribute('src');
  $('preview').hidden = true;
  $('preview-placeholder').hidden = false;
  status('相机和只读位姿连接已关闭');
});
$('create').onclick = () => run(async () => {
  session = await api('create', {
    job: $('job').value,
    board: board(),
    name: $('session-name').value,
    mount_id: $('mount-id').value
  });
  await reload();
  renderSession();
  status('记录已创建，当前板参数已保存');
});
$('sessions').onchange = () => run(async () => {
  if (!$('sessions').value) {
    session = null;
    $('session-info').textContent = '新建或选择一份记录后开始采样。';
    $('sample-count').textContent = '0 个样本';
    $('sample-list').replaceChildren();
    $('quality').replaceChildren();
    $('result').textContent = '尚未选择记录';
    renderValidation();
    $('procedure').textContent = procedure($('job').value);
    return;
  }
  session = await api(`session?id=${$('sessions').value}`);
  const saved = session.board;
  $('board-kind').value = saved.kind;
  $('columns').value = saved.columns;
  $('rows').value = saved.rows;
  $('square-mm').value = saved.square_mm;
  $('marker-ratio').value = saved.marker_ratio;
  $('dictionary').value = saved.dictionary;
  updateBoard();
  if (!rig.open) {
    $('job').value = session.job;
    changeJob();
    $('camera').value = session.camera.role;
    $('serial').value = session.camera.serial;
  }
  renderSession();
  status('已载入标定记录');
});
$('reload').onclick = () => run(reload);
$('manual-pose').onchange = () => {
  $('manual-fields').hidden = !$('manual-pose').checked;
};
$('detect').onclick = () => run(async () => {
  requireSession();
  const result = await api('detect', {
    id: session.id,
    rotation_quarters: Number($('corner-rotation').value)
  });
  $('preview-live').checked = false;
  showPreview(result.image);
  status(`检测到 ${result.count} 个角点。请核对标号 0 的物理位置，再保存样本。`);
});
$('capture').onclick = () => run(async () => {
  requireSession();
  let pose = null;
  if ($('manual-pose').checked && session.job !== 'intrinsics') {
    pose = $('pose').value.trim().split(/[,，\s]+/).map(Number);
    if (pose.length !== 6 || !pose.every(Number.isFinite)) throw new Error('请输入 6 个有限数：xyz(m), roll/pitch/yaw(rad)');
  }
  session = await api('capture', {
    id: session.id,
    rotation_quarters: Number($('corner-rotation').value),
    pose,
    purpose: $('sample-purpose').value,
    scene_id: $('scene-id').value,
    still_confirmed: $('still').checked
  });
  renderSession();
  await reload();
  $('still').checked = false;
  status(`已保存${purposeNames[$('sample-purpose').value]}。求解样本 ${session.samples.length} 个，验证观测 ${(session.validation_samples||[]).length} 个。`);
});
$('solve').onclick = () => run(async () => {
  requireSession();
  status('正在求解…');
  session = await api('solve', {
    id: session.id
  });
  renderSession();
  await reload();
  status(session.result.warnings.length ? '求解完成，请检查质量提示。' : '求解完成，请结合样本残差检查结果。');
});
$('apply-intrinsics').onclick = () => run(async () => {
  requireSession();
  session = await api('use-intrinsics', {
    id: session.id,
    source: $('intrinsic-source').value
  });
  renderSession();
  status('已应用内参，请重新求解外参');
});
$('export').onclick = () => run(async () => {
  const ids = exportIds();
  const result = await api('export', {
    ids,
    accept_warnings: $('accept-warnings').checked
  });
  exportPath = result.path;
  $('export-validation').textContent=JSON.stringify({derived_transforms:result.bundle.derived_transforms,validation:result.bundle.calibrations.map(c=>({job:c.job,...c.validation_summary})),dual_base_validation:result.bundle.dual_base_validation},null,2);
  $('export-path').textContent = exportPath;
  $('export-path').hidden = false;
  $('copy-export').hidden = false;
  $('download-export').hidden = false;
  $('download-export').href = `api/export?id=${result.export_id}`;
  status(result.bundle.missing_jobs.length ? `已导出部分标定；尚缺：${result.bundle.missing_jobs.map(j=>titles[j]).join('、')}` : '四项外参、双基座变换及验证记录已导出。');
});
$('copy-export').onclick = async () => {
  try {
    let copied = false;
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(exportPath);
        copied = true;
      } catch {}
    }
    if (!copied) {
      const text = document.createElement('textarea');
      text.value = exportPath;
      document.body.append(text);
      text.select();
      const ok = document.execCommand('copy');
      text.remove();
      if (!ok) throw new Error('请从路径栏手动复制');
    }
    status('已复制标定包路径');
  } catch (e) {
    status(e.message, true);
  }
};
async function preview() {
  if (!rig.open || !$('preview-live').checked || previewBusy || document.hidden || busy) return;
  previewBusy = true;
  try {
    const response = await fetch(`api/frame?t=${Date.now()}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error);
    }
    const url = URL.createObjectURL(await response.blob());
    const old = $('preview').src;
    showPreview(url);
    if (old.startsWith('blob:')) URL.revokeObjectURL(old);
  } catch (e) {
    status(e.message, true);
  } finally {
    previewBusy = false;
  }
}
run(async () => {
  defaults = await api('bootstrap');
  $('storage').textContent = defaults.directory;
  $('session-name').value = `左腕_${new Date().toLocaleDateString('sv')}`;
  changeJob();
  updateBoard();
  await reload();
  updatePurpose();
  if(defaults.validation_version!==1)status('标定服务仍运行旧代码，请关闭设备后重启标定模块以使用新采样与验证功能',true);
});
setInterval(preview, 500);
