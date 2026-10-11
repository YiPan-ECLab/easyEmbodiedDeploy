'use strict';

const fpsLabel = (fps) => fps === 'mixed' ? 'mixed' : `${fps} Hz`;

const $ = (id) => document.getElementById(id);
const numberFields = new Set([
  'rtc_max_delay','lora_rank','lora_alpha','batch_size','steps','save_freq','log_freq',
  'num_workers','seed','n_action_steps',
]);
const booleanFields = new Set(['gradient_checkpointing','save_checkpoint','wandb_enabled','dry_run']);
const activeStates = new Set(['validating','launching','loading','running','saving','stopping']);
const stateLabels = {
  idle:'空闲',validating:'验证中',launching:'启动中',loading:'加载模型',running:'训练中',
  saving:'保存中',stopping:'正在停止',stopped:'已停止',completed:'已完成',failed:'失败',
};

let bootstrap = null;
let selectedGpus = new Set([0]);
let lastStatus = null;
let lastLogId = 0;
let previewTimer = null;
let statusTimer = null;
let logsTimer = null;

function firstVisibleGpu() {
  const gpus = bootstrap?.status?.gpu || [];
  return gpus.length ? gpus[0].index : 0;
}

function markProfile(profile) {
  document.querySelectorAll('[data-profile]').forEach((node) => node.classList.toggle('active', node.dataset.profile === profile));
}

function toast(message, type = '') {
  const node = document.createElement('div');
  node.className = `toast ${type}`;
  node.textContent = message;
  $('toastStack').append(node);
  setTimeout(() => node.remove(), 4200);
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  let body;
  try { body = await response.json(); } catch { body = {error: `HTTP ${response.status}`}; }
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function post(path, payload = {}) {
  return request(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
}

function formConfig() {
  const config = {};
  for (const element of $('configForm').elements) {
    if (!element.name) continue;
    if (booleanFields.has(element.name)) config[element.name] = element.checked;
    else if (numberFields.has(element.name)) config[element.name] = Number(element.value);
    else config[element.name] = element.value.trim();
  }
  config.gpu_ids = [...selectedGpus].sort((a,b) => a-b).join(',') || '0';
  return config;
}

function setField(name, value) {
  const element = $(name);
  if (!element) return;
  if (booleanFields.has(name)) element.checked = Boolean(value);
  else element.value = value ?? '';
}

function applyConfig(config) {
  for (const [key, value] of Object.entries(config)) setField(key, value);
  const requested = String(config.gpu_ids ?? 'auto').trim().toLowerCase();
  selectedGpus = requested === 'auto'
    ? new Set([firstVisibleGpu()])
    : new Set(requested.split(',').map(Number).filter(Number.isInteger));
  if (!selectedGpus.size) selectedGpus.add(0);
  updateConditionalFields();
}

function fillDiscoveries(data) {
  $('datasetOptions').replaceChildren(...data.datasets.map((item) => {
    const option = document.createElement('option');
    option.value = item.path;
    option.label = `${item.label} · ${fpsLabel(item.fps)} · ${item.action_mode || 'unknown'}`;
    return option;
  }));
  $('modelOptions').replaceChildren(...data.models.map((item) => {
    const option = document.createElement('option');
    option.value = item.path;
    option.label = `${item.label} · ${item.kind} · chunk ${item.chunk_size}`;
    return option;
  }));
  $('resumeOptions').replaceChildren(...data.resume_configs.map((item) => {
    const option = document.createElement('option');
    option.value = item.path;
    option.label = `${item.label}${item.step == null ? '' : ` · step ${item.step}`}`;
    return option;
  }));
}

function matchedDataset() {
  if (!bootstrap) return null;
  const value = $('dataset_root').value.replace(/\/+$/, '');
  return bootstrap.datasets.find((item) =>
    item.path.replace(/\/+$/, '') === value || item.resolved_root.replace(/\/+$/, '') === value
  );
}

function updateDatasetSelection() {
  const item = matchedDataset();
  if (!item) {
    $('datasetHint').textContent = '训练前会检查 14 维轴顺序、三路图像、统计量和首帧解码';
    return;
  }
  if (item.action_mode) $('action_mode').value = item.action_mode;
  $('datasetHint').textContent = `${item.episodes} episodes · ${Number(item.frames).toLocaleString()} frames · ${fpsLabel(item.fps)} · ${item.action_mode || '动作模式未知'}`;
  $('flowDataset').textContent = `${fpsLabel(item.fps)} dataset`;
}

function updateConditionalFields() {
  const lora = $('finetune_mode').value === 'lora';
  document.querySelectorAll('.lora-field').forEach((node) => node.classList.toggle('dimmed', !lora));
  $('training_scope').disabled = lora;
  $('finetuneHint').textContent = lora ? '训练 PI05 默认 adapter 目标，显存需求较低' : '直接更新选定范围内的原始模型权重';
  $('scopeHint').textContent = lora ? 'LoRA 固定使用动作专家 q/v 与状态/动作投影' : '该范围会直接决定冻结哪些原始权重';
  const rtc = $('rtc_mode').value === 'trained';
  $('rtc_max_delay').disabled = !rtc;
  document.querySelectorAll('.rtc-delay-field').forEach((node) => node.classList.toggle('dimmed', !rtc));
  $('rtcHint').textContent = rtc ? '在训练中注入动作块延迟，用于部署 trained RTC' : '标准 PI05 训练；部署端仍可使用 guided RTC';
  const wandb = $('wandb_enabled').checked;
  document.querySelectorAll('.wandb-field input,.wandb-field select').forEach((node) => node.disabled = !wandb);
  const cuda = $('device').value === 'cuda';
  $('gpuPicker').classList.toggle('disabled', !cuda);
  $('flowMode').textContent = lora ? 'PI05 + LoRA' : `PI05 · ${$('training_scope').selectedOptions[0]?.textContent || 'direct'}`;
  $('flowRtc').textContent = rtc ? `RTC delay 1–${$('rtc_max_delay').value}` : 'Normal action chunk';
}

function applyProfile(profile) {
  if (profile === 'lora') {
    setField('finetune_mode','lora'); setField('training_scope','expert_only');
    setField('lora_rank',16); setField('lora_alpha',16); setField('gradient_checkpointing',true);
  } else if (profile === 'full') {
    setField('finetune_mode','full'); setField('training_scope','full_model');
    setField('batch_size',1); setField('gradient_checkpointing',true);
  } else if (profile === 'smoke') {
    setField('finetune_mode','lora'); setField('training_scope','expert_only'); setField('batch_size',1);
    setField('steps',1); setField('save_freq',1); setField('log_freq',1); setField('num_workers',0);
    setField('save_checkpoint',false); setField('wandb_enabled',false); setField('dry_run',false);
    setField('device','cuda'); selectedGpus = new Set([0]);
    const root = bootstrap?.defaults?.output_dir ? bootstrap.defaults.output_dir.replace(/openpi_pi05_.+$/, '') : '';
    const stamp = new Date().toISOString().replace(/[-:TZ.]/g,'').slice(0,14);
    setField('output_dir', `${root}openpi_pi05_gpu_smoke_${stamp}`);
  }
  renderGpuPicker(lastStatus?.gpu || bootstrap?.status?.gpu || []);
  updateConditionalFields(); markProfile(profile); updatePreviewSoon();
}

function gpuMemoryText(gpu) {
  return `${(gpu.memory_used_mb/1024).toFixed(1)} / ${(gpu.memory_total_mb/1024).toFixed(1)} GiB`;
}

function renderGpuPicker(gpus) {
  const picker = $('gpuPicker');
  picker.replaceChildren();
  if (!gpus.length) { picker.innerHTML = '<span class="muted">未检测到 NVIDIA GPU</span>'; return; }
  for (const gpu of gpus) {
    const label = document.createElement('label');
    const busy = gpu.processes.length > 0;
    label.className = `gpu-choice ${selectedGpus.has(gpu.index) ? 'selected' : ''} ${busy ? 'busy' : ''}`;
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox'; checkbox.checked = selectedGpus.has(gpu.index); checkbox.disabled = $('device').value !== 'cuda';
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) selectedGpus.add(gpu.index); else selectedGpus.delete(gpu.index);
      if (!selectedGpus.size) { selectedGpus.add(gpu.index); checkbox.checked = true; toast('CUDA 模式至少选择一张 GPU','error'); }
      $('gpu_ids').value = [...selectedGpus].sort((a,b)=>a-b).join(',');
      renderGpuPicker(gpus); updatePreviewSoon();
    });
    const title = document.createElement('strong'); title.textContent = `GPU ${gpu.index} · ${gpu.name}`;
    const detail = document.createElement('span'); detail.textContent = `${gpuMemoryText(gpu)} · ${gpu.utilization}%`;
    const badge = document.createElement('em'); badge.textContent = busy ? `${gpu.processes.length} process` : '空闲';
    label.append(checkbox,title,detail,badge); picker.append(label);
  }
}

function renderGpuList(gpus) {
  const list = $('gpuList'); list.replaceChildren();
  if (!gpus.length) { list.innerHTML = '<p class="muted">未检测到 NVIDIA GPU</p>'; return; }
  for (const gpu of gpus) {
    const row = document.createElement('div'); row.className = `gpu-row ${gpu.processes.length ? 'busy' : ''}`;
    const pct = Math.min(100, gpu.memory_total_mb ? gpu.memory_used_mb / gpu.memory_total_mb * 100 : 0);
    const head = document.createElement('div'); head.className = 'gpu-row-head';
    const name = document.createElement('strong'); name.textContent = `GPU ${gpu.index} · ${gpu.name}`;
    const temp = document.createElement('span'); temp.textContent = `${gpu.temperature}°C · ${gpu.utilization}% util`;
    head.append(name,temp);
    const bar = document.createElement('div'); bar.className='gpu-bar'; const fill=document.createElement('i'); fill.style.width=`${pct}%`; bar.append(fill);
    const meta=document.createElement('div'); meta.className='gpu-row-meta'; meta.innerHTML=`<span>显存 ${gpuMemoryText(gpu)}</span><span>${pct.toFixed(1)}%</span>`;
    row.append(head,bar,meta);
    if (gpu.processes.length) {
      const p=document.createElement('div'); p.className='gpu-processes';
      p.textContent=gpu.processes.map(x=>`PID ${x.pid} · ${x.memory_mb} MiB`).join('  |  '); row.append(p);
    }
    list.append(row);
  }
  $('gpuRefreshTime').textContent = new Date().toLocaleTimeString('zh-CN',{hour12:false});
}

function renderExternal(items) {
  $('externalNotice').classList.toggle('hidden', !items.length);
  if (!items.length) return;
  $('externalText').textContent = items.map((item) => {
    const job = item.command.match(/--job-name\s+([^ ]+)/)?.[1];
    return `PID ${item.pid} · OpenPI JAX${job ? ` · ${job}` : ''}`;
  }).join('；');
}

function renderDataset(report) {
  if (!report) return;
  $('datasetEpisodes').textContent = Number(report.episodes).toLocaleString();
  $('datasetFrames').textContent = Number(report.frames).toLocaleString();
  $('datasetFps').textContent = `${fpsLabel(report.fps)}`;
  $('datasetState').textContent = report.sample_checked ? '首帧已验证' : '元数据通过';
  document.querySelector('.dataset-panel').classList.add('valid');
  $('stepDataset').className = 'ready';
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const h=Math.floor(seconds/3600),m=Math.floor(seconds%3600/60),s=Math.floor(seconds%60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
}

function renderStatus(status) {
  lastStatus = status;
  const training = status.training;
  const active = activeStates.has(training.state);
  const pct = training.total_steps ? Math.min(100, training.step/training.total_steps*100) : 0;
  $('runState').textContent = stateLabels[training.state] || training.state;
  $('runMeta').textContent = training.pid ? `PID ${training.pid} · ${formatDuration(training.elapsed_s)}` : '等待配置';
  $('progressValue').textContent = `${pct.toFixed(1)}%`;
  $('progressMeta').textContent = `${Number(training.step).toLocaleString()} / ${Number(training.total_steps).toLocaleString()} steps`;
  $('progressRing').style.setProperty('--progress',`${pct}%`);
  const loss = training.metrics.loss;
  $('lossValue').textContent = Number.isFinite(loss) ? loss.toFixed(4) : '—';
  $('speedMeta').textContent = training.seconds_per_step ? `${training.seconds_per_step.toFixed(3)} s/step · ETA ${training.eta}` : '等待首条指标';
  const ids = training.gpu_ids.length ? training.gpu_ids : [...selectedGpus];
  $('gpuValue').textContent = $('device').value === 'cpu' ? 'CPU' : ids.map(x=>`GPU ${x}`).join(' + ');
  const selected = status.gpu.filter(g=>ids.includes(g.index));
  $('gpuMeta').textContent = selected.length ? selected.map(g=>`${g.utilization}% · ${(g.memory_used_mb/1024).toFixed(1)} GiB`).join(' | ') : '未分配';
  $('runMetric').classList.toggle('running', active);
  $('runMetric').classList.toggle('failed', training.state === 'failed');
  $('startButton').disabled = active;
  $('stopButton').disabled = !active;
  $('stopTop').disabled = !active;
  $('stepTrain').className = training.state === 'failed' ? 'failed' : active ? 'active' : training.state === 'completed' ? 'ready' : '';
  $('stepGpu').className = active || training.state === 'completed' ? 'ready' : '';
  $('actionNote').textContent = active ? `本页进程 PID ${training.pid} 正在${stateLabels[training.state] || training.state}。外部训练不在本页进程组中。` : training.state === 'failed' ? `训练退出码 ${training.exit_code}，请查看末尾错误日志。` : '启动前会拒绝使用已有计算进程的 GPU。停止按钮只作用于本页面启动的进程组。';
  renderGpuPicker(status.gpu); renderGpuList(status.gpu); renderExternal(status.external_training); renderDataset(status.dataset); drawChart(training.history);
}

function drawChart(history) {
  const canvas=$('metricChart'), empty=$('chartEmpty');
  const points=history.filter(x=>Number.isFinite(x.loss) || Number.isFinite(x.grdn));
  empty.classList.toggle('hidden', points.length > 0);
  const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  const width=Math.max(300,rect.width),height=210;
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
  const pad={l:42,r:16,t:12,b:27};ctx.strokeStyle='rgba(24,54,46,.10)';ctx.lineWidth=1;
  for(let i=0;i<5;i++){const y=pad.t+(height-pad.t-pad.b)*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(width-pad.r,y);ctx.stroke()}
  if(!points.length)return;
  const losses=points.map(p=>p.loss).filter(Number.isFinite),grads=points.map(p=>p.grdn).filter(Number.isFinite);
  const max=Math.max(...losses,...grads,1e-6),min=Math.min(...losses,...grads,0);
  const x=(i)=>pad.l+(width-pad.l-pad.r)*(points.length===1?1:i/(points.length-1));
  const y=(v)=>pad.t+(height-pad.t-pad.b)*(1-(v-min)/(max-min||1));
  const line=(key,color)=>{ctx.beginPath();let begun=false;points.forEach((p,i)=>{if(!Number.isFinite(p[key]))return;begun?(ctx.lineTo(x(i),y(p[key]))):(ctx.moveTo(x(i),y(p[key])),begun=true)});ctx.strokeStyle=color;ctx.lineWidth=1.8;ctx.stroke()};
  line('loss','#12664d');line('grdn','#167a83');
  ctx.fillStyle='#69726e';ctx.font='9px monospace';ctx.fillText(max.toPrecision(3),3,pad.t+3);ctx.fillText(min.toPrecision(3),3,height-pad.b+3);
  ctx.fillText(String(points[0].step),pad.l,height-8);ctx.textAlign='right';ctx.fillText(String(points.at(-1).step),width-pad.r,height-8);ctx.textAlign='left';
}

function appendLogs(items) {
  if (!items.length) return;
  const terminal=$('terminal');terminal.querySelector('.terminal-placeholder')?.remove();
  for(const item of items){const line=document.createElement('div');line.className=`terminal-line ${item.level}`;const time=document.createElement('time');time.textContent=item.time;const source=document.createElement('b');source.textContent=item.source;const message=document.createElement('span');message.textContent=item.message;line.append(time,source,message);terminal.append(line)}
  while(terminal.children.length>5000)terminal.firstElementChild.remove();
  if($('autoScroll').checked)terminal.scrollTop=terminal.scrollHeight;
}

async function pollStatus() {
  try {
    const status=await request('api/status');renderStatus(status);
    $('connectionPill').className='connection-pill online';$('connectionText').textContent='控制台已连接';
  } catch(error) {
    $('connectionPill').className='connection-pill error';$('connectionText').textContent='连接中断';
  } finally { statusTimer=setTimeout(pollStatus,1500); }
}

async function pollLogs() {
  try {const data=await request(`api/logs?after=${lastLogId}`);appendLogs(data.items);lastLogId=data.last_id;} catch {}
  finally {logsTimer=setTimeout(pollLogs,850)}
}

async function preview(silent=false) {
  try {
    const result=await post('api/preview',{config:formConfig()});
    $('commandPreview').textContent=result.openpi;
    $('cudaVisibility').textContent=`CUDA_VISIBLE_DEVICES = ${result.cuda_visible_devices || '(hidden)'}`;
    $('launcherMode').textContent=result.config.gpu_ids.length>1?'JAX · FSDP':'JAX · single process';
    renderDataset(result.dataset);
    if(!silent)toast('配置与命令检查通过','success');
  } catch(error) {if(!silent)toast(error.message,'error');}
}

function updatePreviewSoon(){clearTimeout(previewTimer);previewTimer=setTimeout(()=>preview(true),650)}

async function validateDatasetDeep() {
  const button=$('validateButton');button.disabled=true;button.textContent='正在解码首帧…';
  try {const result=await post('api/validate',{config:formConfig()});renderDataset(result.dataset);toast('数据元数据、统计量和首帧解码全部通过','success');}
  catch(error){$('stepDataset').className='failed';toast(error.message,'error')}
  finally{button.disabled=false;button.textContent='深度验证数据'}
}

async function startTraining() {
  const config=formConfig();
  try {await post('api/start',{config});toast(config.dry_run?'Dry run 已启动':'训练已在隔离进程中启动','success');await pollOnce();}
  catch(error){toast(error.message,'error')}
}

async function pollOnce(){try{renderStatus(await request('api/status'))}catch{}}

function openStop(){if(!$('stopButton').disabled)$('confirmDialog').showModal()}
async function confirmStop(){
  $('confirmDialog').close();
  try{await post('api/stop');toast('已发送 SIGINT，正在等待本页训练退出','success');await pollOnce()}catch(error){toast(error.message,'error')}
}

async function clearLogs(){
  try{await post('api/logs/clear');$('terminal').innerHTML='<div class="terminal-placeholder">日志显示已清空…</div>';lastLogId=0;toast('日志显示已清空','success')}catch(error){toast(error.message,'error')}
}

async function initialize() {
  try {
    bootstrap=await request('api/bootstrap');fillDiscoveries(bootstrap);applyConfig(bootstrap.defaults);
    updateDatasetSelection();renderStatus(bootstrap.status);renderGpuPicker(bootstrap.status.gpu);renderGpuList(bootstrap.status.gpu);renderExternal(bootstrap.status.external_training);
    $('connectionPill').className='connection-pill online';$('connectionText').textContent='控制台已连接';
    preview(true);pollStatus();pollLogs();
  } catch(error) {toast(error.message,'error');$('connectionPill').className='connection-pill error';$('connectionText').textContent='启动失败';}
}

$('configForm').addEventListener('input',(event)=>{if(event.target.id==='dataset_root')updateDatasetSelection();updateConditionalFields();updatePreviewSoon()});
$('configForm').addEventListener('change',(event)=>{if(event.target.id==='wandb_mode'&&event.target.value==='disabled')$('wandb_enabled').checked=false;if(event.target.id==='device')renderGpuPicker(lastStatus?.gpu||[]);if(event.target.id==='finetune_mode')markProfile(event.target.value==='lora'?'lora':'full');updateConditionalFields();updatePreviewSoon()});
document.querySelectorAll('[data-profile]').forEach(node=>node.addEventListener('click',()=>applyProfile(node.dataset.profile)));
$('startButton').addEventListener('click',startTraining);$('stopButton').addEventListener('click',openStop);$('stopTop').addEventListener('click',openStop);
$('validateButton').addEventListener('click',validateDatasetDeep);$('previewButton').addEventListener('click',()=>preview(false));
$('refreshButton').addEventListener('click',async()=>{await pollOnce();toast('状态已刷新','success')});
$('copyCommand').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('commandPreview').textContent);toast('命令已复制','success')}catch{toast('浏览器未允许剪贴板访问','error')}});
$('clearLogs').addEventListener('click',clearLogs);$('cancelStop').addEventListener('click',()=>$('confirmDialog').close());$('confirmStop').addEventListener('click',confirmStop);
window.addEventListener('resize',()=>drawChart(lastStatus?.training?.history||[]));
window.addEventListener('beforeunload',()=>{clearTimeout(statusTimer);clearTimeout(logsTimer)});
initialize();
