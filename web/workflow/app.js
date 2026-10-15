'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const uid = () => crypto.randomUUID().replaceAll('-', '');
let plan, savedId = null, library = [], selected = 0, live = {}, editingPoint = null, frozen = null, pixel = null, dirty = false, toastTimer, previewTimer;
let compatible = false, actionError = "";
const sideName = {left:'左从臂',right:'右从臂'};
const modeNames = {absolute_joint:'绝对关节角',delta_joint:'增量关节角',absolute_base_pose:'绝对末端位姿',camera_delta_wrist:'腕部相机增量位姿',camera_delta_global:'全局相机增量位姿'};
async function api(path, body) {
  const response = await fetch(`api/${path}`, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `请求失败 ${response.status}`);
  return result;
}
function toast(message, error=false) { $('toast').textContent=message; $('toast').classList.toggle('error',error); $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').hidden=true,6000); }
const act = fn => async event => {
  const button=event?.currentTarget?.tagName==='BUTTON'?event.currentTarget:null;
  if(button)button.disabled=true;
  try { actionError='';await fn(event); }
  catch(error) { actionError=error.message;toast(error.message,true); }
  finally { if(button)button.disabled=false;await poll(); }
};
function changed() { dirty=true; $('dirty').textContent='草稿已在浏览器自动保存'; localStorage.setItem('workflowDraft',JSON.stringify({plan,savedId})); }
function tab(name) { document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==`tab-${name}`);document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===name)); }
function options(rows, chosen) { return rows.map(([value,label])=>`<option value="${esc(value)}" ${value===chosen?'selected':''}>${esc(label)}</option>`).join(''); }
function pointOptions(chosen) { return options(plan.points.map(p=>[p.id,p.name]),chosen); }
function modelOptions(chosen) { return options(plan.models.map(p=>[p.id,p.name]),chosen); }
function renderSaved(rows) { $('saved').innerHTML='<option value="">打开已保存工作流</option>'+options(rows.map(p=>[p.id,`${p.name} · ${p.steps} 步`]),savedId); }
function render() { $('plan-name').value=plan.name;renderPoints();renderModels();renderSteps();renderSettings();renderState(live); }
function pointCoordinates(point) {
  return Object.entries(point.arms).map(([side,q])=>{
    const labels=point.space==='joint'?['J1','J2','J3','J4','J5','J6']:['X','Y','Z','Roll','Pitch','Yaw'];
    const feedback=point.source?.feedback?.[side];
    const values=q.slice(0,6).map((v,i)=>{
      const range=(feedback?.joint_limits||live.joint_limits)?.[i];
      const outside=point.space==='joint'&&range&&(v<range[0]-1e-7||v>range[1]+1e-7);
      const label=`${labels[i]} ${(v*(point.space==='pose'&&i<3?1000:180/Math.PI)).toFixed(3)}${point.space==='pose'&&i<3?' mm':'°'}`;
      return `<span ${outside?'class="error"':''}>${label}${outside?'（超出可下发范围）':''}</span>`;
    });
    const freshness=feedback?.received_at_s?`${esc(feedback.channel)} · 最旧反馈 ${feedback.max_age_ms.toFixed(1)} ms · 组间差 ${feedback.group_span_ms.toFixed(1)} ms`:point.source?.simulated?'模拟数据':point.source?.kind==='manual'?'手动输入坐标':'旧记录：没有完整反馈时间戳，准确性需重新核对';
    return `<div class="point-coordinates"><strong>${sideName[side]} · 保存的原始坐标</strong><div class="joint-values">${values.join('')}<span>夹爪 ${(q[6]*1000).toFixed(3)} mm</span></div><small>${freshness}</small></div>`;
  }).join('');
}
function renderPoints() {
  $('point-count').textContent=plan.points.length;
  $('add-point-select').innerHTML=pointOptions($('add-point-select').value);
  const search=$('point-search').value.trim().toLowerCase();
  const points=plan.points.filter(p=>`${p.name} ${p.notes||''}`.toLowerCase().includes(search));
  $('points').innerHTML=points.length?points.map(p=>{
    const count=plan.steps.filter(s=>s.type==='point'&&s.point===p.id).length;
    return `<div class="point-row" data-point="${esc(p.id)}"><div><h3>${esc(p.name)}</h3><p>${Object.keys(p.arms).map(s=>sideName[s]).join(' + ')} · ${p.space==='joint'?'关节角':'末端位姿'} · 已引用 ${count} 次</p>${pointCoordinates(p)}<p class="muted">${esc(p.notes||'可反复添加到步骤中')}</p></div><div class="toolbar"><button data-point-action="add">添加步骤</button><button data-point-action="test">试运行</button><button data-point-action="edit">编辑 / 改名</button><button data-point-action="recapture">用当前位置重录</button><details class="point-more"><summary>更多</summary><div class="toolbar"><button data-point-action="copy">复制</button><button data-point-action="delete">删除</button></div></details></div></div>`;
  }).join(''):'<div class="empty">先录制一个基准点，或从共享点库导入。</div>';
}
function pointSummary(id) { const p=plan.points.find(p=>p.id===id);return p?`${Object.keys(p.arms).map(s=>sideName[s]).join(' / ')} · ${p.space==='joint'?'关节角':'末端位姿'}`:'请选择关键点'; }
function renderSteps() {
  $('steps').innerHTML=plan.steps.length?plan.steps.map((s,i)=>{
    let fields='';
    if(s.type==='point') fields=`<label>关键点引用<select data-field="point">${pointOptions(s.point)}</select><small>${pointSummary(s.point)}</small></label><label>期望运动时长（秒）<input data-field="duration" type="number" min=".05" step=".1" value="${s.duration}"></label><label class="check"><input data-field="blend" type="checkbox" ${s.blend!==false?'checked':''}>与下一个关节点平滑衔接</label>`;
    if(s.type==='vla') fields=`<label>模型<select data-field="model">${modelOptions(s.model)}</select></label><label>控制侧别<select data-field="side">${options([['both','双臂'],['left','仅左臂'],['right','仅右臂']],s.side||'both')}</select></label><label>结束条件<select data-field="until">${options([['duration','执行指定秒数'],['actions','取出指定动作数'],['manual','手动点击下一步']],s.until||'duration')}</select></label><label>${s.until==='actions'?'动作数':'时长（秒）'}<input data-field="value" type="number" min="1" value="${s.value||10}" ${s.until==='manual'?'disabled':''}></label><label>此步骤任务指令<textarea data-field="task" placeholder="留空使用模型默认指令">${esc(s.task||'')}</textarea></label>`;
    if(s.type==='wait') fields=`<label>等待时长（秒）<input data-field="duration" type="number" min="0" step=".1" value="${s.duration}"></label>`;
    if(s.type==='gripper') fields=`<label>机械臂<select data-field="side">${options([['both','双臂'],['left','左臂'],['right','右臂']],s.side)}</select></label><label>夹爪宽度（mm）<input data-field="width" type="number" min="0" max="100" value="${s.width*1000}"></label><label>时长（秒）<input data-field="duration" type="number" min=".05" step=".1" value="${s.duration}"></label>`;
    return `<article class="step ${i===selected?'selected':''}" draggable="true" data-index="${i}"><div class="step-head"><span class="step-number">⠿ ${String(i+1).padStart(2,'0')}</span><input class="step-name" data-field="name" aria-label="步骤名称" value="${esc(s.name)}"><button data-step-action="up" title="上移">↑</button><button data-step-action="down" title="下移">↓</button><button data-step-action="copy">复制</button><button data-step-action="delete">删除</button></div><div class="form-grid">${fields}</div></article>`;
  }).join(''):'<div class="empty">添加关键点、VLA、夹爪或等待步骤，组成你的执行流。</div>';
  highlight();renderState(live);
}
function addStep(type, reference) {
  if(type==='point'&&!reference) throw new Error('请先录制或选择一个关键点');
  if(type==='vla'&&!plan.models.length) {tab('models');throw new Error('请先添加 VLA 模型');}
  let s={id:uid(),type};
  if(type==='point') Object.assign(s,{name:plan.points.find(p=>p.id===reference).name,point:reference,duration:1,blend:true});
  if(type==='vla') Object.assign(s,{name:'VLA 控制',model:plan.models[0].id,until:'duration',value:10,task:''});
  if(type==='gripper') Object.assign(s,{name:'夹爪开合',side:'both',width:.04,duration:1});
  if(type==='wait') Object.assign(s,{name:'等待',duration:1});
  plan.steps.push(s);selected=plan.steps.length-1;changed();renderSteps();renderPoints();tab('flow');
}
function installStepSorting() {
  const list=$('steps');
  let drag=null, interactive=false;
  list.addEventListener('pointerdown',event=>{
    interactive=!!event.target.closest('input,select,textarea,button,a');
  });
  function clear() {
    if(!drag)return;
    cancelAnimationFrame(drag.frame);
    drag.card.classList.remove('drag-source');
    drag.placeholder.remove();
    list.classList.remove('sorting');
    drag=null;
  }
  function showPlaceholder() {
    if(!drag||drag.placeholder.isConnected)return;
    drag.card.before(drag.placeholder);
    drag.card.classList.add('drag-source');
    list.classList.add('sorting');
  }
  list.addEventListener('dragstart',event=>{
    const card=event.target.closest('.step');
    if(!card||interactive){event.preventDefault();return;}
    clear();
    const placeholder=document.createElement('div');
    placeholder.className='step-placeholder';
    placeholder.style.height=`${card.getBoundingClientRect().height}px`;
    placeholder.textContent='松开鼠标，插入此处';
    placeholder.setAttribute('aria-hidden','true');
    drag={card,placeholder,id:plan.steps[Number(card.dataset.index)].id,frame:null};
    event.dataTransfer.effectAllowed='move';
    event.dataTransfer.setData('text/plain',drag.id);
    // Let the browser capture the original card as its drag image first.
    drag.frame=requestAnimationFrame(showPlaceholder);
  });
  list.addEventListener('dragover',event=>{
    if(!drag)return;
    event.preventDefault();
    event.dataTransfer.dropEffect='move';
    showPlaceholder();
    const gap=drag.placeholder.getBoundingClientRect();
    // Keeping the existing gap under the pointer avoids alternating slots.
    if(event.clientY>=gap.top&&event.clientY<=gap.bottom)return;
    const next=[...list.querySelectorAll('.step:not(.drag-source)')].find(card=>{
      const box=card.getBoundingClientRect();
      return event.clientY<box.top+box.height/2;
    });
    list.insertBefore(drag.placeholder,next||null);
  });
  list.addEventListener('drop',event=>{
    if(!drag)return;
    event.preventDefault();
    showPlaceholder();
    const children=[...list.children];
    const position=children.slice(0,children.indexOf(drag.placeholder))
      .filter(node=>node.matches('.step:not(.drag-source)')).length;
    const original=plan.steps.findIndex(step=>step.id===drag.id);
    clear();
    if(original<0)return;
    selected=position;
    if(original!==position){
      plan.steps.splice(position,0,plan.steps.splice(original,1)[0]);
      changed();
    }
    renderSteps();
  });
  // Escape and dropping outside the list only remove the visual preview.
  list.addEventListener('dragend',clear);
}
function renderModels() {
  const numeric=(m,key,label,fallback,min=0)=>`<label>${label}<input data-model-field="${key}" type="number" min="${min}" step="any" value="${m[key]??fallback}"></label>`;
  $('model-editor').innerHTML=plan.models.map(m=>{
    const inference=m.inference_mode||(m.rtc===false?'sync':'rtc');
    const openpi=m.backend==='openpi';
    return `<div class="model-card" data-model="${esc(m.id)}"><div class="section-head"><strong>${esc(m.name)}</strong><div class="toolbar"><button data-model-load>预加载</button><button data-model-unload>卸载</button><button data-model-delete>删除模型</button></div></div>
      <div class="form-grid"><label>名称<input data-model-field="name" value="${esc(m.name)}"></label><label>后端<select data-model-field="backend">${options([['lerobot','LeRobot / PyTorch'],['openpi','OpenPI / JAX'],['mock','模拟（仅离线测试）']],m.backend)}</select></label><label>checkpoint 路径<input data-model-field="checkpoint" value="${esc(m.checkpoint||'')}"></label><label>动作模式<select data-model-field="action_mode">${options(Object.entries(modeNames),m.action_mode)}</select></label>${numeric(m,'policy_hz','policy Hz（动作队列取出频率）',10,.001)}</div>
      <details><summary>设备、RTC 与运控高级参数</summary><div class="form-grid">
      <label>设备<input data-model-field="device" value="${esc(m.device||'cuda:0')}"></label><label>相机安装版本 mount_id<input data-model-field="mount_id" value="${esc(m.mount_id||'')}"></label>
      ${numeric(m,'control_hz','control Hz（本模型运控频率）',plan.settings.control_hz,.001)}
      ${numeric(m,'smoothing_time_constant','时间平滑常数 s（0 关闭）',.1)}
      <label>推理方式<select data-model-field="inference_mode">${options([['rtc','异步 RTC'],['sync','同步 chunk']],inference)}</select></label>
      <label>RTC 模式<select data-model-field="rtc_mode">${options([['guided','Guided RTC'],['trained','Trained RTC（需对应训练权重）']],m.rtc_mode||'guided')}</select></label>
      ${numeric(m,'rtc_horizon','RTC horizon（前缀覆盖步数）',20,1)}
      ${numeric(m,'rtc_queue_threshold','RTC queue threshold（剩余动作触发阈值）',40)}
      ${openpi?`${numeric(m,'sample_steps','去噪采样步数',10,1)}${numeric(m,'n_action_steps','同步 chunk 复用步数（0 跟随模型）',0)}<label>RTC schedule<select data-model-field="rtc_schedule">${options([['exp','exp'],['linear','linear'],['ones','ones'],['zeros','zeros']],m.rtc_schedule||'exp')}</select></label>${numeric(m,'rtc_max_guidance_weight','RTC 最大引导权重',5)}<label class="check"><input data-model-field="rtc_mask_grippers" type="checkbox" ${m.rtc_mask_grippers!==false?'checked':''}>RTC 屏蔽夹爪</label>`:''}
      </div><p class="muted">policy Hz 只控制取动作的频率；control Hz 控制本 VLA 步骤下发频率。时间平滑常数沿用部署模块的一阶滤波，0 关闭；与切入 VLA 的衔接时间独立。修改模型参数后，执行时会按新配置重新加载。</p></details><label>默认任务指令<textarea data-model-field="task">${esc(m.task||'')}</textarea></label></div>`;
  }).join('')||'<div class="empty">只使用关键点时，无需添加模型。</div>';
}
function renderSettings() {
  const labels={control_hz:'运控 Hz',joint_speed:'关节最大速度 rad/s',joint_acceleration:'关节最大加速度 rad/s²',linear_speed:'末端速度 m/s',angular_speed:'末端角速度 rad/s',gripper_speed:'夹爪速度 m/s',transition_s:'VLA 进入衔接时间 s',prefetch_s:'提前准备下一 VLA 动作 s'};
  $('settings').innerHTML=`<label>关节下发接口<select data-setting="joint_command">${options([['move_js','move_js · 透传跟随（默认）'],['move_j','move_j · 固件规划']],plan.settings.joint_command||'move_js')}</select><small>作用于关节关键点和关节型 VLA。仍按运控 Hz 下发插值目标；move_j 会额外经过固件规划，不保证更快。末端运动与回零接口保持原样。</small></label>`+
    Object.entries(plan.settings).filter(([key])=>key!=='joint_command').map(([key,value])=>`<label>${esc(labels[key]||key)}<input data-setting="${esc(key)}" type="number" min=".001" step="any" value="${value}"></label>`).join('');
}
function highlight() {document.querySelectorAll('.step').forEach(el=>{el.classList.toggle('selected',Number(el.dataset.index)===selected);el.classList.toggle('running',live.busy&&plan.steps[Number(el.dataset.index)]?.id===live.step_id);});}
function renderState(state) {
  const limitsChanged=JSON.stringify(live.joint_limits)!==JSON.stringify(state.joint_limits);
  live=state;highlight();if(limitsChanged&&plan)renderPoints();
  const labels={idle:'待机',running:'执行中',paused:'暂停',completed:'已完成',stopped:'已停止',failed:'执行失败',emergency:'急停'};
  $('run-state').textContent=labels[state.state]||state.state;
  $('run-detail').textContent=state.detail||'';
  $('run-progress').value=state.progress||0;
  $('connection-state').textContent=state.connected?`${state.simulated?'模拟器':'真实从臂'}已连接${state.teaching?' · 连续录点中':''}`:'未连接设备';
  const available=compatible&&state.connected&&!state.busy&&!state.emergency;
  $('teach').textContent=state.teaching?'结束录点':'开始连续录点';
  $('connect').disabled=!compatible||state.connected||state.busy;
  $('connection-mode').disabled=state.connected;
  if(state.connected)$('connection-mode').value=state.simulated?'simulation':'physical';
  for(const id of ['left-channel','right-channel'])$(id).disabled=state.connected;
  $('disconnect').disabled=!state.connected||state.busy;
  for(const id of ['capture','teach','home','recover','telemetry','update-feedback'])$(id).disabled=!available;
  $('capture-side').disabled=!!state.teaching||!!state.busy;
  for(const id of ['run','run-from'])$(id).disabled=!available||!plan?.steps.length;
  $('pause').disabled=!state.busy||state.state==='paused'||['home','preparing'].includes(state.operation);
  $('pause').hidden=state.state==='paused';$('resume').hidden=state.state!=='paused';
  $('resume').disabled=state.state!=='paused';
  $('next').hidden=!state.busy||['home','preparing'].includes(state.operation)||!['vla','wait'].includes(state.step_type);
  $('stop').disabled=!state.busy;$('emergency').disabled=!state.connected;
  $('preload').disabled=!compatible||state.busy;$('unload').disabled=!compatible||state.busy;
  $('inline-error').textContent=actionError||state.error||'';$('inline-error').hidden=!$('inline-error').textContent;
  $('model-progress-card').hidden=!(state.models?.length||plan?.models.length);
  $('next-hint').textContent=!compatible?'请先重启工作流服务，使用更新后的工作流功能。':
    !state.connected?'下一步：连接从臂；也可以选择模拟器熟悉操作。':
    state.emergency?'已急停：在设备侧检查并复位，然后重新连接。急停不会回零。':
    state.busy?(['home','preparing'].includes(state.operation)?`${state.detail||'正在准备运动'}；需要中断时点击「停止并保持」。`:'正在执行，右侧显示当前步骤和模型状态。'):
    state.teaching?`连续录点：${(state.teaching_sides||[]).map(s=>sideName[s]).join('、')}。通过机身示教按钮拖动并录制；网页不释放电机。退出机身示教后，执行会自动准备回零。`:
    !plan?.points.length?'下一步：选择从臂，用机身示教按钮拖动后录制当前位置。':
    !plan.steps.length?'下一步：在关键点列表点击「添加步骤」，同一点可以重复添加。':'下一步：在编排页确认顺序，保存工作流，点击右侧「执行工作流」。';
  document.querySelectorAll('[data-point-action="test"], [data-point-action="recapture"]').forEach(b=>b.disabled=!available);
  const openLogs=new Set([...$('model-status').querySelectorAll('details[open]')].map(d=>d.closest('[data-model-state]').dataset.modelState));
  $('model-status').innerHTML=(state.models||[]).map(m=>{
    const stages=['starting','loading','warming','ready'],index=stages.indexOf(m.stage);
    const label={starting:'启动进程',loading:'加载权重',warming:'预热推理',ready:'已就绪',failed:'失败',unloaded:'已卸载'}[m.stage]||m.stage;
    return `<div class="model-status-row ${esc(m.stage)}" data-model-state="${esc(m.id)}"><strong>${esc(m.name)}</strong> <span class="badge">${label}</span><div class="stages">${stages.map((_,i)=>`<span class="${i<index||index===3?'done':i===index?'current':''}"></span>`).join('')}</div><small>${esc(m.detail)} · ${m.elapsed}s</small>${m.logs.length?`<details ${openLogs.has(m.id)?'open':''}><summary>查看日志</summary><pre>${esc(m.logs.join('\n'))}</pre></details>`:''}</div>`;
  }).join('')||'还没有加载模型';
  $('events').innerHTML=(state.events||[]).slice().reverse().map(e=>`<p>${new Date(e.time*1000).toLocaleTimeString()} · ${esc(e.message)}</p>`).join('')||'执行记录将在这里显示';
  const v=state.verification;
  if(v) {
    let detail='';
    for(const [side,o] of [['left',0],['right',7]]) detail+=v.space==='joint'?`${sideName[side]}关节最大误差 ${(Math.max(...v.error.slice(o,o+6))*180/Math.PI).toFixed(2)}°；`:`${sideName[side]}位置误差 ${(Math.hypot(...v.error.slice(o,o+3))*1000).toFixed(1)} mm；`;
    const open=$('verification').querySelector('details')?.open;
    const rows=['left','right'].map((side,k)=>Array.from({length:7},(_,i)=>{
      const index=k*7+i,scale=i===6||v.space==='pose'&&i<3?1000:180/Math.PI;
      const axis=i===6?'夹爪 mm':v.space==='joint'?`J${i+1} °`:['X mm','Y mm','Z mm','Roll °','Pitch °','Yaw °'][i];
      return `<tr><td>${sideName[side]} ${axis}</td><td>${(v.target[index]*scale).toFixed(2)}</td><td>${(v.actual[index]*scale).toFixed(2)}</td><td>${(v.error[index]*scale).toFixed(2)}</td></tr>`;
    }).join('')).join('');
    $('verification').innerHTML=`<p>${v.simulated?'模拟反馈':'实际反馈'}：${v.reached?'已到位':'尚未到位'}。${detail}</p><details ${open?'open':''}><summary>目标 / 实际 / 误差明细</summary><table><thead><tr><th>轴</th><th>目标</th><th>实际</th><th>误差</th></tr></thead><tbody>${rows}</tbody></table></details>`;
  } else $('verification').textContent='';
}
async function save(copy=false) { const result=await api('save',{plan,id:copy?null:savedId});savedId=result.id;renderSaved(result.workflows);changed();dirty=false;$('dirty').textContent='已保存';toast('工作流已保存'); }
async function storePoint(point) { const saved=await api('point/save',{point});mergePoint(saved);return saved; }
function mergePoint(point) {const i=plan.points.findIndex(p=>p.id===point.id);if(i<0)plan.points.push(point);else plan.points[i]=point;library=library.filter(p=>p.id!==point.id).concat([point]);changed();renderPoints();renderSteps();}
function openPoint(point=null) {
  editingPoint=point?structuredClone(point):{id:uid(),name:'新关键点',notes:'',space:'joint',arms:{left:[0,.2,-.2,0,0,-.785,.04],right:[0,.2,-.2,0,0,.785,.04]}};
  $('point-name').value=editingPoint.name;$('point-notes').value=editingPoint.notes||'';$('point-space').value=editingPoint.space;
  $('point-side').value=Object.keys(editingPoint.arms).length===2?'both':Object.keys(editingPoint.arms)[0];
  const count=plan.steps.filter(s=>s.point===editingPoint.id).length;
  $('point-references').textContent=count?`当前工作流有 ${count} 个步骤引用此点，保存后这些步骤都会使用新坐标。`:'此点可在多个步骤中复用。';
  renderCoordinates();$('point-coordinates').open=!point;$('point-dialog').showModal();
}
function renderCoordinates() {
  const space=$('point-space').value,side=$('point-side').value;
  $('coordinate-fields').innerHTML=['left','right'].filter(s=>side==='both'||side===s).map(s=>{
    const value=editingPoint.space===space?editingPoint.arms[s]:null;
    const labels=space==='joint'?['J1 °','J2 °','J3 °','J4 °','J5 °','J6 °','夹爪 mm']:['X mm','Y mm','Z mm','Roll °','Pitch °','Yaw °','夹爪 mm'];
    return `<h3>${sideName[s]}</h3><div class="coordinates">${labels.map((label,i)=>{
      const scale=i===6||space==='pose'&&i<3?1000:180/Math.PI;
      return `<label>${label}<input data-coordinate="${s}:${i}" type="number" step="any" required value="${((value?.[i]||0)*scale).toFixed(3)}"></label>`;
    }).join('')}</div>`;
  }).join('');
}
function readPoint() {
  const space=$('point-space').value,arms={};
  let coordinatesChanged=space!==editingPoint.space;
  document.querySelectorAll('[data-coordinate]').forEach(input=>{
    const [side,index]=input.dataset.coordinate.split(':'),i=Number(index),scale=i===6||space==='pose'&&i<3?1000:180/Math.PI;
    const original=editingPoint.space===space?editingPoint.arms[side]?.[i]:undefined;
    const unchanged=original!==undefined && Number(input.value)===Number((original*scale).toFixed(3));
    arms[side]??=Array(7).fill(0);arms[side][i]=unchanged?original:Number(input.value)/scale;
    coordinatesChanged ||= !unchanged;
  });
  coordinatesChanged ||= Object.keys(arms).sort().join()!==Object.keys(editingPoint.arms).sort().join();
  return {...editingPoint,name:$('point-name').value.trim(),notes:$('point-notes').value,space,arms,
    source:coordinatesChanged?{kind:'manual'}:editingPoint.source};
}
function renderLibrary() {const query=$('library-search').value.toLowerCase();$('library-list').innerHTML=library.filter(p=>`${p.name} ${p.notes||''}`.toLowerCase().includes(query)).map(p=>`<div class="point-row"><div><strong>${esc(p.name)}</strong><p>${esc(p.notes||'')} · ${p.space==='joint'?'关节角':'末端位姿'}</p></div><div class="toolbar"><button data-library-import="${esc(p.id)}">导入快照</button><button data-library-delete="${esc(p.id)}">从共享库删除</button></div></div>`).join('')||'<p class="muted">还没有共享关键点</p>';}
async function run(start=0, override=null) {await api('run',{plan:override||plan,start,repetitions:override?1:Number($('repetitions').value),home_before:!override});toast(live.simulated?'模拟工作流已开始':'真实机械臂工作流已开始');await poll();}
async function poll() {try{renderState(await api('state'));}catch(e){$('run-detail').textContent=`服务连接中断：${e.message}`;}}
function triple(id) {const a=$(id).value.split(/[,，\s]+/).filter(Boolean).map(Number);if(a.length!==3||a.some(x=>!Number.isFinite(x)))throw new Error('XYZ 偏移须为 3 个数值');return a.map(v=>v/1000);}
async function init() {
  const boot=await api('bootstrap');compatible=boot.api_version===4;$('version-notice').hidden=compatible;plan=boot.plan;library=boot.points;renderSaved(boot.workflows);
  const draft=localStorage.getItem('workflowDraft');if(draft){try{const parsed=JSON.parse(draft);plan=await api('validate',{plan:parsed.plan});savedId=parsed.savedId;dirty=true;$('dirty').textContent='已恢复本地草稿';}catch{toast('旧草稿无法恢复，已新建工作流',true);}}
  for(const [role,serial] of Object.entries(boot.serials)) $(`serial-${role}`).value=serial;
  $('calibrations').innerHTML=boot.calibrations.map(p=>`<option value="${esc(p)}"></option>`).join('');if(boot.calibrations.length)$('calibration').value=boot.calibrations.at(-1);
  renderSaved(boot.workflows);render();renderState(boot.state);
  document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>tab(b.dataset.tab));
  $('plan-name').oninput=()=>{plan.name=$('plan-name').value;changed();};
  $('save').onclick=act(()=>save());$('save-copy').onclick=act(()=>save(true));
  $('new').onclick=act(async()=>{if(dirty&&!confirm('新建会替换当前草稿，继续？'))return;plan=(await api('bootstrap')).plan;savedId=null;changed();render();});
  $('saved').onchange=act(async()=>{if(!$('saved').value)return;if(dirty&&!confirm('打开会替换当前草稿，继续？'))return;savedId=$('saved').value;plan=await api(`workflow?id=${savedId}`);selected=0;changed();render();});
  $('export').onclick=()=>{const link=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(plan,null,2)],{type:'application/json'}));link.href=url;link.download=`${plan.name||'workflow'}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  $('import').onchange=act(async()=>{const file=$('import').files[0];if(!file)return;const imported=await api('validate',{plan:JSON.parse(await file.text())});if(dirty&&!confirm('导入会替换当前草稿，继续？'))return;plan=imported;savedId=null;selected=0;changed();render();});
  $('connect').onclick=act(async()=>{const serials=Object.fromEntries(['agentview','left_eye_in_hand','right_eye_in_hand'].map(r=>[r,$(`serial-${r}`).value]));renderState(await api('connect',{simulated:$('connection-mode').value==='simulation',config:{channels:{left:$('left-channel').value,right:$('right-channel').value},serials,depth_roles:$('depth').checked?['agentview']:[]}}));});
  $('disconnect').onclick=act(async()=>{renderState(await api('disconnect',{}));frozen=null;pixel=null;});
  $('teach').onclick=act(async()=>{
    renderState(await api('teach',{enabled:!live.teaching,side:$('capture-side').value}));
  });
  $('home').onclick=act(async()=>{renderState(await api('home',{close_grippers:$('home-grippers').checked}));});
  $('recover').onclick=act(async()=>{
    renderState(await api('recover',{}));
  });
  $('go-flow').onclick=()=>tab('flow');
  $('telemetry').onclick=act(async()=>{const s=await api('telemetry');$('feedback').textContent=['left','right'].map((side,i)=>`${sideName[side]} J1–J6：${s.joints.slice(i*7,i*7+6).map(v=>(v*180/Math.PI).toFixed(1)).join(' / ')}°，夹爪 ${(s.joints[i*7+6]*1000).toFixed(1)} mm`).join('  ｜  ');});
  $('capture').onclick=act(async()=>{mergePoint(await api('capture',{name:$('capture-name').value||`关键点 ${plan.points.length+1}`,side:$('capture-side').value,space:$('capture-space').value}));$('capture-name').value='';toast('已录制并保存；可继续拖动录下一个点');});
  $('manual-point').onclick=()=>openPoint();$('point-search').oninput=renderPoints;
  $('point-space').onchange=renderCoordinates;$('point-side').onchange=renderCoordinates;
  $('close-point').onclick=()=>$('point-dialog').close();
  $('point-form').onsubmit=act(async event=>{event.preventDefault();await storePoint(readPoint());$('point-dialog').close();toast('关键点已保存');});
  $('update-feedback').onclick=act(async()=>{const p=await api('capture',{persist:false,name:$('point-name').value,space:$('point-space').value,side:$('point-side').value});editingPoint={...editingPoint,space:p.space,arms:p.arms,source:p.source};renderCoordinates();toast('已读取当前坐标，保存后生效');});
  $('points').onclick=act(async event=>{const button=event.target.closest('[data-point-action]');if(!button)return;const id=button.closest('[data-point]').dataset.point,p=plan.points.find(p=>p.id===id);switch(button.dataset.pointAction){case'add':addStep('point',id);break;case'edit':openPoint(p);break;case'recapture':{const side=Object.keys(p.arms).length===2?'both':Object.keys(p.arms)[0];const sample=await api('capture',{persist:false,name:p.name,side,space:p.space});await storePoint({...p,arms:sample.arms,source:sample.source});toast('已用当前位置重录；名称和步骤引用保留，请核对角度后保存工作流');break;}case'copy':await storePoint({...structuredClone(p),id:uid(),name:`${p.name} 副本`});break;case'delete':{const refs=plan.steps.filter(s=>s.point===id).length;if(!confirm(`从当前流程移除「${p.name}」${refs?`以及引用它的 ${refs} 个步骤`:''}？共享点库保留。`))return;plan.points=plan.points.filter(p=>p.id!==id);plan.steps=plan.steps.filter(s=>s.point!==id);changed();render();break;}case'test':{const test=structuredClone(plan);test.steps=[{id:uid(),type:'point',point:id,name:`试运行 ${p.name}`,duration:3,blend:false}];test.models=[];await run(0,test);break;}}});
  $('import-library').onclick=act(async()=>{library=(await api('bootstrap')).points;renderLibrary();$('library-dialog').showModal();});$('library-search').oninput=renderLibrary;$('close-library').onclick=()=>$('library-dialog').close();
  $('library-list').onclick=act(async event=>{const imp=event.target.closest('[data-library-import]'),del=event.target.closest('[data-library-delete]');if(imp){const p=structuredClone(library.find(p=>p.id===imp.dataset.libraryImport));if(plan.points.some(x=>x.id===p.id)&&!confirm('当前流程已有此点，使用点库坐标更新？'))return;mergePoint(p);toast('已导入到当前工作流');}if(del&&confirm('从共享点库删除？已保存工作流中的快照仍保留。')){library=(await api('point/delete',{id:del.dataset.libraryDelete})).points;renderLibrary();}});
  $('hdf-import').onclick=act(async()=>{mergePoint(await api('from-hdf5',{path:$('hdf-path').value,frame:Number($('hdf-frame').value),name:$('capture-name').value||'轨迹点',space:$('capture-space').value,side:$('capture-side').value}));toast('已提取关键点，源文件未修改');});
  $('add-point-step').onclick=act(()=>addStep('point',$('add-point-select').value));for(const type of ['vla','gripper','wait'])$(`add-${type}`).onclick=act(()=>addStep(type));
  $('steps').addEventListener('focusin', event=>{
    const input=event.target;
    if(!input.matches('input[type="number"][data-field="duration"]'))return;
    const value=input.value;
    // Number inputs have no setSelectionRange. Reassign after the initial
    // click places the caret at the end, preserving native numeric controls.
    requestAnimationFrame(()=>{
      if(document.activeElement===input&&input.value===value){input.value='';input.value=value;}
    });
  });
  $('steps').onclick=event=>{const card=event.target.closest('[data-index]');if(!card)return;selected=Number(card.dataset.index);highlight();const action=event.target.closest('[data-step-action]')?.dataset.stepAction;if(!action)return;const i=selected;if(action==='delete')plan.steps.splice(i,1);if(action==='copy')plan.steps.splice(i+1,0,{...structuredClone(plan.steps[i]),id:uid()});if(action==='up'&&i>0){[plan.steps[i-1],plan.steps[i]]=[plan.steps[i],plan.steps[i-1]];selected--;}if(action==='down'&&i<plan.steps.length-1){[plan.steps[i+1],plan.steps[i]]=[plan.steps[i],plan.steps[i+1]];selected++;}changed();renderSteps();renderPoints();};
  $('steps').onchange=event=>{const input=event.target.closest('[data-field]');if(!input)return;const s=plan.steps[Number(input.closest('[data-index]').dataset.index)],key=input.dataset.field;s[key]=input.type==='checkbox'?input.checked:input.type==='number'?Number(input.value)/(key==='width'?1000:1):input.value;changed();if(['until','point','model'].includes(key)){renderSteps();renderPoints();}};
  installStepSorting();
  $('settings').onchange=e=>{if(e.target.dataset.setting){plan.settings[e.target.dataset.setting]=e.target.dataset.setting==='joint_command'?e.target.value:Number(e.target.value);changed();}};
  $('add-model').onclick=()=>{plan.models.push({id:uid(),name:`模型 ${plan.models.length+1}`,backend:'lerobot',checkpoint:'',action_mode:'delta_joint',policy_hz:10,device:'cuda:0',rtc:true,task:'',mount_id:''});changed();renderModels();renderSteps();};
  $('model-editor').onchange=e=>{const key=e.target.dataset.modelField;if(!key)return;const m=plan.models.find(m=>m.id===e.target.closest('[data-model]').dataset.model);m[key]=e.target.type==='checkbox'?e.target.checked:e.target.type==='number'?Number(e.target.value):e.target.value;changed();if(['name','backend'].includes(key)){renderModels();renderSteps();}};
  $('model-editor').onclick=act(async e=>{const card=e.target.closest('[data-model]');if(!card)return;if(e.target.closest('[data-model-load]')){await api('preload',{plan,ids:[card.dataset.model]});await poll();return;}if(e.target.closest('[data-model-unload]')){await api('unload',{id:card.dataset.model});await poll();return;}if(!e.target.closest('[data-model-delete]'))return;const id=e.target.closest('[data-model]').dataset.model,refs=plan.steps.filter(s=>s.model===id).length;if(!confirm(`删除模型配置${refs?`和引用它的 ${refs} 个步骤`:''}？`))return;plan.models=plan.models.filter(m=>m.id!==id);plan.steps=plan.steps.filter(s=>s.model!==id);changed();renderModels();renderSteps();});
  $('preload').onclick=act(async()=>{await api('preload',{plan});await poll();});$('unload').onclick=act(async()=>{await api('unload',{});await poll();});
  $('run').onclick=act(()=>run());$('run-from').onclick=act(()=>run(selected));for(const cmd of ['pause','resume','next','stop','emergency'])$(cmd).onclick=act(async()=>{await api(cmd,{});await poll();});
  $('live-camera').onclick=()=>{frozen=null;pixel=null;clearInterval(previewTimer);previewTimer=setInterval(()=>{if(!frozen&&live.connected&&!$('tab-camera').hidden&&$('camera-image').complete)$('camera-image').src=`api/frame?camera=${$('camera-role').value}&t=${Date.now()}`;},500);$('pick-marker').hidden=true;$('camera-image').src=`api/frame?camera=${$('camera-role').value}&t=${Date.now()}`;$('pixel-info').textContent='预览画面：请先冻结再选点。';};
  $('camera-image').onerror=()=>toast('相机帧尚未就绪，请稍后刷新；确认设备已连接且相机可用',true);
  $('freeze').onclick=act(async()=>{clearInterval(previewTimer);frozen=await api('snapshot',{camera:$('camera-role').value});pixel=null;$('pick-marker').hidden=true;$('camera-image').src=frozen.image;$('pixel-info').textContent=frozen.has_depth?'画面已冻结，点击物体表面选择目标。':'画面已冻结，但此相机没有对齐深度。';});
  $('camera-image').onclick=e=>{if(!frozen||!frozen.has_depth){toast('请先冻结具有对齐深度的画面',true);return;}const r=e.target.getBoundingClientRect();pixel=[Math.min(frozen.camera.width-1,Math.floor((e.clientX-r.left)/r.width*frozen.camera.width)),Math.min(frozen.camera.height-1,Math.floor((e.clientY-r.top)/r.height*frozen.camera.height))];$('pixel-info').textContent=`已选择像素 (${pixel.join(', ')})；保存目标点后可在点库中试运行并查看误差。`;$('pick-marker').style.left=`${(e.clientX-r.left)/r.width*100}%`;$('pick-marker').style.top=`${(e.clientY-r.top)/r.height*100}%`;$('pick-marker').hidden=false;};
  $('save-pick').onclick=act(async()=>{if(!frozen||!pixel)throw new Error('请先冻结并点击目标像素');mergePoint(await api('pick',{snapshot:frozen.id,pixel,calibration:$('calibration').value,side:$('pick-side').value,tcp_offset:triple('tcp-offset'),offset:triple('base-offset'),name:$('pick-name').value}));toast('选点已保存，可在关键点库中试运行');tab('points');});
  $('example').onclick=act(async()=>{if(dirty&&!confirm('用模拟示例替换当前草稿？'))return;plan=(await api('bootstrap')).plan;plan.name='模拟：基准点 → VLA → 返回基准点';const p={id:uid(),name:'双臂基准点',space:'joint',arms:{left:[0,.2,-.2,0,0,-.785,.04],right:[0,.2,-.2,0,0,.785,.04]},notes:'同一点在第 1、3 步重复引用'};plan.points=[p];const m={id:uid(),name:'模拟 VLA',backend:'mock',action_mode:'absolute_joint',policy_hz:10,task:'保持位置',rtc:true};plan.models=[m];plan.steps=[{id:uid(),type:'point',name:'进入基准点',point:p.id,duration:1,blend:true},{id:uid(),type:'vla',name:'交给 VLA',model:m.id,until:'duration',value:2},{id:uid(),type:'point',name:'返回同一基准点',point:p.id,duration:1,blend:false}];savedId=null;changed();render();tab('flow');toast('模拟示例已准备，请连接模拟器后执行');});
  window.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();act(()=>save())();}});
  setInterval(poll,700);
}
init().catch(error=>toast(error.message,true));
