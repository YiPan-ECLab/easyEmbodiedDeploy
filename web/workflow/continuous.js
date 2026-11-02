'use strict';
let continuousAvailable = false, continuousSwitchesAvailable = false, jointTriggersAvailable = false, jointPoseTriggersAvailable = false, highZTriggersAvailable = false;
function hasJointTriggers() { return continuousMode() && continuousConfig().takeovers.some(e=>['right_joint_sum','right_j6_pose','right_joint_sum_pose'].includes(e.trigger?.type)); }
function hasHighZTriggers() { return continuousMode() && continuousConfig().takeovers.some(e=>e.trigger?.high_z_mm!=null); }
function defaultHighZ(z=250) { return z<750?Math.min(750,Math.max(500,Number(z)+100)):null; }
function matchPathLabel(path) { return ({both:'两条分支均达标',sum_and_height:'角度＋高度联合分支',high_z:'高位独立分支'})[path]||'尚未达标'; }
function jointPoseTriggerDefaults() {
  return {type:'right_joint_sum_pose',end_s:25,sum_deg:30,z_mm:250,high_z_mm:500,hold_s:0.05};
}
function renderPoseTrigger(event) {
  const t=event.trigger;
  return `<div class="form-grid"><label>触发方式<select data-trigger-field="type"><option value="time">按时间接管</option><option value="right_joint_sum_pose" selected>右臂 J4＋J6 合计＋末端高度</option></select></label><label>允许触发的开始时间（运行秒数）<input data-event-field="at_s" type="number" min="0" step="0.1" value="${event.at_s}"></label><label>允许触发的结束时间（运行秒数）<input data-trigger-field="end_s" type="number" min="0" step="0.1" value="${t.end_s}"></label></div><div class="notice"><p>时间窗口内，「合计角度达标且高度达标」或「高度达到独立触发下限」，任一路成立并持续指定时间后执行本段一次。</p><div class="form-grid"><label>J4＋J6 合计角度至少（°）<input data-trigger-field="sum_deg" type="number" min="-220" max="220" step="1" value="${t.sum_deg}"></label><label>末端法兰高度 Z 至少（mm）<input data-trigger-field="z_mm" type="number" min="-700" max="750" step="10" value="${t.z_mm}"></label><label>触发条件持续（秒）<input data-trigger-field="hold_s" type="number" min="0.05" max="5" step="0.05" value="${t.hold_s}"></label></div><label class="check"><input data-trigger-field="high_z_enabled" type="checkbox" ${t.high_z_mm!=null?'checked':''}>或：末端足够高时独立触发，不要求角度达标</label>${t.high_z_mm!=null?`<div class="form-grid"><label>独立触发：法兰 Z 至少（mm）<input data-trigger-field="high_z_mm" type="number" min="${Number(t.z_mm)+1}" max="750" step="10" value="${t.high_z_mm}"></label></div>`:''}<p class="muted">按右从臂实际反馈直接计算 J4＋J6：有符号求和，不取绝对值，不使用相对基准，不分别限制 J4/J6；例如 J4=99°、J6=−68°，合计31°。Z 由同一份完整六轴反馈按 Piper 正运动学计算，原点为右臂基座，位置为法兰，不含夹爪 TCP 偏移。高位独立分支默认500 mm，可取消勾选关闭；它仅绕过角度条件，仍受同一时间窗口和持续确认限制。新建默认12～25秒、(J4＋J6 ≥ 30° 且 Z ≥ 250 mm) 或 Z ≥ 500 mm、持续0.05秒。超时跳过；暂停重新累计持续时间；固定段完成后交还 VLA。</p></div>`;
}
function renderTakeoverTrigger(event) {
  if(event.trigger?.type==='right_joint_sum_pose')return renderPoseTrigger(event);
  return `<div class="form-grid"><label>触发方式<select data-trigger-field="type"><option value="time" selected>按时间接管</option><option value="right_joint_sum_pose" ${!jointPoseTriggersAvailable?'disabled':''}>右臂 J4＋J6 合计＋末端高度</option></select></label><label>接管时间（运行秒数）<input data-event-field="at_s" type="number" min="0" step="0.1" value="${event.at_s}"></label></div>`;
}
function renderTakeoverConditions(state) {
  const target=$('takeover-condition-state');if(!target)return;
  const rows=(state.takeover_conditions||[]).filter(s=>s.baseline_deg||['right_j6_pose','right_joint_sum_pose'].includes(s.type)||s.state==='expired');
  target.hidden=!rows.length;
  const names={watching:'等待角度条件',matching:'条件持续确认中',triggered:'已触发',expired:'已超时跳过'};
  target.innerHTML=rows.map(s=>`<p><strong>${esc(s.name||'关节接管')}</strong> · ${names[s.state]||esc(s.state)}${s.type==='right_joint_sum_pose'?`<br>J4 ${s.j4_deg.toFixed(1)}° ＋ J6 ${s.j6_deg.toFixed(1)}° ＝ ${s.sum_deg.toFixed(1)}° / 阈值 ${s.sum_threshold_deg.toFixed(1)}°；法兰 Z ${s.z_mm.toFixed(1)} mm / 阈值 ${s.z_threshold_mm.toFixed(1)} mm${s.high_z_threshold_mm!=null?`；或 Z ≥ ${s.high_z_threshold_mm.toFixed(1)} mm`:""} · ${s.matched_by?matchPathLabel(s.matched_by):"等待条件"} · 已持续 ${(s.held_s||0).toFixed(2)} 秒`:''}${s.type==='right_j6_pose'?`<br>J6 ${s.j6_deg.toFixed(1)}° / 阈值 ${s.j6_threshold_deg.toFixed(1)}°；法兰 Z ${s.z_mm.toFixed(1)} mm / 阈值 ${s.z_threshold_mm.toFixed(1)} mm · 已持续 ${(s.held_s||0).toFixed(2)} 秒`:''}${s.delta_deg?`<br>J4 转角 ${s.delta_deg[0].toFixed(1)}° ＋ J6 转角 ${s.delta_deg[1].toFixed(1)}° ＝ ${(s.sum_deg??0).toFixed(1)}° · 已持续 ${(s.held_s||0).toFixed(2)} 秒<br><small>实际角度 J4 ${(s.j4_deg??0).toFixed(1)}° / J6 ${s.j6_deg.toFixed(1)}°；基准 ${s.baseline_deg.map(a=>a.toFixed(1)+'°').join(' / ')}</small>`:''}</p>`).join('');
}
function continuousConfig() {
  const config=plan.continuous_vla ??= {model:'',task:'',side:'both',takeovers:[],switches:[]};
  config.switches ??= [];
  for(const event of config.takeovers){
    if(highZTriggersAvailable && event.trigger?.type==='right_joint_sum_pose' && event.trigger.high_z_mm===undefined){
      event.trigger.high_z_mm=defaultHighZ(event.trigger.z_mm);
    }
    // Upgrade editable plans/drafts, not immutable historical run recordings.
    if(jointPoseTriggersAvailable && ['right_joints','right_joint_sum','right_j6_pose'].includes(event.trigger?.type)){
      const old=event.trigger;
      event.trigger={...jointPoseTriggerDefaults(),end_s:old.end_s??25,
        hold_s:old.hold_s??0.05,z_mm:old.z_mm??250,high_z_mm:defaultHighZ(old.z_mm??250)};
      event.at_s ??= 12;
    }
  }
  return config;
}
function continuousMode() { return plan?.mode === 'continuous_vla'; }
function allStepReferences() { return [...plan.steps,...(plan.continuous_vla?.takeovers||[]).flatMap(e=>e.steps)]; }
function activeModelIds() {
  return continuousMode() ? [...new Set([continuousConfig().model,...continuousConfig().switches.map(s=>s.model)].filter(Boolean))] : [...new Set(plan.steps.filter(s=>s.type==='vla').map(s=>s.model))];
}
function renderContinuous() {
  const config=continuousConfig(), enabled=continuousMode();
  $('execution-mode').value=plan.mode||'sequential';
  $('continuous-editor').hidden=!enabled;$('sequential-editor').hidden=enabled;
  $('continuous-model').innerHTML='<option value="">请选择起始 VLA</option>'+modelOptions(config.model);
  $('continuous-side').value=config.side||'both';$('continuous-task').value=config.task||'';
  if(!enabled)return;
  $('step-count').textContent=`接管动作段：${config.takeovers.length} · 模型切换：${config.switches.length}`;
  renderModelSwitches();
  $('takeovers').innerHTML=config.takeovers.map((event,index)=>`<article class="step takeover" data-event="${esc(event.id)}">
    <label>动作段名称<input data-event-field="name" value="${esc(event.name||`动作段 ${index+1}`)}"></label>
    ${renderTakeoverTrigger(event)}
    ${event.steps.map((step,i)=>`<div class="takeover-point" data-point-index="${i}"><div class="form-grid"><label>关键点<select data-takeover-field="point">${pointOptions(step.point)}</select></label><label>期望运动时长（秒）<input data-takeover-field="duration" type="number" min="0.05" step="0.1" value="${step.duration}"></label><label class="check"><input data-takeover-field="blend" type="checkbox" ${step.blend!==false?'checked':''}>与下一个关节点平滑衔接</label></div><div class="toolbar"><button data-takeover-action="up">上移</button><button data-takeover-action="down">下移</button><button data-takeover-action="remove-point">移除此点</button></div></div>`).join('')}
    <div class="toolbar"><select data-new-point aria-label="动作段 ${index+1} 要添加的关键点">${pointOptions(plan.points[0]?.id)}</select><button data-takeover-action="add-point">添加关键点</button><button data-takeover-action="remove-event">删除动作段</button></div>
    <p class="muted">这一段所有关键点到位后，自动交还届时选中的 VLA。</p></article>`).join('')||'<p class="muted">暂未设置定时动作。执行后 VLA 将持续控制，直到点击停止。</p>';
}
function initContinuous(boot) {
  continuousAvailable=!!boot.features?.continuous_vla;
  continuousSwitchesAvailable=!!boot.features?.continuous_vla_switches;
  jointPoseTriggersAvailable=!!boot.features?.joint_sum_pose_takeover_triggers;
  highZTriggersAvailable=!!boot.features?.high_z_override_takeover_triggers;
  jointTriggersAvailable=!!boot.features?.joint_sum_takeover_triggers;
  $('add-model-switch').disabled=!continuousSwitchesAvailable;
  $('execution-mode').querySelector('[value="continuous_vla"]').disabled=!continuousAvailable;
  if(!continuousAvailable){$('version-notice').hidden=false;$('version-notice').textContent='定时接管模式需要重启自定义工作流服务后使用；刷新页面不会自动断开已连接的设备。';}
  if(continuousAvailable&&!continuousSwitchesAvailable){$('version-notice').hidden=false;$('version-notice').textContent='多 VLA 定时切换需要重启自定义工作流服务后使用。';}
  initModelSwitches();
  $('execution-mode').onchange=()=>{
    plan.mode=$('execution-mode').value;
    if(continuousMode())continuousConfig().model ||= plan.models[0]?.id||'';
    changed();renderSteps();
  };
  for(const [id,key] of [['continuous-model','model'],['continuous-side','side'],['continuous-task','task']]){
    $(id).onchange=()=>{continuousConfig()[key]=$(id).value;changed();renderState(live);};
  }
  $('add-takeover').onclick=()=>{
    const config=continuousConfig();
    config.takeovers.push({id:uid(),name:`动作段 ${config.takeovers.length+1}`,at_s:12,trigger:jointPoseTriggerDefaults(),steps:plan.points.length?[{id:uid(),type:'point',point:plan.points[0].id,duration:1,blend:true}]:[]});
    changed();renderContinuous();
  };
  $('import-takeovers').onclick=()=>{
    const config=continuousConfig();
    if(config.takeovers.length&&!confirm('替换当前定时动作段？顺序编排保持不变。'))return;
    let time=0,group=null;config.takeovers=[];
    for(const step of plan.steps){
      if(step.type==='point'){
        if(!group){group={id:uid(),name:`固定动作段 ${config.takeovers.length+1}`,at_s:time,steps:[]};config.takeovers.push(group);}
        group.steps.push({...structuredClone(step),id:uid()});time+=Number(step.duration||1);
      }else{group=null;time+=step.type==='vla'&&step.until==='duration'?Number(step.value||0):Number(step.duration||0);}
    }
    changed();renderContinuous();toast('已导入固定动作段；请核对接管时间，模型切换仍按独立时间表执行');
  };
  $('takeovers').onchange=event=>{
    const card=event.target.closest('[data-event]');if(!card)return;
    const entry=continuousConfig().takeovers.find(e=>e.id===card.dataset.event),input=event.target;
    if(input.dataset.eventField)entry[input.dataset.eventField]=input.type==='number'?Number(input.value):input.value;
    if(input.dataset.triggerField){
      const key=input.dataset.triggerField;
      if(key==='type'){
        if(input.value==='right_joint_sum_pose'){entry.at_s=12;entry.trigger=jointPoseTriggerDefaults();}
        else entry.trigger={type:'time'};
      }else if(key==='high_z_enabled'){
        entry.trigger.high_z_mm=input.checked?defaultHighZ(entry.trigger.z_mm):null;
      }else{
        entry.trigger[key]=input.type==='number'||key.endsWith('_direction')?(input.value===''?'':Number(input.value)):input.value;
      }
      changed();
      if(['type','reference','high_z_enabled'].includes(key))renderContinuous();
      renderState(live);return;
    }
    if(input.dataset.takeoverField){
      const step=entry.steps[Number(input.closest('[data-point-index]').dataset.pointIndex)],key=input.dataset.takeoverField;
      step[key]=input.type==='checkbox'?input.checked:input.type==='number'?Number(input.value):input.value;
    }
    changed();
  };
  $('takeovers').onclick=event=>{
    const button=event.target.closest('[data-takeover-action]');if(!button)return;
    const card=button.closest('[data-event]'),config=continuousConfig(),entry=config.takeovers.find(e=>e.id===card.dataset.event);
    const index=Number(button.closest('[data-point-index]')?.dataset.pointIndex),action=button.dataset.takeoverAction;
    if(action==='remove-event')config.takeovers=config.takeovers.filter(e=>e!==entry);
    if(action==='remove-point')entry.steps.splice(index,1);
    if(action==='up'&&index>0)[entry.steps[index-1],entry.steps[index]]=[entry.steps[index],entry.steps[index-1]];
    if(action==='down'&&index<entry.steps.length-1)[entry.steps[index+1],entry.steps[index]]=[entry.steps[index],entry.steps[index+1]];
    if(action==='add-point'){
      const point=card.querySelector('[data-new-point]').value;if(!point){toast('请先录制或新增关键点',true);return;}
      entry.steps.push({id:uid(),type:'point',point,duration:1,blend:true});
    }
    changed();renderContinuous();renderPoints();
  };
}

function renderModelSwitches() {
  const config=continuousConfig();
  $('model-switches').innerHTML=config.switches.map((entry,index)=>`<article class="step" data-switch="${esc(entry.id)}"><div class="form-grid"><label>第 ${index+1} 次切换时间（运行秒数）<input data-switch-field="at_s" type="number" min="0" step="0.1" value="${entry.at_s}"></label><label>切换到 VLA 模型<select data-switch-field="model">${modelOptions(entry.model)}</select></label><label>切换后任务指令<textarea data-switch-field="task" placeholder="留空使用该模型默认指令">${esc(entry.task||'')}</textarea></label></div><button data-remove-switch>删除这次切换</button></article>`).join('')||'<p class="muted">没有定时切换时，起始 VLA 始终运行。可设置 A → B → A，模型不会在切换时卸载。</p>';
}
function initModelSwitches() {
  $('add-model-switch').onclick=()=>{
    const config=continuousConfig();
    const current=config.switches.at(-1)?.model||config.model;
    const model=plan.models.find(m=>m.id!==current)||plan.models[0];
    if(!model){toast('请先新增 VLA 模型',true);return;}
    config.switches.push({id:uid(),at_s:Math.max(0,...config.switches.map(s=>Number(s.at_s)))+5,model:model.id,task:''});
    changed();renderModelSwitches();renderState(live);
  };
  $('model-switches').onchange=event=>{
    const input=event.target,card=input.closest('[data-switch]');if(!card||!input.dataset.switchField)return;
    const entry=continuousConfig().switches.find(s=>s.id===card.dataset.switch);
    entry[input.dataset.switchField]=input.type==='number'?Number(input.value):input.value;
    changed();renderState(live);
  };
  $('model-switches').onclick=event=>{
    const button=event.target.closest('[data-remove-switch]');if(!button)return;
    const id=button.closest('[data-switch]').dataset.switch;
    continuousConfig().switches=continuousConfig().switches.filter(s=>s.id!==id);
    changed();renderModelSwitches();renderState(live);
  };
}
