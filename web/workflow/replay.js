'use strict';
const el=id=>document.getElementById(id);
let data=null,samples=[],cursor=0,playing=false,previous=null;
const conditionDefs=()=>data?.metadata?.plan?.continuous_vla?.takeovers||[];
async function api(url){const r=await fetch(url);const v=await r.json();if(!r.ok)throw Error(v.error||'读取失败');return v;}
function error(e){el('error').textContent=e?.message||String(e);}
async function refresh(){
  try{const runs=(await api('api/recordings')).runs;const requested=el('runs').value||new URLSearchParams(location.search).get('id');
    el('runs').replaceChildren();for(const r of runs){const o=document.createElement('option');o.value=r.id;o.textContent=`${new Date(r.started_at*1000).toLocaleString()} · ${r.simulated?'模拟':'真机'} · ${r.name} · ${r.finished?'已保存':'记录中/尚无结束摘要'}`;el('runs').append(o);}
    if(!runs.length){el('meta').textContent='暂无记录。启用自动记录后，新执行的工作流或单点试运行会分别保存。';return;}
    el('runs').value=runs.some(r=>r.id===requested)?requested:runs[0].id;await load();
  }catch(e){error(e);}
}
async function load(){
  playing=false;el('play').textContent='播放';previous=null;el('error').textContent='';
  try{const id=el('runs').value;data=await api('api/recording?id='+encodeURIComponent(id));samples=data.samples;cursor=0;el('scrub').value=0;
    for(const [element,file] of [['csv','samples.csv'],['raw','samples.jsonl'],['judgments','events.jsonl']]){el(element).href='api/recording?id='+encodeURIComponent(id)+'&file='+file;el(element).download=id+'-'+file;}
    const end=data.events.findLast(e=>e.kind==='run_end');
    el('meta').textContent=`${data.metadata.simulated?'模拟器反馈':'真实从臂反馈'}｜原始采样 ${data.total_samples} 条｜约 ${data.metadata.nominal_hz} Hz｜${data.summary?'记录已结束':'尚无结束摘要；可刷新读取已落盘数据'}${end?'｜结束状态 '+end.data.state:''}｜记录目录 logs/workflow-runs/${id}`;
    const failures=samples.filter(s=>s.feedback_error).length;
    if(data.summary?.error)error(data.summary.error);
    else if(data.summary?.dropped_events)error(`记录队列丢失 ${data.summary.dropped_events} 条事件，请结合原始日志检查。`);
    else if(failures)error(`当前显示样本中有 ${failures} 条反馈缺口；曲线不会用缓存角度填补。`);
    if(data.display_stride>1)el('meta').textContent+=`｜图中每 ${data.display_stride} 条显示1条，下载文件保留全部采样`;
    el('events').textContent=data.events.filter(e=>e.kind==='message'||e.kind==='run_end'||e.kind==='condition'&&e.data.conditions.some(c=>c.state==='expired'||c.state==='triggered')).map(e=>{
      if(e.kind==='message')return `${e.elapsed_s.toFixed(2)}s  ${e.data.message}`;
      if(e.kind==='run_end')return `${e.elapsed_s.toFixed(2)}s  结束：${e.data.state}${e.data.error?'；'+e.data.error:''}`;
      return `${e.elapsed_s.toFixed(2)}s  VLA时钟 ${e.data.schedule_elapsed.toFixed(2)}s  `+e.data.conditions.filter(c=>c.state==='expired'||c.state==='triggered').map(c=>`${c.name}：${c.state==='triggered'?'已触发':'超时跳过'}`).join('；');
    }).join('\n');draw();update();
  }catch(e){error(e);}
}
function series(){const side=el('side').value,sides=side==='both'?['left','right']:[side],out=[];
  for(const s of sides){for(let j=0;j<6;j++)out.push({name:`${s==='right'?'右':'左'}臂 J${j+1}`,unit:'°',side:s,value:r=>r.joints_deg?.[(s==='right'?6:0)+j],kind:j===5&&s==='right'?'j6_deg':null});
    if(s==='right')out.push({name:'右臂 J4＋J6 合计',unit:'°',side:s,value:r=>r.joints_deg?r.joints_deg[9]+r.joints_deg[11]:undefined,kind:'sum_deg'});
    out.push({name:`${s==='right'?'右':'左'}臂法兰高度 Z`,unit:'mm',side:s,value:r=>r.flange_xyz_mm?.[s]?.[2],kind:s==='right'?'z_mm':null});}return out;}
function draw(){el('charts').replaceChildren();for(const spec of series()){const canvas=document.createElement('canvas');canvas.dataset.name=spec.name;el('charts').append(canvas);paint(canvas,spec);}}
function paint(canvas,spec){
  const ratio=devicePixelRatio||1,w=canvas.clientWidth||620,h=220;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext('2d');c.scale(ratio,ratio);
  const vals=samples.map(spec.value).filter(Number.isFinite),thresholds=conditionDefs().flatMap(e=>{const t=e.trigger;if(!((t?.type==='right_j6_pose'&&['j6_deg','z_mm'].includes(spec.kind))||(t?.type==='right_joint_sum_pose'&&['sum_deg','z_mm'].includes(spec.kind))))return [];const rows=[{value:t[spec.kind],label:'联合下限',color:'#b54e39'}];if(spec.kind==='z_mm'&&t.high_z_mm!=null)rows.push({value:t.high_z_mm,label:'高位独立下限',color:'#7a4694'});return rows;});
  if(!vals.length){c.fillText(spec.name+'：无有效反馈',20,25);return;}
  let lo=Infinity,hi=-Infinity;for(const v of [...vals,...thresholds.map(t=>t.value)]){lo=Math.min(lo,v);hi=Math.max(hi,v);}const margin=Math.max((hi-lo)*.12,spec.unit==='°'?1:5);lo-=margin;hi+=margin;
  const end=samples.at(-1)?.elapsed_s||1,L=58,R=w-16,T=32,B=h-30,x=t=>L+t/end*(R-L),y=v=>B-(v-lo)/(hi-lo)*(B-T);
  c.font='13px system-ui';c.fillStyle='#213a3e';c.fillText(spec.name+`（${spec.unit}）`,L,20);c.font='11px system-ui';
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;c.strokeStyle='#e4ebeb';c.beginPath();c.moveTo(L,y(v));c.lineTo(R,y(v));c.stroke();c.fillStyle='#60757a';c.fillText(v.toFixed(1),3,y(v)+4);}
  for(let i=0;i<=4;i++){const t=end*i/4;c.fillText(t.toFixed(1)+'s',x(t)-9,h-8);}
  c.strokeStyle=spec.side==='right'?'#198574':'#3478ad';c.lineWidth=1.4;c.beginPath();let prior=null;
  for(const row of samples){const v=spec.value(row);if(!Number.isFinite(v)){prior=null;continue;}if(prior===null||row.elapsed_s-prior>.2)c.moveTo(x(row.elapsed_s),y(v));else c.lineTo(x(row.elapsed_s),y(v));prior=row.elapsed_s;}c.stroke();
  c.setLineDash([5,4]);for(const threshold of thresholds){c.strokeStyle=threshold.color;c.beginPath();c.moveTo(L,y(threshold.value));c.lineTo(R,y(threshold.value));c.stroke();c.fillStyle=threshold.color;c.fillText(threshold.label+' '+threshold.value.toFixed(1),R-135,y(threshold.value)-5);}c.setLineDash([]);
  c.strokeStyle='#344f5a';c.beginPath();c.moveTo(x(cursor),T);c.lineTo(x(cursor),B);c.stroke();
}
function update(){if(!samples.length)return;let best=samples[0];for(const s of samples){if(Math.abs(s.elapsed_s-cursor)<Math.abs(best.elapsed_s-cursor))best=s;}
  el('time').textContent=`${cursor.toFixed(2)} / ${(samples.at(-1)?.elapsed_s||0).toFixed(2)} 秒`;
  const lines=[`VLA运行时钟 ${(best.schedule_elapsed||0).toFixed(2)} 秒｜${best.paused?'暂停｜':''}状态 ${best.state}｜控制权 ${best.controller_owner||'准备/保持'}`];
  for(const side of el('side').value==='both'?['left','right']:[el('side').value]){const q=best.joints_deg?.slice(side==='right'?6:0,side==='right'?12:6);lines.push(`${side==='right'?'右':'左'} J1～J6：${q?q.map(v=>v.toFixed(2)+'°').join(' / '):'反馈缺失'}；Z ${best.flange_xyz_mm?.[side]?.[2]?.toFixed(2)??'缺失'} mm`);}
  const event=data.events.filter(e=>e.kind==='condition'&&e.elapsed_s<=cursor).at(-1);
  if(event){const age=cursor-event.elapsed_s;lines.push(`最近一次条件判断距光标 ${age.toFixed(2)} 秒（非当前帧）：`);for(const s of event.data.conditions){const angle=s.type==='right_joint_sum_pose'?`J4 ${s.j4_deg.toFixed(2)}° ＋ J6 ${s.j6_deg.toFixed(2)}° ＝ ${s.sum_deg.toFixed(2)}° / 下限 ${s.sum_threshold_deg}°`:`J6 ${s.j6_deg?.toFixed(2)??'—'}° / 下限 ${s.j6_threshold_deg??'—'}°`;lines.push(`${s.name}：${s.state}；${angle}；Z ${s.z_mm?.toFixed(2)??'—'} / 下限 ${s.z_threshold_mm??'—'} mm${s.high_z_threshold_mm!=null?`；或 Z ≥ ${s.high_z_threshold_mm} mm（${s.high_z_matched?'达标':'未达标'}）`:''}${s.matched_by?'；触发分支 '+({high_z:'高位独立',sum_and_height:'角度＋高度',both:'两条分支均满足'}[s.matched_by]||s.matched_by):''}；持续 ${(s.held_s||0).toFixed(2)} 秒`);}}
  el('readout').textContent=lines.join('\n');for(const [i,canvas] of [...el('charts').children].entries())paint(canvas,series()[i]);
}
el('refresh').onclick=refresh;el('runs').onchange=load;el('side').onchange=()=>{draw();update();};
el('scrub').oninput=()=>{cursor=Number(el('scrub').value)/1000*(samples.at(-1)?.elapsed_s||0);update();};
el('play').onclick=()=>{playing=!playing;previous=null;if(playing&&cursor>=(samples.at(-1)?.elapsed_s||0))cursor=0;el('play').textContent=playing?'暂停回放':'播放';};
function animate(now){if(playing&&samples.length){if(previous!==null)cursor+=(now-previous)/1000*Number(el('speed').value);previous=now;const end=samples.at(-1).elapsed_s;if(cursor>=end){cursor=end;playing=false;el('play').textContent='播放';}el('scrub').value=end?cursor/end*1000:0;update();}requestAnimationFrame(animate);}requestAnimationFrame(animate);window.addEventListener('resize',()=>{if(data){draw();update();}});refresh();
