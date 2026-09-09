#!/usr/bin/env python3
from flask import Flask,request,jsonify,render_template_string
import pathlib,subprocess,threading,signal,re,os
ROOT=pathlib.Path(__file__).resolve().parents[2]; PY=ROOT/'thirdParty/.venvs/lerobot-pi05/bin/python'; TR=ROOT/'scripts/LerobotTrainer/LerobotTrainer.py'; PIDFILE=ROOT/'models/.lerobot_trainer.pid'; app=Flask(__name__); S={'p':None,'status':'idle','lines':[],'step':0,'steps':0}
PAGE='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PI05 Trainer</title><style>body{font:14px system-ui;background:#eef2f7;color:#172033;margin:0}header{background:#172554;color:white;padding:25px calc((100% - 1100px)/2)}main{max-width:1100px;margin:20px auto;display:grid;grid-template-columns:1fr 350px;gap:16px}.card{background:white;border-radius:14px;padding:20px;box-shadow:0 5px 20px #0001}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{font-weight:600}input,select{display:block;width:100%;padding:9px;margin-top:5px;border:1px solid #ccd5e1;border-radius:7px}button{padding:11px 20px;border:0;border-radius:8px;margin-top:18px;font-weight:bold}#start{background:#2563eb;color:white}#cancel{background:#fee2e2;color:#991b1b;margin-left:8px}button:disabled{opacity:.4}progress{width:100%;height:17px}#log{height:390px;overflow:auto;background:#0f172a;color:#cbd5e1;padding:12px;white-space:pre-wrap;font-size:12px}.badge{padding:5px 10px;border-radius:99px;background:#e2e8f0}@media(max-width:800px){main{display:block;margin:10px}.side{margin-top:12px}.grid{grid-template-columns:1fr}} </style><header><h1>LeRobot · PI05 Trainer</h1><span>本地训练控制台</span></header><main><section class="card"><h2>训练配置</h2><div class="grid">%s</div><button id="start" onclick="start()">开始训练</button><button id="cancel" onclick="cancel()" disabled>终止训练</button></section><aside class="card side"><h2>状态 <span id="badge" class="badge">空闲</span></h2><p id="count">0 / 0</p><progress id="bar" max="1" value="0"></progress><h3>日志</h3><pre id="log">等待开始…</pre></aside></main><script>const K=%s,$=x=>document.getElementById(x);function data(){let o={};K.forEach(k=>o[k]=$(k).value);return o}function lock(x){K.forEach(k=>$(k).disabled=x);$('start').disabled=x;$('cancel').disabled=!x}async function start(){let r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data())});if(!r.ok)return alert(await r.text());lock(1);poll()}async function cancel(){await fetch('/api/cancel',{method:'POST'})}async function poll(){let j=await(await fetch('/api/status')).json();$('badge').textContent=j.status;$('count').textContent=j.step+' / '+j.steps;$('bar').max=j.steps||1;$('bar').value=j.step;$('log').textContent=j.lines.join('')||'等待日志…';$('log').scrollTop=99999;if(j.status==='running'||j.status==='cancelling')setTimeout(poll,800);else lock(0)}poll()</script>'''
FIELDS='''<label>数据集<input id="dataset_root" list="datasets" value="datasets/lerobot/"><datalist id="datasets"></datalist></label><label>Base checkpoint<input id="pretrained" list="bases" value="models/base_models/pi05_base"><datalist id="bases"></datalist></label><label>输出目录<input id="output_dir" list="fits" value="models/finetuned_models/pi05_run"><datalist id="fits"></datalist></label><label>Job name<input id="job_name" value="pi05_piper"></label><label>归一化<select id="normalization_mode"><option>quantiles</option><option>min_max</option><option>mean_std</option></select></label><label>微调<select id="finetune_mode"><option>full</option><option>lora</option></select></label><label>RTC<select id="rtc_mode"><option>normal</option><option>trained</option></select></label><label>LoRA rank<input id="lora_rank" value="16" type="number"></label><label>Batch size<input id="batch_size" value="8" type="number"></label><label>训练步数<input id="steps" value="30000" type="number"></label><label>保存频率<input id="save_freq" value="5000" type="number"></label><label>Workers<input id="num_workers" value="4" type="number"></label><label>RTC最大延迟<input id="rtc_max_delay" value="5" type="number"></label><label>设备<select id="device"><option>cuda</option><option>cpu</option></select></label><label>W&B项目<input id="wandb_project" value="piper-pi05"></label><label>W&B entity<input id="wandb_entity" placeholder="可选"></label><label>W&B模式<select id="wandb_mode"><option value="offline">offline</option><option value="online">online（需登录）</option><option value="disabled">disabled</option></select></label><label>W&B<select id="wandb"><option value="false">关闭</option><option value="true">启用</option></select></label>'''
K=['dataset_root','pretrained','output_dir','job_name','normalization_mode','finetune_mode','rtc_mode','lora_rank','batch_size','steps','save_freq','num_workers','rtc_max_delay','device','gpu_ids','wandb_project','wandb_entity','wandb_mode','wandb']
FIELDS=FIELDS.replace('<label>设备','<label>GPU IDs<input id="gpu_ids" value="0" placeholder="例如 0,1 或 0,2"></label><label>设备')
FIELDS=FIELDS.replace('<option value="false">关闭</option><option value="true">启用</option>','<option value="true">启用</option><option value="false">关闭</option>')
FIELDS=FIELDS.replace('<option value="false">关闭</option></select>','<option value="false">关闭</option></select>')
PAGE=PAGE.replace("poll();</script>","async function paths(){let j=await (await fetch('/api/paths')).json();for(let [id,a] of [['datasets',j.datasets],['bases',j.bases],['fits',j.fits]])$(id).innerHTML=a.map(x=>'<option value=\\\"'+x+'\\\">').join('')}paths();poll();</script>")
PAGE=PAGE.replace('</div><script>', '<label>GPU IDs<input id="gpu_ids" value="0" placeholder="例如 0,1 或 0,2"></label></div><script>')
@app.get('/')
def index():
    # Avoid %-formatting: CSS contains literal percent signs (e.g. 100%).
    return render_template_string(PAGE.replace('%s', FIELDS, 1).replace('%s', repr(K), 1))

@app.get('/api/paths')
def paths():
    def dirs(rel):
        p=ROOT/rel
        return [str(x.relative_to(ROOT)) for x in sorted(p.iterdir()) if x.is_dir()] if p.is_dir() else []
    return jsonify(datasets=dirs('datasets/lerobot'), bases=dirs('models/base_models'), fits=dirs('models/finetuned_models'))
@app.post('/api/start')
def start():
 if S['p'] and S['p'].poll() is None:return('已有训练在运行',409)
 d=request.get_json() or {}
 missing=[k for k in K if k not in d]
 if missing:return jsonify(error='缺少参数: '+', '.join(missing)),400
 cmd=[str(PY),str(TR)]
 for k in K:
  if k=='wandb' or k=='wandb_entity':continue
  cmd+=['--'+k.replace('_','-'),d[k]]
 if d['wandb']=='false' and d.get('wandb_mode')!='online':cmd+=['--no-wandb']
 if d['wandb_entity']:cmd+=['--wandb-entity',d['wandb_entity']]
 S.update(status='running',lines=[],step=0,steps=int(d['steps']))
 def run():
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True);S['p']=p; PIDFILE.write_text(str(p.pid))
  for line in p.stdout:
   # Keep the UI readable: retain progress, loss/checkpoint messages and errors.
   clean=line.strip()
   important=any(x in clean.lower() for x in ('training:','loss','checkpoint','error','traceback','out of memory','completed'))
   if important and len(S['lines'])<500:S['lines'].append(clean+'\n')
   m=re.search(r'[Ss]tep\s*[=:]?\s*(\d+)',line) or re.search(r'(\d+)\s*/\s*(\d+)',line)
   if m:S['step']=int(m.group(1)); S['steps']=int(m.group(2)) if m.lastindex and m.lastindex>1 else S['steps']
  S['status']='done' if p.returncode==0 else 'failed'
  try: PIDFILE.unlink()
  except FileNotFoundError: pass
 threading.Thread(target=run,daemon=True).start();return jsonify(ok=True)
@app.post('/api/cancel')
def cancel():
 p=S['p']
 if not (p and p.poll() is None):
  found=PIDFILE.read_text().strip() if PIDFILE.exists() else ''
  if found:
   try: os.killpg(int(found),signal.SIGINT); S['status']='cancelling'
   except ProcessLookupError: pass
 elif p.poll() is None:
  try: os.killpg(p.pid,signal.SIGINT)
  except ProcessLookupError: pass
  S['status']='cancelling'
 return jsonify(ok=True)
@app.get('/api/status')
def status():
 if S['status'] in ('idle','failed','done'):
  if PIDFILE.exists():
   try:
    os.kill(int(PIDFILE.read_text()),0); S['status']='running'
   except (ProcessLookupError,ValueError): PIDFILE.unlink(missing_ok=True)
 return jsonify(status=S['status'],lines=S['lines'][-120:],step=S['step'],steps=S['steps'])
if __name__=='__main__':app.run(host='127.0.0.1',port=8890)
