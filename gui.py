"""
gui.py - Roblox AI Web Dashboard
Launch with: python gui.py  (then open http://localhost:5000)
"""

import os, sys, json, time, threading, subprocess
from pathlib import Path
from datetime import datetime
from collections import deque

from flask import Flask, render_template_string, jsonify, request
import config

app  = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

processes   = {}
log_buffer  = deque(maxlen=500)
train_stats = deque(maxlen=200)
_lock       = threading.Lock()

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    with _lock: log_buffer.append(f"[{ts}] {msg}")

def stream_process(name, proc):
    def _read(stream):
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            log(f"[{name}] {line}")
            if name == "train" and "Ep " in line:
                try:
                    parts = line.split()
                    ep = int(parts[parts.index("Ep")+1])
                    tl = float([p for p in parts if p.startswith("train=")][0].split("=")[1])
                    vl = float([p for p in parts if p.startswith("val=")][0].split("=")[1])
                    with _lock: train_stats.append({"epoch": ep, "train": tl, "val": vl})
                except: pass
    for s in (proc.stdout, proc.stderr):
        threading.Thread(target=_read, args=(s,), daemon=True).start()

def run_script(name, script, extra_args=[]):
    with _lock:
        if name in processes and processes[name].poll() is None:
            return {"ok": False, "msg": f"{name} already running."}
    proc = subprocess.Popen(
        [sys.executable, os.path.join(BASE, script)] + extra_args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=BASE)
    with _lock: processes[name] = proc
    stream_process(name, proc)
    log(f"Started {name} (pid {proc.pid})")
    return {"ok": True, "msg": f"{name} started (pid {proc.pid})"}

def stop_script(name):
    with _lock: proc = processes.get(name)
    if proc is None or proc.poll() is not None:
        return {"ok": False, "msg": f"{name} not running."}
    proc.terminate()
    try: proc.wait(timeout=5)
    except: proc.kill()
    log(f"Stopped {name}")
    return {"ok": True, "msg": f"{name} stopped."}

def proc_status(name):
    with _lock: proc = processes.get(name)
    if proc is None: return "idle"
    return "running" if proc.poll() is None else "done"

def sessions_info():
    out = []
    for d in sorted(Path(config.RECORDINGS_DIR).iterdir()) if Path(config.RECORDINGS_DIR).exists() else []:
        if d.is_dir() and (d/"actions.json").exists():
            frames = list((d/"frames").glob("*.jpg")) if (d/"frames").exists() else []
            out.append({"name": d.name, "frames": len(frames)})
    return out

def videos_info():
    out = []
    for d in sorted(Path(config.FRAMES_DIR).iterdir()) if Path(config.FRAMES_DIR).exists() else []:
        if d.is_dir():
            out.append({"name": d.name, "frames": len(list(d.glob("*.jpg")))})
    return out

def models_info():
    out = []
    for f in sorted(Path(config.MODELS_DIR).glob("*.pt")) if Path(config.MODELS_DIR).exists() else []:
        out.append({"name": f.name, "size": f"{f.stat().st_size/1e6:.1f} MB"})
    return out


HTML = r"""
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Roblox AI Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#0f0f13;--panel:#17171f;--border:#2a2a38;--accent:#6c63ff;--green:#43e97b;--red:#ff4f4f;--text:#e0e0f0;--muted:#888}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
header{background:var(--panel);border-bottom:1px solid var(--border);padding:14px 24px;display:flex;align-items:center;gap:16px}
header h1{font-size:1.2rem;font-weight:700;color:var(--accent)}
.layout{display:flex;flex:1;overflow:hidden}
nav{width:170px;background:var(--panel);border-right:1px solid var(--border);padding:16px 0;flex-shrink:0}
nav button{display:block;width:100%;padding:12px 20px;background:none;border:none;color:var(--muted);font-size:.88rem;text-align:left;cursor:pointer;border-left:3px solid transparent;transition:all .15s}
nav button:hover{color:var(--text);background:rgba(108,99,255,.08)}
nav button.active{color:var(--accent);border-left-color:var(--accent);background:rgba(108,99,255,.12)}
.content{flex:1;overflow-y:auto;padding:24px}
.tab{display:none}.tab.active{display:block}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:16px}
.card h2{font-size:.82rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.btn{padding:9px 20px;border-radius:7px;border:none;cursor:pointer;font-size:.85rem;font-weight:600;transition:all .15s}
.btn-primary{background:var(--accent);color:#fff}.btn-primary:hover{opacity:.85}
.btn-danger{background:var(--red);color:#fff}.btn-danger:hover{opacity:.85}
.btn-success{background:var(--green);color:#000}.btn-success:hover{opacity:.85}
.btn-ghost{background:transparent;border:1px solid var(--border);color:var(--text)}.btn-ghost:hover{border-color:var(--accent);color:var(--accent)}
.btn-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
input,textarea,select{background:#1e1e2a;border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:6px;font-size:.85rem;width:100%;outline:none;transition:border-color .15s}
input:focus,textarea:focus,select:focus{border-color:var(--accent)}
label{font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px;margin-top:10px}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:700}
.badge-idle{background:#2a2a38;color:var(--muted)}.badge-running{background:rgba(67,233,123,.15);color:var(--green)}.badge-done{background:rgba(108,99,255,.15);color:var(--accent)}
.stat{background:#1a1a24;border-radius:8px;padding:14px;text-align:center}
.stat .val{font-size:1.6rem;font-weight:700;color:var(--accent)}.stat .lbl{font-size:.75rem;color:var(--muted);margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:.84rem}
th{color:var(--muted);font-weight:600;text-align:left;padding:8px 12px;border-bottom:1px solid var(--border)}
td{padding:8px 12px;border-bottom:1px solid #1e1e2a}
#log-box{background:#0a0a0f;border:1px solid var(--border);border-radius:8px;padding:14px;height:300px;overflow-y:auto;font-family:monospace;font-size:.76rem;color:#a0a0c0;line-height:1.7}
#log-box .err{color:var(--red)}.log-ok{color:var(--green)}
.chart-wrap{position:relative;height:220px}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style></head><body>
<header>
  <h1>🎮 Roblox AI Dashboard</h1>
  <span style="margin-left:auto;font-size:.8rem;color:var(--muted)" id="clock"></span>
</header>
<div class="layout">
<nav>
  <button class="active" onclick="showTab('record',this)">⏺  Record</button>
  <button onclick="showTab('videos',this)">🎬  Videos</button>
  <button onclick="showTab('train',this)">🧠  Train</button>
  <button onclick="showTab('run',this)">🤖  Run AI</button>
  <button onclick="showTab('settings',this)">⚙️  Settings</button>
  <button onclick="showTab('logs',this)">📋  Logs</button>
</nav>
<div class="content">

<div class="tab active" id="tab-record">
  <div class="card"><h2>Recording</h2>
    <div class="grid3">
      <div class="stat"><div class="val" id="rec-sessions">-</div><div class="lbl">Sessions</div></div>
      <div class="stat"><div class="val" id="rec-frames">-</div><div class="lbl">Frames</div></div>
      <div class="stat"><div class="val" id="rec-status">idle</div><div class="lbl">Status</div></div>
    </div>
    <div class="btn-row">
      <button class="btn btn-success" onclick="startScript('record')">▶ Start Recording</button>
      <button class="btn btn-danger"  onclick="stopScript('record')">■ Stop</button>
    </div>
    <p style="color:var(--muted);font-size:.8rem;margin-top:12px">Switch to your game then press <b>F8</b> to begin, <b>F9</b> to stop.</p>
  </div>
  <div class="card"><h2>Sessions</h2>
    <table><thead><tr><th>Session</th><th>Frames</th></tr></thead><tbody id="sessions-tbody"></tbody></table>
  </div>
</div>

<div class="tab" id="tab-videos">
  <div class="card"><h2>YouTube URLs</h2>
    <textarea id="yt-urls" rows="5" placeholder="https://www.youtube.com/watch?v=..."></textarea>
    <div class="btn-row"><button class="btn btn-primary" onclick="extractYT()">⬇ Download & Extract</button></div>
  </div>
  <div class="card"><h2>Extracted Frames</h2>
    <div class="grid3" style="margin-bottom:12px">
      <div class="stat"><div class="val" id="vid-count">-</div><div class="lbl">Sources</div></div>
      <div class="stat"><div class="val" id="vid-frames">-</div><div class="lbl">Total Frames</div></div>
      <div class="stat"><div class="val" id="vid-status">idle</div><div class="lbl">Status</div></div>
    </div>
    <table><thead><tr><th>Source</th><th>Frames</th></tr></thead><tbody id="videos-tbody"></tbody></table>
  </div>
</div>

<div class="tab" id="tab-train">
  <div class="card"><h2>Training</h2>
    <div class="grid2">
      <div><label>Mode</label>
        <select id="train-mode">
          <option value="">Full pipeline (pretrain + finetune)</option>
          <option value="--skip-pretrain">Skip pretraining</option>
          <option value="--resume">Resume checkpoint</option>
        </select></div>
      <div><label>Epochs</label><input type="number" id="train-epochs" value="60" min="1"></div>
    </div>
    <div class="btn-row">
      <button class="btn btn-primary" onclick="startTrain()">🧠 Start Training</button>
      <button class="btn btn-danger"  onclick="stopScript('train')">■ Stop</button>
    </div>
    <div style="margin-top:12px">Status: <span class="badge badge-idle" id="train-badge">idle</span></div>
  </div>
  <div class="card"><h2>Loss Curve</h2>
    <div class="chart-wrap"><canvas id="loss-chart"></canvas></div>
  </div>
  <div class="card"><h2>Models</h2>
    <table><thead><tr><th>File</th><th>Size</th></tr></thead><tbody id="models-tbody"></tbody></table>
  </div>
</div>

<div class="tab" id="tab-run">
  <div class="card"><h2>AI Player</h2>
    <div class="grid2">
      <div><label>Model</label><select id="ai-model"><option value="">best_model.pt (default)</option></select></div>
      <div><label>Mode</label>
        <select id="ai-mode"><option value="">Live (real inputs)</option><option value="--dry-run">Dry run (predict only)</option></select></div>
    </div>
    <div class="btn-row">
      <button class="btn btn-success" onclick="startAI()">▶ Start AI</button>
      <button class="btn btn-danger"  onclick="stopScript('inference')">■ Stop AI</button>
    </div>
    <div style="margin-top:12px">Status: <span class="badge badge-idle" id="ai-badge">idle</span></div>
    <p style="color:var(--muted);font-size:.8rem;margin-top:10px"><b>F10</b>=pause · <b>F11</b>=dry-run · <b>ESC</b>=stop</p>
  </div>
</div>

<div class="tab" id="tab-settings">
  <div class="card"><h2>Config</h2>
    <div class="grid2">
      <div><label>Record FPS</label><input type="number" id="cfg-rfps" value="{{ cfg.RECORD_FPS }}"></div>
      <div><label>Inference FPS</label><input type="number" id="cfg-ifps" value="{{ cfg.INFERENCE_FPS }}"></div>
      <div><label>Frame Width</label><input type="number" id="cfg-fw" value="{{ cfg.FRAME_W }}"></div>
      <div><label>Frame Height</label><input type="number" id="cfg-fh" value="{{ cfg.FRAME_H }}"></div>
      <div><label>Batch Size</label><input type="number" id="cfg-batch" value="{{ cfg.BATCH_SIZE }}"></div>
      <div><label>Learning Rate</label><input type="number" id="cfg-lr" step="0.00001" value="{{ cfg.LEARNING_RATE }}"></div>
      <div><label>Key Threshold</label><input type="number" id="cfg-kth" step="0.01" value="{{ cfg.KEY_CONFIDENCE_THRESH }}"></div>
      <div><label>Mouse Scale</label><input type="number" id="cfg-ms" step="0.1" value="{{ cfg.MOUSE_SCALE }}"></div>
    </div>
    <div class="btn-row"><button class="btn btn-primary" onclick="saveSettings()">💾 Save</button></div>
    <div id="settings-msg" style="margin-top:10px;font-size:.82rem;color:var(--green)"></div>
  </div>
</div>

<div class="tab" id="tab-logs">
  <div class="card"><h2>Console</h2>
    <div class="btn-row" style="margin-bottom:10px">
      <button class="btn btn-ghost" onclick="document.getElementById('log-box').innerHTML=''">🗑 Clear</button>
    </div>
    <div id="log-box"></div>
  </div>
</div>

</div></div>
<script>
function showTab(name,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}
setInterval(()=>{document.getElementById('clock').textContent=new Date().toLocaleTimeString()},1000);

async function api(url,opts={}){
  try{const r=await fetch(url,{headers:{'Content-Type':'application/json'},...opts});return r.json();}
  catch(e){return{ok:false,msg:e.message};}
}
async function startScript(name,args=[]){const r=await api('/api/start',{method:'POST',body:JSON.stringify({name,args})});addLog((r.ok?'OK ':'ERR ')+r.msg);}
async function stopScript(name){const r=await api('/api/stop',{method:'POST',body:JSON.stringify({name})});addLog((r.ok?'OK ':'ERR ')+r.msg);}
async function startTrain(){
  const mode=document.getElementById('train-mode').value;
  const epochs=document.getElementById('train-epochs').value;
  const args=['--epochs',epochs];if(mode)args.push(mode);
  await startScript('train',args);
}
async function startAI(){
  const mode=document.getElementById('ai-mode').value;
  const model=document.getElementById('ai-model').value;
  const args=['--hud'];if(mode)args.push(mode);if(model)args.push('--model',model);
  await startScript('inference',args);
}
async function extractYT(){
  const urls=document.getElementById('yt-urls').value.trim();
  if(!urls){addLog('ERR No URLs entered.');return;}
  const r=await api('/api/extract',{method:'POST',body:JSON.stringify({urls})});
  addLog((r.ok?'OK ':'ERR ')+r.msg);
}
async function saveSettings(){
  const cfg={RECORD_FPS:parseFloat(document.getElementById('cfg-rfps').value),INFERENCE_FPS:parseFloat(document.getElementById('cfg-ifps').value),
    FRAME_W:parseInt(document.getElementById('cfg-fw').value),FRAME_H:parseInt(document.getElementById('cfg-fh').value),
    BATCH_SIZE:parseInt(document.getElementById('cfg-batch').value),LEARNING_RATE:parseFloat(document.getElementById('cfg-lr').value),
    KEY_CONFIDENCE_THRESH:parseFloat(document.getElementById('cfg-kth').value),MOUSE_SCALE:parseFloat(document.getElementById('cfg-ms').value)};
  const r=await api('/api/settings',{method:'POST',body:JSON.stringify(cfg)});
  document.getElementById('settings-msg').textContent=r.ok?'Settings saved.':'Error: '+r.msg;
}

const lossChart=new Chart(document.getElementById('loss-chart').getContext('2d'),{
  type:'line',data:{labels:[],datasets:[
    {label:'Train',data:[],borderColor:'#6c63ff',tension:.3,pointRadius:2,fill:false},
    {label:'Val',  data:[],borderColor:'#ff6584',tension:.3,pointRadius:2,fill:false}]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#a0a0c0',boxWidth:12}}},
    scales:{x:{ticks:{color:'#666'},grid:{color:'#1e1e2a'}},y:{ticks:{color:'#666'},grid:{color:'#1e1e2a'}}}}});

function addLog(line){
  const box=document.getElementById('log-box');
  const d=document.createElement('div');
  d.className=line.includes('ERR')||line.toLowerCase().includes('error')?'err':'';
  d.textContent=line; box.appendChild(d); box.scrollTop=box.scrollHeight;
}

let _logIdx=0;
async function poll(){
  const d=await api('/api/status');if(!d)return;
  ['train','inference'].forEach(n=>{
    const el=document.getElementById(n==='train'?'train-badge':'ai-badge');if(!el)return;
    const s=d.procs[n]||'idle';el.textContent=s;el.className='badge badge-'+s;});
  document.getElementById('rec-sessions').textContent=d.sessions.length;
  document.getElementById('rec-frames').textContent=d.sessions.reduce((a,s)=>a+s.frames,0).toLocaleString();
  document.getElementById('rec-status').textContent=d.procs.record||'idle';
  document.querySelector('#sessions-tbody').innerHTML=d.sessions.map(s=>`<tr><td>${s.name}</td><td>${s.frames.toLocaleString()}</td></tr>`).join('')||'<tr><td colspan="2" style="color:var(--muted)">None yet</td></tr>';
  document.getElementById('vid-count').textContent=d.videos.length;
  document.getElementById('vid-frames').textContent=d.videos.reduce((a,v)=>a+v.frames,0).toLocaleString();
  document.querySelector('#videos-tbody').innerHTML=d.videos.map(v=>`<tr><td>${v.name}</td><td>${v.frames.toLocaleString()}</td></tr>`).join('')||'<tr><td colspan="2" style="color:var(--muted)">None yet</td></tr>';
  document.querySelector('#models-tbody').innerHTML=d.models.map(m=>`<tr><td>${m.name}</td><td>${m.size}</td></tr>`).join('')||'<tr><td colspan="2" style="color:var(--muted)">None yet</td></tr>';
  const sel=document.getElementById('ai-model');const cur=sel.value;
  sel.innerHTML='<option value="">best_model.pt (default)</option>'+d.models.map(m=>`<option value="models/${m.name}">${m.name}</option>`).join('');
  sel.value=cur;
  if(d.train_stats.length>0){
    lossChart.data.labels=d.train_stats.map(s=>`Ep ${s.epoch}`);
    lossChart.data.datasets[0].data=d.train_stats.map(s=>s.train);
    lossChart.data.datasets[1].data=d.train_stats.map(s=>s.val);
    lossChart.update('none');}
}
async function pollLogs(){
  const d=await api('/api/logs?since='+_logIdx);
  if(d&&d.lines){d.lines.forEach(addLog);_logIdx+=d.lines.length;}
}
setInterval(poll,2000);setInterval(pollLogs,1000);poll();
</script></body></html>
"""

@app.route("/")
def index(): return render_template_string(HTML, cfg=config)

@app.route("/api/status")
def status():
    return jsonify({"procs": {n: proc_status(n) for n in ["record","train","inference","extract"]},
                    "sessions": sessions_info(), "videos": videos_info(),
                    "models": models_info(), "train_stats": list(train_stats)})

@app.route("/api/logs")
def get_logs():
    since = int(request.args.get("since", 0))
    with _lock: all_logs = list(log_buffer)
    return jsonify({"lines": all_logs[since:]})

@app.route("/api/start", methods=["POST"])
def start():
    data = request.json; name = data.get("name"); args = data.get("args", [])
    scripts = {"record": "record_gameplay.py", "train": "train.py", "inference": "inference.py"}
    if name not in scripts: return jsonify({"ok": False, "msg": f"Unknown: {name}"})
    return jsonify(run_script(name, scripts[name], args))

@app.route("/api/stop", methods=["POST"])
def stop(): return jsonify(stop_script(request.json.get("name")))

@app.route("/api/extract", methods=["POST"])
def extract():
    urls = [u.strip() for u in request.json.get("urls","").splitlines() if u.strip()]
    if not urls: return jsonify({"ok": False, "msg": "No URLs."})
    urls_file = os.path.join(BASE, "data", "_yt_urls.txt")
    os.makedirs(os.path.dirname(urls_file), exist_ok=True)
    with open(urls_file, "w") as f: f.write("\n".join(urls))
    return jsonify(run_script("extract", "extract_frames.py", ["--urls-file", urls_file]))

@app.route("/api/settings", methods=["POST"])
def save_settings():
    import re
    data = request.json
    cfg_path = os.path.join(BASE, "config.py")
    try:
        with open(cfg_path) as f: src = f.read()
        for key, val in data.items():
            if val is None: continue
            src = re.sub(rf"^({key}\s*=\s*)[\S]+", lambda m, v=val, k=key: f"{k} = {v}", src, flags=re.MULTILINE)
        with open(cfg_path, "w") as f: f.write(src)
        log("Settings saved.")
        return jsonify({"ok": True, "msg": "Saved."})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    print(f"\nRoblox AI Dashboard\nOpen: http://{args.host}:{args.port}\nStop: Ctrl+C\n")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
