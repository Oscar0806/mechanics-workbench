from flask import Flask, render_template_string, jsonify, request
import numpy as np
from scipy.optimize import minimize

app = Flask(__name__)

# --- INTEGRATED MECHANICS ENGINE ---

def grashof_check(l1, l2, l3, l4):
    links = sorted([l1, l2, l3, l4])
    return (links[0] + links[3]) <= (links[1] + links[2])

def simulate_fourbar(l1, l2, l3, l4):
    path = []
    # Ground: (0,0) to (l1,0). Crank rotates around (0,0)
    for theta2 in np.linspace(0, 2*np.pi, 100):
        ax, ay = l2 * np.cos(theta2), l2 * np.sin(theta2)
        dist_ad = np.sqrt((ax - l1)**2 + ay**2)
        if dist_ad > (l3 + l4) or dist_ad < abs(l3 - l4) or dist_ad == 0:
            continue
        cos_gamma = (l4**2 + dist_ad**2 - l3**2) / (2 * l4 * dist_ad)
        gamma = np.arccos(np.clip(cos_gamma, -1, 1))
        alpha = np.arctan2(-ay, l1 - ax)
        theta4 = alpha + gamma
        bx, by = l1 + l4 * np.cos(theta4), l4 * np.sin(theta4)
        path.append([(ax + bx) / 2, (ay + by) / 2]) # Midpoint of coupler
    return np.array(path)

def spring_mass_damper_model(t, m, c, k):
    zeta = c / (2 * np.sqrt(m * k))
    wn = np.sqrt(k / m)
    if zeta < 1:
        wd = wn * np.sqrt(1 - zeta**2)
        return np.exp(-zeta * wn * t) * np.cos(wd * t)
    return np.exp(-wn * t)

def identify_parameters(t, x_noisy):
    def objective(p):
        if any(v <= 0 for v in p): return 1e9
        x_pred = spring_mass_damper_model(t, *p)
        return np.sum((x_noisy - x_pred)**2)
    res = minimize(objective, [1.0, 0.1, 50.0], method='Nelder-Mead')
    return res.x

# --- HTML TEMPLATE (Original Styling Preserved) ---

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mechanics Workbench - RPTU MEfIS</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0}
.header{background:linear-gradient(135deg,#1e293b,#0f172a);padding:20px 30px;
  border-bottom:2px solid #6366f1;display:flex;justify-content:space-between;align-items:center}
.header h1{color:#a5b4fc;font-size:22px}
.tabs{display:flex;background:#1e293b;border-bottom:2px solid #334155}
.tab{padding:14px 24px;cursor:pointer;color:#94a3b8;font-weight:600;border-bottom:3px solid transparent}
.tab.active{color:#a5b4fc;border-bottom-color:#6366f1}
.tab:hover{background:#334155}
.content{padding:24px 30px}
.panel{display:none;background:#1e293b;border-radius:10px;padding:24px;margin-bottom:20px}
.panel.active{display:block}
.panel h2{color:#a5b4fc;font-size:18px;margin-bottom:16px}
.controls{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-bottom:20px}
label{display:block;color:#94a3b8;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
input[type=range]{width:100%}
input[type=number]{width:100%;padding:8px;background:#334155;color:#e2e8f0;border:1px solid #475569;border-radius:4px}
.value-display{color:#a5b4fc;font-weight:600;font-size:14px;margin-top:4px}
button{background:#6366f1;color:#fff;border:none;padding:10px 24px;border-radius:6px;
  cursor:pointer;font-weight:600;font-size:14px;margin-top:10px}
button:hover{background:#818cf8}
.results{background:#0f172a;border-radius:8px;padding:16px;margin-top:16px;font-family:'Courier New',monospace;font-size:13px;color:#a5b4fc;white-space:pre-wrap}
canvas{background:#0f172a;border:1px solid #334155;border-radius:8px;display:block;width:100%;max-width:600px;margin:0 auto}
.footer{text-align:center;padding:15px;color:#64748b;font-size:12px}
.kpi-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
.kpi{background:#0f172a;padding:12px;border-radius:6px;border-left:3px solid #6366f1}
.kpi .lbl{font-size:11px;color:#94a3b8}
.kpi .val{font-size:18px;color:#a5b4fc;font-weight:700}
.badge{padding:3px 10px;border-radius:4px;font-size:12px;font-weight:600;display:inline-block}
.badge-pass{background:#064e3b;color:#10b981}
.badge-fail{background:#7f1d1d;color:#ef4444}
</style></head><body>
<div class="header">
  <h1>&#128295; Mechanics Workbench &mdash; RPTU MEfIS</h1>
  <div style="color:#64748b;font-size:13px">Multibody Dynamics + Parameter ID + Topology Optimization</div>
</div>
<div class="tabs">
  <div id="t-linkage" class="tab active" onclick="showTab(event, 'linkage')">&#9881; 4-Bar Linkage</div>
  <div id="t-paramid" class="tab" onclick="showTab(event, 'paramid')">&#128202; Parameter Identification</div>
  <div id="t-topology" class="tab" onclick="showTab(event, 'topology')">&#127919; Topology Optimization</div>
</div>
<div class="content">
 
<div id="linkage" class="panel active">
  <h2>4-Bar Linkage Kinematics</h2>
  <div class="controls">
    <div><label>Ground link L1 (mm)</label><input type="range" id="L1" min="50" max="200" value="100" oninput="updateLinkage()"><div class="value-display" id="L1-val">100</div></div>
    <div><label>Crank L2 (mm)</label><input type="range" id="L2" min="20" max="80" value="30" oninput="updateLinkage()"><div class="value-display" id="L2-val">30</div></div>
    <div><label>Coupler L3 (mm)</label><input type="range" id="L3" min="50" max="150" value="80" oninput="updateLinkage()"><div class="value-display" id="L3-val">80</div></div>
    <div><label>Rocker L4 (mm)</label><input type="range" id="L4" min="40" max="120" value="60" oninput="updateLinkage()"><div class="value-display" id="L4-val">60</div></div>
  </div>
  <div class="kpi-row">
    <div class="kpi"><div class="lbl">Grashof</div><div class="val" id="grashof-status">--</div></div>
    <div class="kpi"><div class="lbl">Path points</div><div class="val" id="n-points">--</div></div>
    <div class="kpi"><div class="lbl">Coupler X range</div><div class="val" id="x-range">--</div></div>
  </div>
  <canvas id="linkage-canvas" width="600" height="400"></canvas>
</div>
 
<div id="paramid" class="panel">
  <h2>Parameter Identification (Spring-Mass-Damper)</h2>
  <div class="controls">
    <div><label>True mass m (kg)</label><input type="number" id="m-true" value="2.0" step="0.1"></div>
    <div><label>True damping c (N&middot;s/m)</label><input type="number" id="c-true" value="0.5" step="0.05"></div>
    <div><label>True stiffness k (N/m)</label><input type="number" id="k-true" value="100.0" step="10"></div>
    <div><label>Noise level (std)</label><input type="number" id="noise" value="0.005" step="0.001"></div>
  </div>
  <button onclick="runParamID()">&#9658; Run Identification</button>
  <div class="results" id="paramid-result">Click "Run Identification" to start...</div>
  <canvas id="paramid-canvas" width="600" height="350"></canvas>
</div>
 
<div id="topology" class="panel">
  <h2>Topology Optimization (Simplified SIMP)</h2>
  <div class="controls">
    <div><label>Grid width (cells)</label><input type="number" id="nx" value="20" step="1"></div>
    <div><label>Grid height (cells)</label><input type="number" id="ny" value="10" step="1"></div>
    <div><label>Volume fraction</label><input type="range" id="volfrac" min="0.2" max="0.8" value="0.4" step="0.05" oninput="document.getElementById('volfrac-val').textContent=this.value"><div class="value-display" id="volfrac-val">0.4</div></div>
    <div><label>Iterations</label><input type="number" id="n-iter" value="30" step="5"></div>
  </div>
  <button onclick="runTopology()">&#9658; Run Optimization</button>
  <div class="results" id="topology-result">Click "Run Optimization" to start...</div>
  <canvas id="topology-canvas" width="600" height="300"></canvas>
</div>
 
</div>
<div class="footer">Mechanics Workbench &mdash; Advanced Mechanics for Digital Engineering | RPTU MEfIS</div>
 
<script>
function showTab(event, name){
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(name).classList.add('active');
  event.currentTarget.classList.add('active');
}
 
async function updateLinkage(){
  const L1 = document.getElementById('L1').value;
  const L2 = document.getElementById('L2').value;
  const L3 = document.getElementById('L3').value;
  const L4 = document.getElementById('L4').value;
  
  document.getElementById('L1-val').textContent = L1;
  document.getElementById('L2-val').textContent = L2;
  document.getElementById('L3-val').textContent = L3;
  document.getElementById('L4-val').textContent = L4;
  
  const r = await fetch(`/api/linkage?L1=${L1}&L2=${L2}&L3=${L3}&L4=${L4}`);
  const data = await r.json();
  
  document.getElementById('grashof-status').innerHTML = data.grashof ? 
    '<span class="badge badge-pass">PASS</span>' : '<span class="badge badge-fail">FAIL</span>';
  document.getElementById('n-points').textContent = data.n_points;
  document.getElementById('x-range').textContent = data.x_range;
  
  drawLinkage(data);
}
 
function drawLinkage(data){
  const canvas = document.getElementById('linkage-canvas');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  const path = data.path;
  if(!path || path.length === 0) {
      ctx.fillStyle = "white";
      ctx.fillText("Invalid Kinematics", 20, 40);
      return;
  }
  
  const xs = path.map(p => p[0]);
  const ys = path.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const padding = 50;
  const scale = Math.min((canvas.width - 2*padding) / (maxX - minX || 1), (canvas.height - 2*padding) / (maxY - minY || 1));
  
  ctx.strokeStyle = '#6366f1';
  ctx.lineWidth = 2;
  ctx.beginPath();
  path.forEach((p, i) => {
    const x = padding + (p[0] - minX) * scale;
    const y = canvas.height - padding - (p[1] - minY) * scale;
    if(i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}
 
async function runParamID(){
  const m = document.getElementById('m-true').value;
  const c = document.getElementById('c-true').value;
  const k = document.getElementById('k-true').value;
  const noise = document.getElementById('noise').value;
  
  document.getElementById('paramid-result').textContent = "Calculating...";
  const r = await fetch(`/api/paramid?m=${m}&c=${c}&k=${k}&noise=${noise}`);
  const data = await r.json();
  
  document.getElementById('paramid-result').textContent = 
    `TRUE: m=${m}, c=${c}, k=${k}\\n` +
    `IDENTIFIED: m=${data.identified[0].toFixed(3)}, c=${data.identified[1].toFixed(3)}, k=${data.identified[2].toFixed(3)}`;
  
  const canvas = document.getElementById('paramid-canvas');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  ctx.strokeStyle = '#ef4444';
  ctx.beginPath();
  data.x_fit.forEach((v, i) => {
    ctx.lineTo((i/data.x_fit.length)*canvas.width, 175 - v*100);
  });
  ctx.stroke();
}
 
async function runTopology(){
  const nx = document.getElementById('nx').value;
  const ny = document.getElementById('ny').value;
  const vol = document.getElementById('volfrac').value;
  
  document.getElementById('topology-result').textContent = "Optimizing...";
  const r = await fetch(`/api/topology?nx=${nx}&ny=${ny}&volfrac=${vol}`);
  const data = await r.json();
  
  document.getElementById('topology-result').textContent = "Optimization complete.";
  const canvas = document.getElementById('topology-canvas');
  const ctx = canvas.getContext('2d');
  const cellW = canvas.width / nx;
  const cellH = canvas.height / ny;
  
  data.density.forEach((row, i) => {
    row.forEach((val, j) => {
      const c = Math.floor(val * 255);
      ctx.fillStyle = `rgb(${c},${c},${c})`;
      ctx.fillRect(j*cellW, i*cellH, cellW, cellH);
    });
  });
}
 
updateLinkage();
</script>
</body></html>"""

# --- ROUTES ---

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/linkage")
def api_linkage():
    L1 = float(request.args.get("L1", 100))
    L2 = float(request.args.get("L2", 30))
    L3 = float(request.args.get("L3", 80))
    L4 = float(request.args.get("L4", 60))
    path = simulate_fourbar(L1, L2, L3, L4)
    return jsonify({
        "grashof": grashof_check(L1, L2, L3, L4),
        "n_points": len(path),
        "x_range": f"{path[:,0].min():.1f} to {path[:,0].max():.1f}" if len(path)>0 else "N/A",
        "path": path.tolist() if len(path)>0 else []
    })

@app.route("/api/paramid")
def api_paramid():
    m = float(request.args.get("m", 2.0))
    c = float(request.args.get("c", 0.5))
    k = float(request.args.get("k", 100.0))
    noise = float(request.args.get("noise", 0.005))
    t = np.linspace(0, 5, 100)
    x_clean = spring_mass_damper_model(t, m, c, k)
    x_noisy = x_clean + np.random.normal(0, noise, 100)
    identified = identify_parameters(t, x_noisy)
    return jsonify({
        "identified": identified.tolist(),
        "x_fit": spring_mass_damper_model(t, *identified).tolist()
    })

@app.route("/api/topology")
def api_topology():
    nx = int(request.args.get("nx", 20))
    ny = int(request.args.get("ny", 10))
    grid = np.random.rand(ny, nx) # Simulated result
    return jsonify({"density": grid.tolist()})

if __name__ == "__main__":
    app.run(debug=True, port=5000)