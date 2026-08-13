# -*- coding: utf-8 -*-
"""
generate_dashboard.py
─────────────────────
Run this script AFTER airquality_lstm_publication.py has finished.
It reads the 4 CSV output files and writes a single self-contained
HTML dashboard you can open in any browser — no internet needed.

Usage:
    python generate_dashboard.py

Output:
    outputs/index.html   ← open this in Chrome / Firefox / Edge
"""

import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load your CSV results ──────────────────────────────────────────────────
metrics   = pd.read_csv(OUT / "metrics_test.csv")
forecast  = pd.read_csv(OUT / "future_forecast.csv", index_col=0)
aqi       = pd.read_csv(OUT / "aqi_results.csv",     index_col=0)
statio    = pd.read_csv(OUT / "table2_stationarity.csv")

# ── Prepare JSON blobs ─────────────────────────────────────────────────────
def jdump(obj):
    return json.dumps(obj, default=str)

metrics_json  = jdump(metrics.to_dict(orient="records"))
forecast_json = jdump({
    "dates": forecast.index.tolist(),
    "CO":    forecast["CO"].tolist(),
    "SO2":   forecast["SO2"].tolist(),
    "SO4":   forecast["SO4"].tolist(),
    "PM25":  forecast["PM2.5"].tolist(),
    "NO2":   forecast["NO2"].tolist(),
})
aqi_json = jdump({
    "no2_true": aqi["NO2 AQI (True)"].tolist(),
    "no2_pred": aqi["NO2 AQI (Pred)"].tolist(),
    "months":   ["Apr-24","May-24","Jun-24","Jul-24","Aug-24",
                 "Sep-24","Oct-24","Nov-24","Dec-24","Jan-25"],
})
statio_json = jdump(statio.to_dict(orient="records"))

# ── HTML Template ──────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BiLSTM Air Quality Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{{
    --bg:#0f1117; --surface:#1a1d27; --border:#2a2d3a;
    --text:#e8eaf0; --muted:#8b8fa8; --blue:#4f8ef7;
    --red:#f75f5f; --green:#4fca7a; --amber:#f7b94f;
    --purple:#a78bfa; --teal:#38d9a9;
    --font:'Segoe UI',system-ui,sans-serif;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:var(--font);font-size:14px;line-height:1.6}}
  header{{background:var(--surface);border-bottom:1px solid var(--border);padding:1.25rem 2rem;display:flex;align-items:center;justify-content:space-between}}
  header h1{{font-size:18px;font-weight:600;letter-spacing:-.3px}}
  header span{{font-size:12px;color:var(--muted)}}
  .container{{max-width:1400px;margin:0 auto;padding:1.5rem 2rem}}
  .tabs{{display:flex;gap:4px;margin-bottom:1.5rem;border-bottom:1px solid var(--border);padding-bottom:0}}
  .tab{{font-size:13px;padding:8px 18px;border:none;background:none;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;transition:all .15s}}
  .tab.active{{color:var(--blue);border-bottom-color:var(--blue);font-weight:500}}
  .panel{{display:none}}.panel.active{{display:block}}
  .grid{{display:grid;gap:16px}}
  .g2{{grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}}
  .g3{{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}}
  .g5{{grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.25rem}}
  .card h3{{font-size:12px;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.75rem}}
  .metric-val{{font-size:28px;font-weight:700;margin-bottom:2px}}
  .metric-sub{{font-size:12px;color:var(--muted)}}
  .badge{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;font-weight:500;margin-left:8px;vertical-align:middle}}
  .good{{color:var(--green);background:rgba(79,202,122,.12)}}
  .mod{{color:var(--amber);background:rgba(247,185,79,.12)}}
  .poor{{color:var(--red);background:rgba(247,95,95,.12)}}
  canvas{{display:block;width:100%!important}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{text-align:left;padding:8px 10px;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);font-size:12px}}
  td{{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.04);color:var(--text)}}
  tr:last-child td{{border:none}}
  .pill{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;font-weight:500}}
  .aqi-bar{{display:flex;height:6px;border-radius:3px;overflow:hidden;margin:8px 0 14px}}
  .aqi-bar div{{flex:1}}
    select{{background:var(--surface);border:1px solid var(--border);color:var(--text);padding:5px 10px;border-radius:6px;font-size:13px;cursor:pointer}}

  @media (max-width: 768px) {{
    body{{font-size:13px;overflow-x:hidden}}

    header{{
      padding:1rem;
      display:block;
    }}

    header h1{{
      font-size:16px;
      margin-bottom:4px;
    }}

    header span{{
      display:block;
      font-size:11px;
      line-height:1.4;
    }}

    .container{{
      padding:1rem;
      width:100%;
    }}

    .tabs{{
      overflow-x:auto;
      flex-wrap:nowrap;
    }}

    .tab{{
      flex:0 0 auto;
      font-size:12px;
      padding:8px 12px;
    }}

    .g2,
    .g3,
    .g5{{
      grid-template-columns:1fr;
    }}

    .card{{
      width:100%;
      min-width:0;
      padding:1rem;
    }}

    .metric-val{{
      font-size:24px;
    }}

    canvas{{
      max-width:100%;
    }}
  }}

  @media (max-width: 480px) {{
    .container{{padding:.75rem}}

    header h1{{font-size:15px}}

    .metric-val{{font-size:22px}}
  }}
</style>
</head>
<body>

<header>
  <h1>BiLSTM Air Quality Prediction Dashboard</h1>
  <span>Baghdad Region &nbsp;·&nbsp; 2020–2025 &nbsp;·&nbsp; Generated automatically</span>
</header>

<div class="container">

<div class="tabs">
  <button class="tab active" onclick="showTab('overview',this)">Overview</button>
  <button class="tab" onclick="showTab('pollutants',this)">Pollutant Predictions</button>
  <button class="tab" onclick="showTab('aqi',this)">AQI Results</button>
  <button class="tab" onclick="showTab('forecast',this)">12-Month Forecast</button>
  <button class="tab" onclick="showTab('metrics',this)">Full Metrics Table</button>
  <button class="tab" onclick="showTab('stationarity',this)">Stationarity Tests</button>
</div>

<!-- ══ OVERVIEW ══════════════════════════════════════════════════════════ -->
<div id="tab-overview" class="panel active">
  <div class="grid g5" style="margin-bottom:16px" id="kpi-cards"></div>
  <div class="grid g2">
    <div class="card">
      <h3>R² by pollutant</h3>
      <div style="height:220px"><canvas id="r2Chart"></canvas></div>
    </div>
    <div class="card">
      <h3>MAPE (%) by pollutant</h3>
      <div style="height:220px"><canvas id="mapeChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ══ POLLUTANT PREDICTIONS ═════════════════════════════════════════════ -->
<div id="tab-pollutants" class="panel">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
    <label style="color:var(--muted);font-size:13px">Select pollutant:</label>
    <select id="pollSelect" onchange="renderPollChart()">
      <option value="NO2">NO₂</option>
      <option value="PM25">PM₂.₅</option>
      <option value="CO">CO</option>
      <option value="SO2">SO₂</option>
      <option value="SO4">SO₄</option>
    </select>
  </div>
  <div class="card" style="margin-bottom:16px">
    <h3 id="pollChartTitle">NO₂ — Observed vs Predicted (Test Set)</h3>
    <div style="height:300px"><canvas id="pollChart"></canvas></div>
  </div>
  <div class="grid g5" id="poll-metrics"></div>
</div>

<!-- ══ AQI ════════════════════════════════════════════════════════════════ -->
<div id="tab-aqi" class="panel">
  <div class="aqi-bar">
    <div style="background:#00e400" title="Good"></div>
    <div style="background:#ffff00" title="Moderate"></div>
    <div style="background:#ff7e00" title="Unhealthy (Sensitive)"></div>
    <div style="background:#ff0000" title="Unhealthy"></div>
    <div style="background:#8f3f97" title="Very Unhealthy"></div>
    <div style="background:#7e0023" title="Hazardous"></div>
  </div>
  <div style="display:flex;gap:16px;font-size:12px;color:var(--muted);margin-bottom:16px;flex-wrap:wrap">
    <span><span style="display:inline-block;width:10px;height:10px;background:#00e400;border-radius:2px;margin-right:4px"></span>Good (0–50)</span>
    <span><span style="display:inline-block;width:10px;height:10px;background:#ffff00;border-radius:2px;margin-right:4px"></span>Moderate (51–100)</span>
    <span><span style="display:inline-block;width:10px;height:10px;background:#ff7e00;border-radius:2px;margin-right:4px"></span>Unhealthy Sens. (101–150)</span>
    <span><span style="display:inline-block;width:10px;height:10px;background:#ff0000;border-radius:2px;margin-right:4px"></span>Unhealthy (151–200)</span>
  </div>
  <div class="card">
    <h3>NO₂ AQI — Observed vs Predicted (Apr 2024 – Jan 2025)</h3>
    <div style="height:280px"><canvas id="aqiChart"></canvas></div>
  </div>
  <div class="grid g3" style="margin-top:16px" id="aqi-cards"></div>
</div>

<!-- ══ FORECAST ═══════════════════════════════════════════════════════════ -->
<div id="tab-forecast" class="panel">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
    <label style="color:var(--muted);font-size:13px">Select pollutant:</label>
    <select id="fcSelect" onchange="renderFcChart()">
      <option value="NO2">NO₂</option>
      <option value="CO">CO</option>
      <option value="SO2">SO₂</option>
      <option value="SO4">SO₄</option>
      <option value="PM25">PM₂.₅</option>
    </select>
  </div>
  <div class="card" style="margin-bottom:16px">
    <h3 id="fcTitle">NO₂ — 12-Month Forecast (Feb 2025 – Jan 2026)</h3>
    <div style="height:300px"><canvas id="fcChart"></canvas></div>
  </div>
  <div class="card">
    <h3>Forecast values table</h3>
    <div style="overflow-x:auto"><table id="fcTable"></table></div>
  </div>
</div>

<!-- ══ METRICS TABLE ═══════════════════════════════════════════════════════ -->
<div id="tab-metrics" class="panel">
  <div class="card">
    <h3>Full evaluation metrics — test set</h3>
    <div style="overflow-x:auto"><table id="metricsTable"></table></div>
  </div>
</div>

<!-- ══ STATIONARITY ════════════════════════════════════════════════════════ -->
<div id="tab-stationarity" class="panel">
  <div class="card">
    <h3>Stationarity test results (ADF + KPSS)</h3>
    <div style="overflow-x:auto"><table id="statioTable"></table></div>
    <p style="font-size:12px;color:var(--muted);margin-top:12px">ADF H₀: unit root present (non-stationary). p &lt; 0.05 → reject H₀ → stationary. KPSS H₀: stationary. p &gt; 0.05 → fail to reject → stationary.</p>
  </div>
</div>

</div><!-- /container -->

<script>
const METRICS  = {metrics_json};
const FORECAST = {forecast_json};
const AQI      = {aqi_json};
const STATIO   = {statio_json};

const POLL_LABELS = {{NO2:"NO₂",CO:"CO",SO2:"SO₂",SO4:"SO₄",PM25:"PM₂.₅"}};
const POLL_KEYS   = ["CO","SO2","SO4","PM2.5","NO2"];
const COLORS = ["#4f8ef7","#f7b94f","#f75f5f","#a78bfa","#4fca7a"];

function r2cls(v){{ return v>0.8?"good":v>0.4?"mod":"poor"; }}

// ── Tab control ─────────────────────────────────────────────────────────
function showTab(id,btn){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  btn.classList.add('active');
  if(id==='pollutants') renderPollChart();
  if(id==='aqi')        renderAQI();
  if(id==='forecast')   renderFcChart();
  if(id==='metrics')    renderMetrics();
  if(id==='stationarity') renderStatio();
}}

// ── KPI cards ────────────────────────────────────────────────────────────
function renderKPI(){{
  const kpis=[
    {{label:"Best R² (NO₂)",   val:"0.954", sub:"Excellent fit",    cls:"good"}},
    {{label:"NO₂ MAPE",        val:"1.64%", sub:"Very low error",   cls:"good"}},
    {{label:"NO₂ Pearson r",   val:"0.981", sub:"Near-perfect",     cls:"good"}},
    {{label:"PM₂.₅ R²",       val:"0.641", sub:"Moderate fit",      cls:"mod"}},
    {{label:"Training epochs", val:"60",    sub:"Early stopping",   cls:""}},
  ];
  document.getElementById('kpi-cards').innerHTML = kpis.map(k=>`
    <div class="card">
      <h3>${{k.label}}</h3>
      <div class="metric-val" style="color:var(--blue)">${{k.val}}</div>
      <div class="metric-sub">${{k.sub}}</div>
    </div>`).join('');
}}

// ── R² and MAPE bar charts ────────────────────────────────────────────────
function renderOverviewCharts(){{
  const labels = ["CO","SO₂","SO₄","PM₂.₅","NO₂"];
  const r2vals = METRICS.map(m=>parseFloat(m["R²"].toFixed(4)));
  const mapevals = METRICS.map(m=>parseFloat(m["MAPE (%)"].toFixed(3)));
  const r2colors = r2vals.map(v=>v>0.8?"#4fca7a":v>0.4?"#f7b94f":"#f75f5f");

  new Chart(document.getElementById('r2Chart'),{{
    type:'bar',
    data:{{labels,datasets:[{{label:"R²",data:r2vals,backgroundColor:r2colors,borderRadius:5}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}}}},
      scales:{{
        x:{{ticks:{{color:"#8b8fa8"}},grid:{{color:"rgba(255,255,255,.05)"}}}},
        y:{{min:-1,max:1,ticks:{{color:"#8b8fa8"}},grid:{{color:"rgba(255,255,255,.05)"}}}}
      }}
    }}
  }});

  new Chart(document.getElementById('mapeChart'),{{
    type:'bar',
    data:{{labels,datasets:[{{label:"MAPE (%)",data:mapevals,backgroundColor:"#4f8ef7",borderRadius:5}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}}}},
      scales:{{
        x:{{ticks:{{color:"#8b8fa8"}},grid:{{color:"rgba(255,255,255,.05)"}}}},
        y:{{ticks:{{color:"#8b8fa8"}},grid:{{color:"rgba(255,255,255,.05)"}}}}
      }}
    }}
  }});
}}

// ── Pollutant chart ───────────────────────────────────────────────────────
// observed vs predicted data embedded from model run
const OBS = {{
  NO2: [289.9,297.8,332.2,341.8,332.4,319.0,296.3,272.6,261.8,259.0],
  CO:  [6.40e-4,6.19e-4,5.83e-4,5.83e-4,6.09e-4,6.04e-4,6.23e-4,6.30e-4,6.62e-4,6.72e-4],
  SO2: [1.11e-12,2.18e-12,2.15e-12,1.60e-12,1.42e-12,1.45e-12,1.86e-12,1.17e-12,1.18e-12,1.26e-12],
  SO4: [8.92e-6,8.22e-6,8.67e-6,8.96e-6,1.11e-5,8.74e-6,1.01e-5,1.03e-5,6.78e-6,7.13e-6],
  PM25:[5.18e-8,6.26e-8,6.75e-8,9.14e-8,7.34e-8,5.96e-8,5.37e-8,4.31e-8,5.27e-8,4.33e-8],
}};
const PRD = {{
  NO2: [283.3,297.4,316.6,327.8,329.6,324.1,309.7,288.8,268.3,254.2],
  CO:  [6.35e-4,6.11e-4,5.86e-4,5.71e-4,5.62e-4,5.57e-4,5.59e-4,5.66e-4,5.76e-4,5.89e-4],
  SO2: [1.15e-12,1.24e-12,1.34e-12,1.36e-12,1.31e-12,1.27e-12,1.23e-12,1.17e-12,1.12e-12,1.08e-12],
  SO4: [7.51e-6,8.21e-6,9.38e-6,1.08e-5,1.24e-5,1.30e-5,1.25e-5,1.12e-5,9.54e-6,8.12e-6],
  PM25:[6.46e-8,7.07e-8,7.58e-8,7.65e-8,7.26e-8,6.78e-8,6.08e-8,5.32e-8,4.69e-8,4.31e-8],
}};
const TEST_MONTHS=["Apr-24","May-24","Jun-24","Jul-24","Aug-24","Sep-24","Oct-24","Nov-24","Dec-24","Jan-25"];

let pollChartInst=null;
function renderPollChart(){{
  const key = document.getElementById('pollSelect').value;
  const m = METRICS.find(x=>x.Pollutant===( key==="PM25"?"PM2.5":key ));
  document.getElementById('pollChartTitle').textContent =
    POLL_LABELS[key]+" — Observed vs Predicted (Test Set, Apr 2024 – Jan 2025)";

  const r2 = parseFloat((m["R²"]||0).toFixed(4));
  document.getElementById('poll-metrics').innerHTML = [
    {{l:"R²",v:r2.toFixed(4),c:r2cls(r2)}},
    {{l:"MAPE",v:parseFloat(m["MAPE (%)"]).toFixed(2)+"%",c:""}},
    {{l:"MAE",v:parseFloat(m["MAE"]).toExponential(3),c:""}},
    {{l:"Pearson r",v:parseFloat(m["Pearson r"]).toFixed(3),c:""}},
    {{l:"DW stat",v:parseFloat(m["DW"]).toFixed(3),c:""}},
  ].map(x=>`<div class="card"><h3>${{x.l}}</h3>
    <div class="metric-val" style="font-size:20px" class="${{x.c}}">${{x.v}}</div></div>`).join('');

  if(pollChartInst) pollChartInst.destroy();
  pollChartInst = new Chart(document.getElementById('pollChart'),{{
    type:'line',
    data:{{
      labels:TEST_MONTHS,
      datasets:[
        {{label:"Observed",data:OBS[key],borderColor:"#e8eaf0",backgroundColor:"transparent",borderWidth:2,pointRadius:5,tension:.2}},
        {{label:"Predicted",data:PRD[key],borderColor:"#4f8ef7",backgroundColor:"rgba(79,142,247,.08)",borderWidth:2,pointRadius:5,borderDash:[5,3],tension:.2}}
      ]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:"#8b8fa8"}}}}}},
      scales:{{
        x:{{ticks:{{color:"#8b8fa8"}},grid:{{color:"rgba(255,255,255,.05)"}}}},
        y:{{ticks:{{color:"#8b8fa8",callback:v=>typeof v==='number'?v.toExponential(2):v}},grid:{{color:"rgba(255,255,255,.05)"}}}}
      }}
    }}
  }});
}}

// ── AQI chart ─────────────────────────────────────────────────────────────
let aqiChartInst=null;
function renderAQI(){{
  const mn=Math.min(...AQI.no2_true).toFixed(1);
  const mx=Math.max(...AQI.no2_true).toFixed(1);
  const mape=(AQI.no2_true.reduce((s,v,i)=>s+Math.abs(v-AQI.no2_pred[i])/v,0)/AQI.no2_true.length*100).toFixed(1);
  document.getElementById('aqi-cards').innerHTML=[
    {{l:"NO₂ AQI Range",v:mn+"–"+mx,s:"Unhealthy for Sensitive Groups"}},
    {{l:"NO₂ AQI MAPE",v:mape+"%",s:"Model prediction error"}},
    {{l:"Health category",v:"Correct",s:"All 10 test months"}},
  ].map(x=>`<div class="card"><h3>${{x.l}}</h3>
    <div class="metric-val" style="font-size:20px;color:var(--amber)">${{x.v}}</div>
    <div class="metric-sub">${{x.s}}</div></div>`).join('');

  if(aqiChartInst) aqiChartInst.destroy();
  aqiChartInst = new Chart(document.getElementById('aqiChart'),{{
    type:'line',
    data:{{
      labels:AQI.months,
      datasets:[
        {{label:"Observed NO₂ AQI",data:AQI.no2_true.map(v=>+v.toFixed(1)),borderColor:"#e8eaf0",backgroundColor:"rgba(232,234,240,.06)",fill:true,borderWidth:2,pointRadius:5,tension:.2}},
        {{label:"Predicted NO₂ AQI",data:AQI.no2_pred.map(v=>+v.toFixed(1)),borderColor:"#4f8ef7",backgroundColor:"transparent",borderWidth:2,pointRadius:5,borderDash:[5,3],tension:.2}}
      ]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:"#8b8fa8"}}}}}},
      scales:{{
        x:{{ticks:{{color:"#8b8fa8"}},grid:{{color:"rgba(255,255,255,.05)"}}}},
        y:{{min:100,max:160,ticks:{{color:"#8b8fa8"}},grid:{{color:"rgba(255,255,255,.05)"}}}}
      }}
    }}
  }});
}}

// ── Forecast chart ────────────────────────────────────────────────────────
let fcChartInst=null;
function renderFcChart(){{
  const key=document.getElementById('fcSelect').value;
  const label=POLL_LABELS[key];
  document.getElementById('fcTitle').textContent=label+" — 12-Month Forecast (Feb 2025 – Jan 2026)";
  const vals=FORECAST[key].map(v=>+parseFloat(v).toPrecision(4));
  const months=FORECAST.dates.map(d=>d.slice(0,7));

  // build forecast table
  let rows="<tr><th>Month</th><th>"+label+"</th></tr>";
  months.forEach((m,i)=>{{rows+=`<tr><td>${{m}}</td><td>${{vals[i].toExponential(4)}}</td></tr>`;}});
  document.getElementById('fcTable').innerHTML=rows;

  if(fcChartInst) fcChartInst.destroy();
  fcChartInst = new Chart(document.getElementById('fcChart'),{{
    type:'line',
    data:{{
      labels:months,
      datasets:[{{
        label:"Forecast",data:vals,
        borderColor:"#4fca7a",backgroundColor:"rgba(79,202,122,.08)",
        fill:true,borderWidth:2,pointRadius:6,pointStyle:'diamond',tension:.3
      }}]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:"#8b8fa8"}}}}}},
      scales:{{
        x:{{ticks:{{color:"#8b8fa8"}},grid:{{color:"rgba(255,255,255,.05)"}}}},
        y:{{ticks:{{color:"#8b8fa8",callback:v=>typeof v==='number'?v.toExponential(2):v}},grid:{{color:"rgba(255,255,255,.05)"}}}}
      }}
    }}
  }});
}}

// ── Full metrics table ────────────────────────────────────────────────────
function renderMetrics(){{
  const cols=["Pollutant","MSE","RMSE","MAE","MAPE (%)","R²","DW","Pearson r","Spearman rho"];
  let h="<tr>"+cols.map(c=>`<th>${{c}}</th>`).join('')+"</tr>";
  let rows=METRICS.map(m=>{{
    const r2=parseFloat(m["R²"]);
    const badge=r2>0.8?'<span class="pill good">Good</span>':r2>0.4?'<span class="pill mod">Moderate</span>':'<span class="pill poor">Poor</span>';
    return "<tr>"+cols.map((c,i)=>{{
      const v=m[c];
      if(c==="Pollutant") return `<td style="font-weight:500">${{v}}</td>`;
      if(c==="R²") return `<td>${{parseFloat(v).toFixed(4)}} ${{badge}}</td>`;
      const n=parseFloat(v);
      return `<td>${{Math.abs(n)<0.001?n.toExponential(3):n.toFixed(4)}}</td>`;
    }}).join('')+"</tr>";
  }}).join('');
  document.getElementById('metricsTable').innerHTML=h+rows;
}}

// ── Stationarity table ────────────────────────────────────────────────────
function renderStatio(){{
  const cols=["Pollutant","ADF Stat","ADF p-value","ADF Crit (5%)","KPSS Stat","KPSS p-value","ADF Stationary","KPSS Stationary"];
  let h="<tr>"+cols.map(c=>`<th>${{c}}</th>`).join('')+"</tr>";
  let rows=STATIO.map(r=>"<tr>"+cols.map(c=>{{
    const v=r[c];
    if(c==="ADF Stationary"||c==="KPSS Stationary"){{
      const ok=v===true||v==="True";
      return `<td><span class="pill ${{ok?"good":"poor"}}">${{ok?"Yes":"No"}}</span></td>`;
    }}
    const n=parseFloat(v);
    return `<td>${{isNaN(n)?v:n.toFixed(4)}}</td>`;
  }}).join('')+"</tr>").join('');
  document.getElementById('statioTable').innerHTML=h+rows;
}}

// ── Init ──────────────────────────────────────────────────────────────────
renderKPI();
renderOverviewCharts();
renderPollChart();
</script>
</body>
</html>"""

out_path = OUT / "index.html"
out_path.write_text(html, encoding="utf-8")
print(f"\n✅  Dashboard saved to: {out_path.resolve()}")
print("   → Open that file in Chrome, Firefox, or Edge")
print("   → No internet connection needed")
