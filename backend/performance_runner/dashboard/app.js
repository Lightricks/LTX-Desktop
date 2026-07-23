"use strict";
const $ = s => document.querySelector(s);
// Escape server-provided strings before they go into innerHTML. Run labels, scenario
// titles etc. originate from run params / the catalog; escaping keeps them from injecting
// markup into this page (which can POST to /api/run and /api/token).
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// What each run kind does + why — surfaced as tooltips on the buttons and rows.
const KIND_INFO = {
  soak: "Compile-on leak gate (the primary ship-gate). Runs N generations back-to-back in one process and checks the inter-generation VRAM floor stays flat. A rising floor means the cache's .to('meta') evict isn't reclaiming (compiled-module / cudagraph / lingering ref). Must run with Torch Compile ON; needs ~40–50 gens for a slow creep to show.",
  ab: "Correctness gate. Fixed seed, identical config, cache OFF then ON; diffs the outputs (pre-decode latents, or frame PSNR) to prove the cache key captures everything and a hit reuses the RIGHT model.",
  decompose: "Wall-time saving. Runs baseline (patch OFF) vs patch_only (patch ON) and reports the per-generation rebuild time the patch eliminates on a stage_2 cache hit.",
  sanity: "Feature smoke sweep. Runs the wired scenarios end-to-end and emits a pass/fail matrix — confirms the whole Desktop surface still generates. Also records each scenario's peak VRAM (which card tier it fits) and checks the output matches the request (resolution/duration/fps/audio).",
  coldstart: "First-generation penalty. Times the cold gen 0 (model load / disk read / build) against the warm median. Run right after a fresh backend start for a true cold reading — this is the latency the user feels on first launch.",
};

const jget = async u => (await fetch(u)).json();
const jpost = async (u, b) => (await fetch(u, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(b)})).json();

let selected = null;          // selected run id (drives the run log + chart)
let logSource = 'run';        // 'run' = selected run's log · 'backend' = session log
let toggleSyncPausedUntil = 0; // don't let a poll clobber a just-clicked toggle

// ==== token + toggles ====================================================== //
async function setToken(){
  const v = $('#tokIn').value.trim();
  if(!v){ $('#tgMsg').textContent = 'paste a token first'; return; }
  await jpost('/api/token', {token:v});
  $('#tokIn').value = '';
  $('#tgMsg').textContent = 'token set';
  refreshTop();
}

async function toggle(route, checked, label){
  toggleSyncPausedUntil = Date.now() + 2500;
  try { await jpost(route, {enabled:checked}); $('#tgMsg').textContent = `${label} ${checked?'ON':'OFF'}`; }
  catch { $('#tgMsg').textContent = `${label} toggle failed (token?)`; }
}

let envKnown = false;
async function refreshEnv(){
  if(envKnown) return;                       // platform doesn't change — resolve once
  let e; try { e = await jget('/api/env'); } catch { return; }
  if(e.cuda == null) return;                 // unknown yet (token not set?) — retry next tick
  envKnown = true;
  // Off CUDA (e.g. Apple Silicon / MPS): Torch Compile and the diffusion-stage-cache
  // patch are no-ops, so hide their toggles and the patch/compile-specific runs — only
  // the scenario sweeps (and general metrics like cold-start) apply.
  document.body.classList.toggle('no-cuda', !e.cuda);
}

async function syncToggles(){
  if(Date.now() < toggleSyncPausedUntil) return;
  try {
    const t = await jget('/api/toggles');
    if(t.cache != null) $('#tgCache').checked = t.cache;
    if(t.torch_compile != null) $('#tgCompile').checked = t.torch_compile;
    if(t.cache == null && t.torch_compile == null) $('#tgMsg').textContent = t.error ? 'settings unreadable (token?)' : '';
    else if($('#tgMsg').textContent === 'reading live settings…') $('#tgMsg').textContent = '';
  } catch {}
}

// ==== launching runs ======================================================= //
function runSummary(kind, p){
  if(kind === 'soak'){
    const n = p.n || 50, mins = Math.max(1, Math.round(n * 80 / 60));
    if(p.pair) return `Start SOAK PAIR — cache_on + cache_off, ${n} gens each (≈ ${mins*2} min total). Compares the two floors. Torch Compile should be ON.`;
    return `Start SOAK — ${n} gens, cache ${(p.set_cache||'on').toUpperCase()}.\n≈ ${mins} min (~80s/gen). Torch Compile should be ON.`;
  }
  if(kind === 'ab') return `Start correctness A/B (2 gens, seed ${p.seed||42}).`;
  if(kind === 'decompose'){ const r = p.reps||5; return `Start decompose — ${r} reps × 2 configs ≈ ${(r+1)*2} gens.`; }
  if(kind === 'coldstart'){ const w = p.warm||3; return `Start cold-start — 1 cold + ${w} warm gens.\nRun right after a fresh backend start for a true cold reading.`; }
  if(kind === 'sanity'){
    const fast = p.fast ? ' — FAST (540p/5s, surface check only)' : '';
    if(p.only && p.only.length) return `Run scenario "${p.only.join(', ')}" end-to-end (1 generation)${fast}.`;
    return `Start sanity sweep${p.include_unwired ? ' (incl. UNWIRED — expect failures)' : ' (wired scenarios)'}${fast}.`;
  }
  return `Start ${kind}?`;
}

async function run(kind, params){
  let msg = runSummary(kind, params);
  if(kind === 'soak'){   // warn if the GPU is already busy (would contaminate wall-time)
    try {
      const g = await jget('/api/gpu-busy');
      if(g.busy) msg += `\n\n⚠ GPU is ~${Math.round(g.util)}% busy — another process is using it. Wall-time will be contaminated; close other GPU apps for a clean run.`;
    } catch {}
  }
  if(!confirm(msg + '\n\nProceed?')) return;
  const r = await jpost('/api/run', {kind, params});
  if(r.id){ selected = r.id; refreshRuns(); }
}

function runScenario(){
  const key = $('#scenSel').value;
  const x = scenarios.find(s => s.key === key);
  if(!key || !x || !scenAvail(x)) return;   // guard: unavailable scenarios don't run
  run('sanity', {only:[key]});
}

async function stopRun(id, ev){
  ev.stopPropagation();
  await jpost(`/api/runs/${id}/stop`, {});
  refreshRuns();
}

async function stopAll(){
  const running = (await jget('/api/runs')).filter(r => r.status.startsWith('running')).length;
  if(!running) return;
  if(!confirm(`Stop ALL ${running} running run(s)? This cannot be undone.`)) return;
  await jpost('/api/runs/stop-all', {});
  refreshRuns();
}

// ==== monitoring: header + scenarios + sanity ============================== //
async function refreshTop(){
  try {
    const h = await jget('/api/health'), b = $('#backend'), t = $('#tok');
    b.textContent = 'backend: ' + h.backend;
    b.style.color = h.backend === 'ok' ? 'var(--ok)' : 'var(--bad)';
    t.textContent = h.token_set ? 'token: set' : 'token: none';
    t.style.color = h.token_set ? 'var(--ok)' : 'var(--bad)';
  } catch {}
  try {
    const g = await jget('/api/gpu'), el = $('#gpu'), gib = m => (m/1024).toFixed(1);
    const disk = g.disk_free_gb != null ? ` · disk ${g.disk_free_gb}GB free` : '';
    const diskLow = g.disk_free_gb != null && g.disk_free_gb < 20;   // models + outputs need room
    if(g.used_mib != null){
      // CUDA: full GPU telemetry (+ host RAM).
      const gb = g.total_mib ? `${gib(g.used_mib)}/${gib(g.total_mib)}GB` : `${g.used_mib}MiB`;
      const ram = g.ram_used_mib != null && g.ram_total_mib ? ` · RAM ${gib(g.ram_used_mib)}/${gib(g.ram_total_mib)}GB` : '';
      el.textContent = `GPU ${g.util??'?'}% · ${gb} · ${g.temp_c??'?'}°C · ${g.clock_mhz??'?'}MHz · ${g.power_w!=null?Math.round(g.power_w):'?'}W${ram}${disk}`;
      // low util + low clock = the GPU is starved/contended, not busy — flag it.
      el.classList.toggle('warn', diskLow || (g.util != null && g.util < 30 && g.clock_mhz != null && g.clock_mhz < 1500));
    } else if(g.ram_total_mib != null){
      // No nvidia-smi (Apple Silicon / MPS): show the GPU name + unified-memory usage, FREE
      // RAM (the number that gates local generation at server start), swap (memory pressure),
      // MPS memory held vs the ceiling, and disk headroom.
      const mpsTag = g.mps && !/mps/i.test(g.name || '') ? ' (MPS)' : '';   // name may already say (MPS)
      const name = g.name ? `${g.name}${mpsTag} · ` : '';
      const free = g.ram_avail_mib != null ? ` · ${gib(g.ram_avail_mib)}GB free` : '';
      const swap = g.swap_used_mib ? ` · swap ${gib(g.swap_used_mib)}GB` : '';
      const mpsMem = (g.mps_driver_mib && g.mps_max_mib) ? ` · MPS ${gib(g.mps_driver_mib)}/${gib(g.mps_max_mib)}GB` : '';
      el.textContent = `${name}RAM ${gib(g.ram_used_mib)}/${gib(g.ram_total_mib)}GB${free}${swap}${mpsMem}${disk}`;
      // amber under the Apple Silicon local-gen floor (~15 GB free), when swap climbs, or low disk.
      el.classList.toggle('warn', diskLow || (g.ram_avail_mib != null && g.ram_avail_mib < 15 * 1024)
                                  || (g.swap_used_mib || 0) > 2 * 1024);
    } else {
      el.textContent = 'GPU: n/a';
    }
  } catch {}
}

let scenarios = [];
// A scenario is unavailable when a required LoRA / IC-LoRA isn't downloaded. Weights come
// from the app's library (SSOT /api/loras + /api/ic-loras); download there, then hit refresh.
const scenAvail = x => x.available !== false;
const scenTip = x => scenAvail(x) ? (x.title || '') : (x.unavailable_reason || 'unavailable');

async function refreshScenarios(){
  scenarios = await jget('/api/scenarios');
  const mark = x => !scenAvail(x) ? '⬇' : x.needs_wiring ? '⚠︎' : '✓';
  $('#scenarios').innerHTML = scenarios.map(x =>
    `<div class="${scenAvail(x)?'':'unavail'}" title="${esc(scenTip(x))}">`
    + `${mark(x)} <b>${esc(x.key)}</b> <span class="mut">${esc(x.title)}</span></div>`).join('');
  const cur = $('#scenSel').value;
  // Unavailable scenarios are disabled in the picker (can't be selected to run).
  $('#scenSel').innerHTML = scenarios.map(x =>
    `<option value="${esc(x.key)}"${scenAvail(x)?'':' disabled'}>`
    + `${!scenAvail(x)?'⬇ ':x.needs_wiring?'⚠︎ ':''}${esc(x.key)} — ${esc(x.title)}${scenAvail(x)?'':' (not downloaded)'}</option>`
  ).join('');
  if(cur && scenarios.some(x => x.key === cur && scenAvail(x))) $('#scenSel').value = cur;
  updateRunScenarioBtn();
}

function updateRunScenarioBtn(){
  const x = scenarios.find(s => s.key === $('#scenSel').value);
  const btn = $('#runScenBtn');
  const ok = x && scenAvail(x);
  btn.disabled = !ok;
  btn.title = ok ? 'Run this scenario end-to-end (1 generation)'
                 : (x ? scenTip(x) : 'select a scenario');
}

async function refreshCatalog(){
  const b = $('#refreshCatBtn');
  if(b) b.disabled = true;
  try { await jpost('/api/catalog/refresh', {}); await refreshScenarios(); }
  finally { if(b) b.disabled = false; }
}

// ==== runs table =========================================================== //
function verdictChip(v){
  if(!v) return '';
  const up = v.toUpperCase();
  const cls = up.startsWith('PASS') ? 'v-pass' : (up.startsWith('FAIL') || up.includes('LEAK')) ? 'v-fail' : 'v-other';
  const short = up.startsWith('PASS') ? 'PASS' : up.startsWith('FAIL') ? 'FAIL' : v.split(/[\s(]/)[0];
  return `<span class="${cls}" title="${esc(v)}">${esc(short)}</span>`;
}

function pick(id){ selected = id; refreshRuns(); refreshLog(); }

async function refreshRuns(){
  const rs = await jget('/api/runs');
  $('#runs').innerHTML = rs.map(r => {
    const running = r.status.startsWith('running');
    const cls = running ? 'st-running' : r.status.startsWith('done') ? 'st-done' : 'st-failed';
    const stop = running ? `<button class="stop" onclick="stopRun('${r.id}',event)">stop</button>` : '';
    const info = (KIND_INFO[r.kind] || '').replace(/"/g,'&quot;');
    const result = r.result_url
      ? `<a class="result" href="${esc(r.result_url)}" target="_blank" onclick="event.stopPropagation()">RESULT ↗</a>`
      : '';
    return `<tr class="${r.id===selected?'sel':''}" onclick="pick('${esc(r.id)}')" title="${esc(r.label)} — ${info}">
      <td>${esc(r.id)}</td><td>${esc(r.kind)}</td><td>${esc(r.label)}</td>
      <td class="${cls}">${esc(r.status)}</td><td>${esc(r.progress||'')}</td><td>${verdictChip(r.verdict)}</td><td>${result}</td><td>${r.elapsed_s}s</td><td>${stop}</td></tr>`;
  }).join('');
  // No queue yet: while any run is active, disable the launch buttons (backend
  // serializes) AND the toggles (flipping cache/compile mid-run corrupts it).
  const anyRunning = rs.some(r => r.status.startsWith('running'));
  document.querySelectorAll('.runbtn').forEach(b => {
    if(b.dataset.tip === undefined) b.dataset.tip = b.getAttribute('title') || '';
    b.disabled = anyRunning;
    // Keep each preset's own tooltip visible during a run — just prepend the note.
    b.title = anyRunning ? 'A run is in progress — wait for it to finish.\n\n' + b.dataset.tip : b.dataset.tip;
  });
  $('#tgCache').disabled = anyRunning;
  $('#tgCompile').disabled = anyRunning;
}

// ==== log + chart ========================================================== //
function setLogSrc(s){
  logSource = s;
  $('#lsRun').classList.toggle('active', s === 'run');
  $('#lsBackend').classList.toggle('active', s === 'backend');
  refreshLog();
}

async function refreshLog(){
  const el = $('#log');
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
  if(logSource === 'backend'){
    const r = await jget('/api/backend-log');
    $('#logsel').textContent = r.file ? '· ' + r.file : '';
    el.textContent = r.log || '(empty)';
  } else if(!selected){
    el.textContent = 'select a run…'; $('#logsel').textContent = ''; return;
  } else {
    $('#logsel').textContent = '· ' + selected;
    el.textContent = (await jget(`/api/runs/${selected}/log`)).log || '(empty)';
    if(selected.startsWith('soak')) refreshChart();
    else clearChart();   // non-soak run selected -> don't leave a stale trend up
  }
  if(atBottom) el.scrollTop = el.scrollHeight;
}

let chart = null;
const SERIES_COLORS = ['#4c8dff', '#f85149', '#a371f7', '#2ea043'];   // cache_on, cache_off, ...
const WALL_SOLO = '#d29922';                                          // wall colour for a single run

async function refreshChart(){
  const d = await jget(`/api/runs/${selected}/soakcsv`);
  const series = (d.series || []).filter(s => s.gen && s.gen.length);
  if(!series.length){ clearChart(); return; }             // soak with no data yet -> blank, not stale
  const multi = series.length > 1;                        // --pair: cache_on + cache_off overlaid
  // Grow the inner width with the point count so the panel height stays fixed and the
  // chart scrolls horizontally once wider than the panel (no zoom needed).
  const labels = series.reduce((a, s) => s.gen.length > a.length ? s.gen : a, []);
  $('#chartInner').style.width = Math.max($('#chartScroll').clientWidth || 600, labels.length * 26) + 'px';
  const datasets = [];
  series.forEach((s, i) => {
    const color = SERIES_COLORS[i % SERIES_COLORS.length];
    // Colour encodes the SERIES (cache_on vs cache_off); solid = floor, dashed = wall.
    datasets.push({label:`${s.label} floor MiB`, data:s.floor, yAxisID:'y',
                   borderColor:color, pointRadius:0, tension:.2, spanGaps:true});
    if(!multi && s.ram && s.ram.some(v => v != null))     // host RAM only on a single run (avoid pair clutter)
      datasets.push({label:'host RAM MiB', data:s.ram, yAxisID:'y',
                     borderColor:'#3fb950', pointRadius:0, tension:.2, spanGaps:true});
    datasets.push({label:`${s.label} wall s`, data:s.wall, yAxisID:'y1',
                   borderColor: multi ? color : WALL_SOLO, pointRadius:0, tension:.2,
                   borderDash: multi ? [4,3] : []});
  });
  const data = { labels, datasets };
  if(chart){ chart.data = data; chart.resize(); chart.update(); return; }
  chart = new Chart($('#chart'), {
    type:'line', data,
    options:{ animation:false, responsive:true, maintainAspectRatio:false,
      scales:{ y:{position:'left', title:{display:true, text:'MiB (floor / RAM)'}},
               y1:{position:'right', grid:{drawOnChartArea:false}, title:{display:true, text:'wall s'}} },
      plugins:{ legend:{labels:{color:'#e6e9ef'}} } },
  });
}

function clearChart(){
  if(chart){ chart.data.labels = []; chart.data.datasets.forEach(ds => ds.data = []); chart.update(); }
}

// ==== wire-up + polling ==================================================== //
$('#tgCache').onchange = e => toggle('/api/cache', e.target.checked, 'cache');
$('#tgCompile').onchange = e => toggle('/api/torch-compile', e.target.checked, 'torch compile (applies next gen)');
$('#scenSel').onchange = updateRunScenarioBtn;

const POLLERS = [[refreshTop,3000], [refreshRuns,3000], [refreshLog,3000], [syncToggles,3000], [refreshEnv,3000]];
refreshScenarios();
for(const [fn, ms] of POLLERS){ fn(); setInterval(fn, ms); }
