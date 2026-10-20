import {complete, compareWithExpected, loadModel, parsePointFile, FOV_DEG, SELFTEST_TOL, ZONE_PITCH_DEG} from './pipeline.js';

const PRED = '#0b8fd6';  // colour of the predicted (completed) object
const SENSOR = '#d9480f';
const SWATCH = {'Red bottle': '#e63946', 'Blue bottle': '#1d78e6', 'Box': '#d98e04'};
const $ = id => document.getElementById(id);

let session = null, manifest = null, busy = false, last = null;

ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
ort.env.wasm.numThreads = 1;  // static hosts and Android WebView don't provide cross-origin isolation
ort.env.wasm.proxy = true;  // run inference in a web worker so the page stays responsive

function setStatus(text, pct, state) {
  $('status-text').textContent = text;
  $('status-pct').textContent = pct == null ? '' : `${Math.round(pct * 100)}%`;
  if (pct != null) $('bar-fill').style.width = `${pct * 100}%`;
  $('model-status').className = `card status ${state || ''}`;
}

function describeModel(m) {
  const note = $('model-note');
  const name = (m.checkpoint || 'unknown').replace(/\.pth$/, '');
  if (m.frame === 'canonical-v2') {
    note.className = 'card notice';
    note.innerHTML = `<b>Model: ${name}.</b> Fine-tuned on simulated hand-held scans that went through the same preprocessing as this page. `
      + 'It was never trained or tuned on the 10 real sample scans.';
  } else {
    note.className = 'card notice warn';
    note.innerHTML = `<b>Model: ${name} (earlier model).</b> It was trained with a different input normalization than the preprocessing this page now uses, `
      + 'so its completions of the real scans below are <b>not meaningful</b>; they only demonstrate that the pipeline runs. '
      + 'Replace it by building the web app with a checkpoint trained on the new preprocessing (<code>checkpoints/adapointr_v3.pth</code>).';
  }
}

async function init() {
  const samples = await (await fetch('samples/samples.json')).json();
  renderSamples(samples);
  $('file').addEventListener('change', onUpload);
  $('show-bg').addEventListener('change', () => last && drawScene(last));
  $('dl-ply').addEventListener('click', () => last && download(`${last.stem}_completed.ply`, plyBlob(last.res.completed, PRED)));
  $('dl-npy').addEventListener('click', () => last && download(`${last.stem}_completed.npy`, npyBlob(last.res.completed)));
  $('dl-scene').addEventListener('click', () => last && download(`${last.stem}_scene.ply`, plyBlob(last.res.completed, PRED, last.res.background)));
  try {
    const native = window.Capacitor?.isNativePlatform?.();  // Android app: the model ships inside the app
    if (native) $('dl-hint').textContent = 'The model is built into the app; no internet needed.';
    ({session, manifest} = await loadModel('model/', p => p < 1 ? setStatus(native ? 'Loading the AI model…' : 'Downloading the AI model…', p)
      : setStatus('Preparing the model on your device (about 20–40 s)…', 1)));
    describeModel(manifest);
    setStatus('Model ready. Pick a scan below.', null, 'ready');
    if (new URLSearchParams(location.search).has('selftest')) return selfTest(samples);
    runSample(samples[samples.length - 1]);
  } catch (e) {
    console.error(e);
    setStatus(`Could not load the model: ${e.message}`, null, 'error');
  }
}

function renderSamples(samples) {
  const groups = {};
  for (const s of samples) (groups[s.user_label] ||= []).push(s);
  const root = $('samples');
  for (const [label, list] of Object.entries(groups)) {
    const g = document.createElement('div');
    g.className = 'group';
    g.innerHTML = `<h3><span class="swatch" style="background:${SWATCH[label] || '#888'}"></span>${label} scans</h3><div class="btns"></div>`;
    list.forEach((s, i) => {
      const b = document.createElement('button');
      b.className = 'btn';
      b.textContent = `Scan ${i + 1}`;
      b.title = s.file;
      b.dataset.file = s.file;
      b.addEventListener('click', () => runSample(s));
      g.querySelector('.btns').appendChild(b);
    });
    root.appendChild(g);
  }
}

async function runSample(s) {
  document.querySelectorAll('.group .btn').forEach(b => b.classList.toggle('active', b.dataset.file === s.file));
  const buf = await (await fetch(`samples/${s.file}`)).arrayBuffer();
  const group = document.querySelector(`.group .btn[data-file="${s.file}"]`).closest('.group');
  const idx = [...group.querySelectorAll('.btn')].findIndex(b => b.dataset.file === s.file);
  await run(s.file.replace(/\.npy$/, ''), buf, s.file, s.user_label, `${s.user_label} scan ${idx + 1}`);
}

async function onUpload(e) {
  const f = e.target.files[0];
  if (!f) return;
  document.querySelectorAll('.group .btn').forEach(b => b.classList.remove('active'));
  await run(f.name.replace(/\.[^.]+$/, ''), await f.arrayBuffer(), f.name, null, f.name);
  e.target.value = '';
}

async function run(stem, buffer, filename, userLabel, title) {
  if (!session) return setStatus('The model is still loading. Please wait a moment…', null);
  if (busy) return;
  busy = true;
  document.querySelectorAll('.btn').forEach(b => b.disabled = true);
  setStatus(`Completing ${title}… (a few seconds)`, null, 'ready');
  try {
    const pts = parsePointFile(filename, buffer);
    await new Promise(r => setTimeout(r, 30));  // let the status paint before the heavy work
    const t = performance.now();
    const res = await complete(session, pts);
    const secs = (performance.now() - t) / 1000;
    last = {stem, userLabel, res, title};
    showResult(last, pts.length / 3, secs);
    setStatus(`Done: ${title}`, null, 'ready');
  } catch (err) {
    console.error(err);
    setStatus(`Could not process ${filename}: ${err.message}`, null, 'error');
  } finally {
    busy = false;
    document.querySelectorAll('.btn').forEach(b => b.disabled = false);
  }
}

// ---------- what the scan says about the object's position (sensor-relative, never centimetres) ----------
function geometryReport(meta) {
  const reasons = [];
  if (meta.mode !== 'stripes') reasons.push('the 8 background stripes of the sensor could not be fitted in this file, so the sensor position inside the scan is unknown');
  if (meta.scale_clamped) reasons.push('the object is at noise distance from the sensor');
  if (meta.n_object_points < 100) reasons.push(`only ${meta.n_object_points} points belong to the object`);
  const ratio = meta.object_depth_ratio, ang = meta.angular_size_deg;
  let distance = null, angle = null;
  if (meta.confident && Number.isFinite(ratio)) {
    distance = {ratio, closer: 1 / ratio};
    if (Number.isFinite(ang) && ang <= FOV_DEG) angle = {deg: ang, zones: ang / ZONE_PITCH_DEG};
  }
  return {confident: meta.confident, distance, angle, reasons,
    angleWhy: meta.confident && !angle
      ? `the measured spread (${Number.isFinite(ang) ? ang.toFixed(0) + '°' : 'n/a'}) exceeds the sensor's ${FOV_DEG}° field of view, which one view cannot show (likely an object held very close and smeared by hand motion)`
      : null};
}

function renderGeometry(meta) {
  const g = geometryReport(meta);
  const rows = [];
  const na = why => `<span class="na">not recoverable</span><small>${why}</small>`;
  const noSensor = g.reasons.length ? `Because ${g.reasons.join('; ')}.` : 'The fit was not confident.';
  rows.push(`<div class="kv"><span>Distance, relative to the background</span><div>${g.distance
    ? `<b>${(g.distance.ratio * 100).toFixed(0)}%</b> of the sensor-to-background distance <small>(about ${g.distance.closer.toFixed(0)}× closer than the background; rough, see below)</small>`
    : na(noSensor)}</div></div>`);
  rows.push(`<div class="kv"><span>Angular size seen by the sensor</span><div>${g.angle
    ? `<b>about ${g.angle.deg.toFixed(0)}°</b> <small>(≈ ${g.angle.zones.toFixed(1)} sensor zones of ${ZONE_PITCH_DEG}° each)</small>`
    : na(g.angleWhy || noSensor)}</div></div>`);
  rows.push('<div class="kv"><span>Real size or distance in cm</span><div><span class="na">not recoverable</span>'
    + '<small>the export divides by the largest coordinate, which erases the metric scale</small></div></div>');
  $('geo-rows').innerHTML = rows.join('');
}

function showResult(r, nIn, secs) {
  $('result').classList.remove('hidden');
  $('r-title').textContent = r.title;
  $('r-chip').textContent = r.userLabel ? `Your label: ${r.userLabel}` : 'Your own file';
  $('r-chip').style.background = r.userLabel ? SWATCH[r.userLabel] : '#666';
  $('legend-pred').style.background = PRED;
  const m = r.res.meta;
  $('r-stats').textContent = `${m.n_input_unique.toLocaleString()} distinct points in this one scan file → ${r.res.observed.length.toLocaleString()} belong to the object → `
    + `${r.res.completed.length.toLocaleString()} points in the one completed object · ${secs.toFixed(1)} s on this device`;
  $('r-label-note').textContent = r.userLabel
    ? `“${r.userLabel}” is how the person who made this scan described it. The model is never given it and does not predict it.` : '';
  renderGeometry(m);
  drawObject(r);
  drawScene(r);
}

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function trace(pts, color, name, size, opacity, max = 9000) {
  const step = Math.max(1, Math.ceil(pts.length / max));
  const p = pts.filter((_, i) => i % step === 0);
  // y is up: plot (x, z, y)
  return {type: 'scatter3d', mode: 'markers', name, x: p.map(q => q[0]), y: p.map(q => q[2]), z: p.map(q => q[1]),
    marker: {size, color, opacity}, hoverinfo: 'skip'};
}

function layout() {
  const ink = cssVar('--muted'), grid = cssVar('--line');
  const axis = title => ({title, color: ink, gridcolor: grid, zerolinecolor: grid, showbackground: false});
  return {margin: {l: 0, r: 0, t: 0, b: 0}, paper_bgcolor: cssVar('--plot-bg'), showlegend: false,
    scene: {aspectmode: 'data', xaxis: axis('x'), yaxis: axis('depth'), zaxis: axis('up'),
      camera: {eye: {x: -1.5, y: -1.5, z: 0.8}}}};
}

const plotConfig = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toImage']};

function drawObject(r) {
  Plotly.react('plot-object', [
    trace(r.res.completed, PRED, 'completed', 1.6, 0.45),
    trace(r.res.observed, cssVar('--seen'), 'seen', 2.4, 1),
  ], layout(), plotConfig);
}

function drawScene(r) {
  const traces = [];
  if ($('show-bg').checked) traces.push(trace(r.res.background, '#9aa0a6', 'background', 1.2, 0.35));
  traces.push(trace(r.res.completed, PRED, 'completed', 1.2, 0.6));
  traces.push(trace(r.res.observed, cssVar('--seen'), 'seen', 1.8, 1));
  const s = r.res.meta.sensor_estimate;
  if (s) traces.push(trace([s.o], SENSOR, 'sensor (estimated)', 6, 1));  // height is assumed, not measured
  Plotly.react('plot-scene', traces, layout(), plotConfig);
}

// ---------- downloads ----------
async function download(name, blob) {
  const plugins = window.Capacitor?.isNativePlatform?.() && window.Capacitor.Plugins;
  if (plugins?.Filesystem && plugins?.Share) {  // Android app: save to cache and open the share sheet
    const b64 = await new Promise(res => {
      const fr = new FileReader();
      fr.onload = () => res(fr.result.split(',')[1]);
      fr.readAsDataURL(blob);
    });
    const {uri} = await plugins.Filesystem.writeFile({path: name, data: b64, directory: 'CACHE'});
    await plugins.Share.share({title: name, files: [uri]});
    return;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

function plyBlob(pts, hex, background = []) {
  const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  const rows = pts.map(p => `${p[0].toFixed(6)} ${p[1].toFixed(6)} ${p[2].toFixed(6)} ${r} ${g} ${b}`)
    .concat(background.map(p => `${p[0].toFixed(6)} ${p[1].toFixed(6)} ${p[2].toFixed(6)} 170 170 170`));
  const header = `ply\nformat ascii 1.0\nelement vertex ${rows.length}\nproperty float x\nproperty float y\nproperty float z\n`
    + 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n';
  return new Blob([header, rows.join('\n'), '\n'], {type: 'text/plain'});
}

function npyBlob(pts) {
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${pts.length}, 3), }`;
  header += ' '.repeat(64 - ((10 + header.length + 1) % 64)) + '\n';
  const pre = new Uint8Array(10);
  pre.set([0x93, 78, 85, 77, 80, 89, 1, 0]);
  new DataView(pre.buffer).setUint16(8, header.length, true);
  return new Blob([pre, header, Float32Array.from(pts.flat())], {type: 'application/octet-stream'});
}

// ---------- self-test against the Python pipeline (?selftest) ----------
// Per file: preprocessing must reproduce majproj.preprocess exactly (mode, object point count, transform centre and
// scale, depth ratio, angular size, and a bit-exact fingerprint of the 2048 x 3 float32 partial); the model output
// must agree with PyTorch in one-sided Chamfer-L1 (canonical units) up to float32 / wasm numerical noise.
async function selfTest(samples) {
  const expected = await (await fetch('samples/expected.json')).json();
  const lines = [`model ${expected.checkpoint} (frame ${expected.frame}); tolerances: preprocessing ${SELFTEST_TOL.pre}, model one-sided CD ${SELFTEST_TOL.cd}`];
  const results = [];
  let ok = true;
  for (const s of samples) {
    const pts = parsePointFile(s.file, await (await fetch(`samples/${s.file}`)).arrayBuffer());
    const res = await complete(session, pts);
    const cmp = compareWithExpected(res, expected.files[s.file]);
    ok &&= cmp.pass;
    results.push({file: s.file, ...cmp});
    lines.push(`${cmp.pass ? 'PASS' : 'FAIL'} ${s.file}: ${cmp.line}`);
  }
  document.title = ok ? 'SELFTEST PASS' : 'SELFTEST FAIL';
  window.__selftest = {ok, lines, results};
  const pre = document.createElement('pre');
  pre.id = 'selftest';
  pre.textContent = `${ok ? 'ALL PASS' : 'SOME FAILED'}\n${lines.join('\n')}`;
  $('result').before(pre);
}

init();
