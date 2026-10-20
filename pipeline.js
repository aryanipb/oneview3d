// Browser port of majproj/preprocess (preprocessing v2) and the analysis/forensics stripe ruler, plus the
// AdaPoinTr ONNX call. Everything runs on-device: file parsing, sensor localisation, object segmentation,
// canonical frame, deterministic farthest-point sampling, the model, and the inverse transform.
//
// Contract: for one exported scan file this produces the SAME partial (2048 x 3 float32, canonical frame) and
// transform {center, scale} as `majproj.preprocess.preprocess`. Nothing is random: every step is a port of a
// deterministic Python step, and the numpy numerics that decide discrete outcomes (percentile interpolation,
// histogram binning, pairwise summation, stable sorts) are mirrored exactly. scripts/build_web.py writes the
// Python outputs into samples/expected.json and `index.html?selftest` compares against them.
//
// NOT ported (reported by the self-test as a mode mismatch when it would matter):
//   * analysis.forensics.estimate_sensor (whole-cloud search, used by Python only if the stripe fit fails AND its
//     score >= 0.55): it relies on numpy's PCG64 sub-sampling and scipy rotations. It was never accepted on any
//     real or simulated file we tried; here a failed stripe fit goes straight to the 'scale_free' mode.
//   * LAPACK eigh in the receding-sheet test is replaced by a 3x3 Jacobi solver: same decision, not bit-identical.
//   * np.argsort(-mass) (only when more than 8 stripe peaks tie in mass) is replaced by a stable sort.

export const N_IN = 2048;          // CANON_POINTS
const MIN_SCALE = 0.03;
const WALL_FRACTION = 0.80;
const STRIPE_REL_RMS_MAX = 0.02;
const STRIPE_MIN_COUNT = 4;
const MIN_OBJECT_POINTS = 40;
const RECEDING_FRACTION = 0.35;
const CONFIDENT_REL_RMS = 0.008;
const CONFIDENT_MIN_STRIPES = 5;
export const ZONE_PITCH_DEG = 5.9;  // one VL53L5CX zone
export const FOV_DEG = 47;          // full horizontal/vertical field of view of the sensor

// BEGIN GENERATED CONSTANTS (scripts/build_web.py checks that these equal analysis.forensics at build time)
const TAN_STRIPE = [-0.39086689308311157, -0.2670726705290085, -0.1595591676417891, -0.052474054725718366, 0.05247405472571835, 0.15955916764178907, 0.2670726705290085, 0.3908668930831116];
const TAN_ROW_UP = 0.42487261826233463;
const ROLL_COS = [0.9396926207859084, 0.9426414910921784, 0.9455185755993168, 0.9483236552061993, 0.9510565162951535, 0.9537169507482269, 0.9563047559630354, 0.958819734868193, 0.9612616959383189, 0.963630453208623, 0.9659258262890683, 0.9681476403781077, 0.9702957262759965, 0.9723699203976766, 0.9743700647852352, 0.9762960071199334, 0.9781476007338057, 0.9799247046208296, 0.981627183447664, 0.9832549075639546, 0.984807753012208, 0.9862856015372314, 0.9876883405951378, 0.9890158633619168, 0.9902680687415704, 0.9914448613738104, 0.992546151641322, 0.9935718556765875, 0.9945218953682733, 0.9953961983671789, 0.9961946980917455, 0.996917333733128, 0.9975640502598242, 0.9981347984218669, 0.9986295347545738, 0.9990482215818578, 0.9993908270190958, 0.9996573249755573, 0.9998476951563913, 0.9999619230641713, 1.0, 0.9999619230641713, 0.9998476951563913, 0.9996573249755573, 0.9993908270190958, 0.9990482215818578, 0.9986295347545738, 0.9981347984218669, 0.9975640502598242, 0.996917333733128, 0.9961946980917455, 0.9953961983671789, 0.9945218953682733, 0.9935718556765875, 0.992546151641322, 0.9914448613738104, 0.9902680687415704, 0.9890158633619168, 0.9876883405951378, 0.9862856015372314, 0.984807753012208, 0.9832549075639546, 0.981627183447664, 0.9799247046208296, 0.9781476007338057, 0.9762960071199334, 0.9743700647852352, 0.9723699203976766, 0.9702957262759965, 0.9681476403781077, 0.9659258262890683, 0.963630453208623, 0.9612616959383189, 0.958819734868193, 0.9563047559630354, 0.9537169507482269, 0.9510565162951535, 0.9483236552061993, 0.9455185755993168, 0.9426414910921784, 0.9396926207859084];  // np.cos(np.deg2rad(d)), d = -20, -19.5 .. 20
const ROLL_SIN = [-0.3420201433256687, -0.3338068592337709, -0.3255681544571567, -0.31730465640509214, -0.3090169943749474, -0.3007057995042731, -0.29237170472273677, -0.28401534470392265, -0.27563735581699916, -0.26723837607825685, -0.25881904510252074, -0.25038000405444144, -0.24192189559966773, -0.2334453638559054, -0.224951054343865, -0.21643961393810288, -0.20791169081775934, -0.1993679344171972, -0.1908089953765448, -0.18223552549214747, -0.17364817766693033, -0.16504760586067765, -0.15643446504023087, -0.14780941112961063, -0.13917310096006544, -0.13052619222005157, -0.12186934340514748, -0.11320321376790672, -0.10452846326765347, -0.09584575252022398, -0.08715574274765817, -0.07845909572784494, -0.0697564737441253, -0.06104853953485687, -0.052335956242943835, -0.043619387365336, -0.03489949670250097, -0.026176948307873153, -0.01745240643728351, -0.008726535498373935, 0.0, 0.008726535498373935, 0.01745240643728351, 0.026176948307873153, 0.03489949670250097, 0.043619387365336, 0.052335956242943835, 0.06104853953485687, 0.0697564737441253, 0.07845909572784494, 0.08715574274765817, 0.09584575252022398, 0.10452846326765347, 0.11320321376790672, 0.12186934340514748, 0.13052619222005157, 0.13917310096006544, 0.14780941112961063, 0.15643446504023087, 0.16504760586067765, 0.17364817766693033, 0.18223552549214747, 0.1908089953765448, 0.1993679344171972, 0.20791169081775934, 0.21643961393810288, 0.224951054343865, 0.2334453638559054, 0.24192189559966773, 0.25038000405444144, 0.25881904510252074, 0.26723837607825685, 0.27563735581699916, 0.28401534470392265, 0.29237170472273677, 0.3007057995042731, 0.3090169943749474, 0.31730465640509214, 0.3255681544571567, 0.3338068592337709, 0.3420201433256687];
// END GENERATED CONSTANTS

// ---------- file parsing ----------
// Every parser returns a flat Float64Array [x0, y0, z0, x1, ...] holding only the finite rows.
export function parsePointFile(name, buffer) {
  const lower = name.toLowerCase();
  if (lower.endsWith('.npy')) return parseNpy(buffer);
  if (lower.endsWith('.ply')) return parsePly(buffer);
  return parseText(new TextDecoder().decode(buffer));
}

function parseNpy(buffer) {
  const u8 = new Uint8Array(buffer);
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.slice(1, 6)) !== 'NUMPY') throw new Error('not a .npy file');
  const major = u8[6];
  const dv = new DataView(buffer);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(u8.slice(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  if (/'fortran_order':\s*True/.test(header)) throw new Error('fortran-ordered .npy is not supported');
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1].split(',').map(s => s.trim()).filter(Boolean).map(Number);
  if (shape.length !== 2 || shape[1] !== 3) throw new Error(`expected an N x 3 array, got shape (${shape})`);
  const off = hstart + hlen;
  let data;
  if (descr === '<f4') data = Float64Array.from(new Float32Array(buffer.slice(off, off + shape[0] * 12)));
  else if (descr === '<f8') data = new Float64Array(buffer.slice(off, off + shape[0] * 24));
  else throw new Error(`unsupported dtype ${descr} (use float32 or float64)`);
  return toPoints(data);
}

function parsePly(buffer) {
  const u8 = new Uint8Array(buffer);
  const headEnd = new TextDecoder().decode(u8.slice(0, Math.min(u8.length, 4096))).indexOf('end_header');
  if (headEnd < 0) throw new Error('invalid .ply header');
  const headerText = new TextDecoder().decode(u8.slice(0, headEnd));
  const bodyStart = headEnd + 'end_header'.length + 1 + (u8[headEnd + 'end_header'.length] === 13 ? 1 : 0);
  const lines = headerText.split(/\r?\n/);
  const format = lines.find(l => l.startsWith('format')).split(/\s+/)[1];
  let count = 0, inVertex = false;
  const props = [];
  for (const l of lines) {
    const t = l.trim().split(/\s+/);
    if (t[0] === 'element') { inVertex = t[1] === 'vertex'; if (inVertex) count = +t[2]; }
    else if (t[0] === 'property' && inVertex) props.push({type: t[1], name: t[t.length - 1]});
  }
  const ix = ['x', 'y', 'z'].map(n => props.findIndex(p => p.name === n));
  if (ix.some(i => i < 0)) throw new Error('.ply has no x/y/z vertex properties');
  const out = new Float64Array(count * 3);
  if (format === 'ascii') {
    const body = new TextDecoder().decode(u8.slice(bodyStart)).split(/\r?\n/);
    for (let i = 0; i < count; i++) {
      const t = body[i].trim().split(/\s+/);
      for (let k = 0; k < 3; k++) out[i * 3 + k] = +t[ix[k]];
    }
  } else if (format === 'binary_little_endian') {
    const size = {char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int16: 2, uint16: 2, int: 4, uint: 4,
      int32: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8};
    const offs = [];
    let stride = 0;
    for (const p of props) { offs.push(stride); stride += size[p.type]; }
    const dv = new DataView(buffer, bodyStart);
    for (let i = 0; i < count; i++) {
      for (let k = 0; k < 3; k++) {
        const p = props[ix[k]], o = i * stride + offs[ix[k]];
        out[i * 3 + k] = p.type.startsWith('double') || p.type === 'float64' ? dv.getFloat64(o, true) : dv.getFloat32(o, true);
      }
    }
  } else throw new Error(`unsupported .ply format ${format}`);
  return toPoints(out);
}

function parseText(text) {
  const vals = [];
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim().split(/[\s,;]+/).map(Number);
    if (t.length >= 3 && t.slice(0, 3).every(Number.isFinite)) vals.push(t[0], t[1], t[2]);
  }
  if (!vals.length) throw new Error('no "x y z" rows found in the file');
  return toPoints(Float64Array.from(vals));
}

function toPoints(flat) {
  const out = [];
  for (let i = 0; i + 2 < flat.length; i += 3) {
    if (Number.isFinite(flat[i]) && Number.isFinite(flat[i + 1]) && Number.isFinite(flat[i + 2])) {
      out.push(flat[i], flat[i + 1], flat[i + 2]);
    }
  }
  if (out.length < 150) throw new Error(`only ${out.length / 3} valid points; need at least 50`);
  return Float64Array.from(out);
}

// ---------- numpy-faithful numerics ----------
// np.add.reduce on a contiguous 1-D float64 array (pairwise summation, identity-initialised).
export function pwSum(a, lo = 0, n = a.length - lo) {
  if (n < 8) {
    let r = 0;
    for (let i = 0; i < n; i++) r += a[lo + i];
    return r;
  }
  if (n <= 128) {
    const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pwSum(a, lo, n2) + pwSum(a, lo + n2, n - n2);
}
const mean1 = a => pwSum(a) / a.length;                       // a.mean() for 1-D
const sortedCopy = a => Float64Array.from(a).sort();

// np.percentile(a, q) (method 'linear') on an ascending-sorted array
export function percentileSorted(s, q) {
  const n = s.length;
  const virt = (n - 1) * (q / 100);
  if (virt >= n - 1) return s[n - 1];
  const prev = Math.floor(virt), gamma = virt - prev;
  const a = s[prev], b = s[prev + 1], diff = b - a;
  return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
}
export const percentile = (values, q) => percentileSorted(sortedCopy(values), q);

function medianSorted(s) {
  const n = s.length, h = n >> 1;
  return n % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}
const median = a => medianSorted(sortedCopy(a));

// np.histogram(a, bins=nbins, range=(lo, hi)) with equal-width bins (fast path of numpy)
function histogram(vals, nbins, lo, hi) {
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const edges = new Float64Array(nbins + 1);
  const step = (hi - lo) / nbins;
  for (let i = 0; i <= nbins; i++) edges[i] = i * step + lo;
  edges[nbins] = hi;
  const norm = nbins / (hi - lo);
  const counts = new Int32Array(nbins);
  for (let n = 0; n < vals.length; n++) {
    const v = vals[n];
    if (!(v >= lo && v <= hi)) continue;
    let k = Math.trunc((v - lo) * norm);
    if (k === nbins) k--;
    if (v < edges[k]) k--;
    if (v >= edges[k + 1] && k !== nbins - 1) k++;
    counts[k]++;
  }
  return {counts, edges};
}

// np.convolve(h, [1/3, 1/3, 1/3], 'same') and np.convolve(h, ones(5), 'same')
function convolve3(h) {
  const w = 1 / 3, n = h.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = ((i > 0 ? h[i - 1] * w : 0) + h[i] * w) + (i + 1 < n ? h[i + 1] * w : 0);
  }
  return out;
}
function convolve5(h) {
  const n = h.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = i - 2; j <= i + 2; j++) if (j >= 0 && j < n) s += h[j];
    out[i] = s;
  }
  return out;
}

function* combinations(n, m, start = 0, prefix = []) {  // itertools.combinations(range(n), m), lexicographic
  if (prefix.length === m) { yield prefix.slice(); return; }
  for (let i = start; i <= n - (m - prefix.length); i++) {
    prefix.push(i);
    yield* combinations(n, m, i + 1, prefix);
    prefix.pop();
  }
}

export function fnv1a(bytes) {  // FNV-1a 32 bit, used to fingerprint the float32 partial exactly
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 0x01000193) >>> 0;
  return h >>> 0;
}

// symmetric 3x3 Jacobi eigen-solver: ascending eigenvalues and the eigenvectors as columns v[row][col]
function eigh3(cov) {
  const a = cov.map(r => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 60; sweep++) {
    let p = 0, q = 1;
    if (Math.abs(a[0][2]) > Math.abs(a[p][q])) { p = 0; q = 2; }
    if (Math.abs(a[1][2]) > Math.abs(a[p][q])) { p = 1; q = 2; }
    if (Math.abs(a[p][q]) < 1e-300) break;
    const th = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const c = Math.cos(th), s = Math.sin(th);
    for (let k = 0; k < 3; k++) {
      const akp = a[k][p], akq = a[k][q];
      a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
    }
    for (let k = 0; k < 3; k++) {
      const apk = a[p][k], aqk = a[q][k];
      a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
    }
    for (let k = 0; k < 3; k++) {
      const vkp = v[k][p], vkq = v[k][q];
      v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  return {w: order.map(i => a[i][i]), v: [0, 1, 2].map(r => order.map(i => v[r][i]))};
}

// ---------- sanitize: finite rows, exact duplicates removed, lexicographic order (np.unique(axis=0)) ----------
function sanitize(flat) {
  const n = flat.length / 3;
  const rows = [];
  for (let i = 0; i < n; i++) {
    const x = flat[3 * i], y = flat[3 * i + 1], z = flat[3 * i + 2];
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) rows.push(i);
  }
  rows.sort((a, b) => {
    for (let k = 0; k < 3; k++) {
      const u = flat[3 * a + k], w = flat[3 * b + k];
      if (u < w) return -1;
      if (u > w) return 1;
    }
    return 0;
  });
  const keep = [];
  for (let r = 0; r < rows.length; r++) {
    if (r > 0) {
      const a = rows[r - 1], b = rows[r];
      if (flat[3 * a] === flat[3 * b] && flat[3 * a + 1] === flat[3 * b + 1] && flat[3 * a + 2] === flat[3 * b + 2]) continue;
    }
    keep.push(rows[r]);
  }
  const p = new Float64Array(keep.length * 3);
  keep.forEach((i, r) => { p[3 * r] = flat[3 * i]; p[3 * r + 1] = flat[3 * i + 1]; p[3 * r + 2] = flat[3 * i + 2]; });
  return p;
}

// ---------- stripe ruler (analysis/forensics.py: estimate_from_stripes) ----------
function wallBand(p, backgroundQuantile) {
  const m = p.length / 3;
  const z = new Float64Array(m);
  for (let i = 0; i < m; i++) z[i] = p[3 * i + 2];
  const zq = percentileSorted(sortedCopy(z), backgroundQuantile);
  const base = [];                                   // indices of far points (z >= quantile)
  for (let i = 0; i < m; i++) if (z[i] >= zq) base.push(i);
  const bz = Float64Array.from(base, i => z[i]);
  if (base.length < 50) return {idx: base, z0: median(bz)};
  let lo = Infinity, hi = -Infinity;
  for (const v of bz) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const {counts, edges} = histogram(bz, 80, lo, hi);
  const hs = convolve3(counts);
  let k = 0;
  for (let i = 1; i < hs.length; i++) if (hs[i] > hs[k]) k = i;
  const centre = (edges[k] + edges[k + 1]) / 2, win = 1.5 * (edges[1] - edges[0]);
  const z0 = median(Float64Array.from(bz.filter(v => Math.abs(v - centre) < win)));
  const band = base.filter(i => Math.abs(z[i] - z0) < 0.05);
  return {idx: band.length >= 200 ? band : base, z0};
}

function stripeCentres(xs, minCountFrac = 0.15, merge = 0.06, bins = 280) {
  const {counts: h, edges: e} = histogram(xs, bins, -1, 1);
  const c = Float64Array.from({length: bins}, (_, i) => (e[i + 1] + e[i]) / 2);
  const mass = convolve5(h);
  let hmax = 0;
  for (const v of h) if (v > hmax) hmax = v;
  const groups = [];
  for (let i = 2; i < bins - 2; i++) {
    const win = Math.max(h[i - 2], h[i - 1], h[i], h[i + 1], h[i + 2]);
    if (h[i] === win && h[i] > minCountFrac * hmax) {
      const last = groups.length ? groups[groups.length - 1] : null;
      if (last && c[i] - c[last[last.length - 1]] < merge) last.push(i);
      else groups.push([i]);
    }
  }
  const cen = [], ms = [];
  for (const g of groups) {
    const w = Float64Array.from(g, i => Math.max(mass[i], 1e-9));
    cen.push(pwSum(Float64Array.from(g, (i, j) => c[i] * w[j])) / pwSum(w));
    ms.push(pwSum(Float64Array.from(g, i => mass[i])));
  }
  return {cen, mass: ms};
}

function fitColumns(cen, mass, maxExtra = 2) {
  const n = cen.length;
  const total = pwSum(Float64Array.from(mass));
  const cands = [];
  for (let m = Math.min(n, 8); m > Math.max(3, n - maxExtra - 1); m--) {
    for (const pidx of combinations(n, m)) {
      const x = Float64Array.from(pidx, i => cen[i]);
      const xm = pwSum(x) / m;
      const xd = x.map(v => v - xm);
      for (const cidx of combinations(8, m)) {
        const t = Float64Array.from(cidx, i => TAN_STRIPE[i]);
        const tm = pwSum(t) / m;
        const td = t.map(v => v - tm);
        const D = pwSum(td.map((v, i) => v * xd[i])) / pwSum(td.map(v => v * v));
        if (!(0.25 < D && D < 4.0)) continue;
        const ox = xm - D * tm;
        const res = Float64Array.from(x, (v, i) => { const r = v - ox - D * t[i]; return r * r; });
        const rms = Math.sqrt(pwSum(res) / m);
        const dropped = 1.0 - pwSum(Float64Array.from(pidx, i => mass[i])) / total;
        cands.push({cost: rms + 0.05 * dropped * 0 + 0.02 * dropped, rms, ox, D, columns: cidx, peaks: pidx});
      }
    }
  }
  if (!cands.length) return {fit: null, alt: 0};
  cands.sort((a, b) => a.cost - b.cost);               // stable, like list.sort(key=...)
  const best = cands[0];
  let alt = 0;
  for (let i = 1; i < cands.length; i++) {
    const c = cands[i];
    if (c.cost < 1.5 * best.cost + 0.002 && Math.abs(c.D - best.D) > 0.05 * best.D) alt++;
  }
  return {fit: best, alt};
}

export function estimateFromStripes(p) {
  const m = p.length / 3;
  const {idx, z0: zw} = wallBand(p, 70);
  if (idx.length < 150) return null;
  const wx = Float64Array.from(idx, i => p[3 * i]), wy = Float64Array.from(idx, i => p[3 * i + 1]);
  let bestK = 0, bestPeak = -1;
  for (let k = 0; k < ROLL_COS.length; k++) {         // stripe tilt (hand roll), -20 .. +20 deg in 0.5 deg steps
    const c = ROLL_COS[k], s = ROLL_SIN[k];
    const {counts} = histogram(wx.map((x, i) => c * x + s * wy[i]), 280, -1, 1);
    let pk = 0;
    for (const v of counts) pk += v * v;
    if (pk > bestPeak) { bestK = k; bestPeak = pk; }
  }
  const c = ROLL_COS[bestK], s = ROLL_SIN[bestK];
  const xs = wx.map((x, i) => c * x + s * wy[i]);
  let {cen, mass} = stripeCentres(xs);
  if (cen.length < 4) return null;
  if (cen.length > 8) {                               // keep the 8 heaviest, still ordered (stable tie-break)
    const keep = cen.map((_, i) => i).sort((a, b) => mass[b] - mass[a]).slice(0, 8).sort((a, b) => a - b);
    cen = keep.map(i => cen[i]);
    mass = keep.map(i => mass[i]);
  }
  const {fit, alt} = fitColumns(cen, mass);
  if (!fit) return null;
  const D = fit.D;
  const ys = sortedCopy(wy.map((y, i) => -s * wx[i] + c * y));
  const yRot = (percentileSorted(ys, 1) + percentileSorted(ys, 99)) / 2;
  const ox = c * fit.ox - s * yRot, oy = s * fit.ox + c * yRot;
  const relRms = fit.rms / D, nStripes = fit.columns.length;
  return {ox, oy, oz: zw - D, wall_depth: D, wall_z: zw, roll_deg: -20 + 0.5 * bestK, n_stripes: nStripes,
    columns: fit.columns, rms: fit.rms, rel_rms: relRms, n_found_peaks: cen.length, n_alternatives: alt,
    wall_points: idx.length, n_unique: m,
    confident: relRms <= CONFIDENT_REL_RMS && nStripes >= CONFIDENT_MIN_STRIPES && alt === 0};
}

function locateSensor(p) {
  const est = estimateFromStripes(p);
  if (!est) return null;
  const m = p.length / 3;
  const z = new Float64Array(m);
  for (let i = 0; i < m; i++) z[i] = p[3 * i + 2];
  const good = est.n_stripes >= STRIPE_MIN_COUNT && est.rel_rms <= STRIPE_REL_RMS_MAX && est.wall_depth > 0.2
    && est.oz < percentileSorted(sortedCopy(z), 5) + 0.05;
  if (!good) return null;
  return {source: 'stripes', ox: est.ox, oy: est.oy, oz: est.oz, wall_depth: est.wall_depth, wall_z: est.wall_z,
    quality: 1.0 - est.rel_rms / STRIPE_REL_RMS_MAX, est};
}

// ---------- object segmentation (majproj/preprocess/core.py) ----------
function knnDistances(P, k1) {                       // k1 smallest distances per point INCLUDING the point itself
  const n = P.length / 3, out = new Float64Array(n * k1), buf = new Float64Array(k1);
  for (let i = 0; i < n; i++) {
    buf.fill(Infinity);
    const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    for (let j = 0; j < n; j++) {
      const dx = x - P[3 * j], dy = y - P[3 * j + 1], dz = z - P[3 * j + 2];
      const d2 = (dx * dx + dy * dy) + dz * dz;
      if (d2 < buf[k1 - 1]) {
        let t = k1 - 1;
        while (t > 0 && buf[t - 1] > d2) { buf[t] = buf[t - 1]; t--; }
        buf[t] = d2;
      }
    }
    for (let t = 0; t < k1; t++) out[i * k1 + t] = Math.sqrt(buf[t]);
  }
  return out;
}

function knnEps(P, k = 5, mult = 3.0, lo = 0.012, hi = 0.06) {
  const n = P.length / 3;
  if (n <= k + 1) return hi;
  const dd = knnDistances(P, k + 1);
  const last = Float64Array.from({length: n}, (_, i) => dd[i * (k + 1) + k]);
  return Math.min(Math.max(mult * median(last), lo), hi);
}

function removeOutliers(P, k = 8, nsigma = 2.5) {    // boolean mask of points to keep
  const n = P.length / 3;
  if (n <= k + 2) return new Uint8Array(n).fill(1);
  const dd = knnDistances(P, k + 1);
  const m = new Float64Array(n);
  for (let i = 0; i < n; i++) {                       // dd[:, 1:].mean(1) over 8 values: numpy's 8-way tree sum
    const r = j => dd[i * (k + 1) + 1 + j];
    m[i] = (((r(0) + r(1)) + (r(2) + r(3))) + ((r(4) + r(5)) + (r(6) + r(7)))) / 8;
  }
  const mu = mean1(m);
  const dev = m.map(v => { const d = v - mu; return d * d; });
  const sd = Math.sqrt(pwSum(dev) / n);
  const thr = mu + nsigma * sd;
  return Uint8Array.from(m, v => (v <= thr ? 1 : 0));
}

function dbscan(Q, eps, minSamples) {               // sklearn DBSCAN semantics: d <= eps, self counted
  const n = Q.length / 3;
  const cell = eps * (1 + 1e-9);
  const cx = new Int32Array(n * 3), grid = new Map();
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) cx[3 * i + a] = Math.floor(Q[3 * i + a] / cell);
    const key = `${cx[3 * i]},${cx[3 * i + 1]},${cx[3 * i + 2]}`;
    const list = grid.get(key);
    if (list) list.push(i); else grid.set(key, [i]);
  }
  const eps2 = eps * eps;
  const nb = new Array(n);
  for (let i = 0; i < n; i++) {
    const out = [], x = Q[3 * i], y = Q[3 * i + 1], z = Q[3 * i + 2];
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const list = grid.get(`${cx[3 * i] + dx},${cx[3 * i + 1] + dy},${cx[3 * i + 2] + dz}`);
      if (!list) continue;
      for (const j of list) {
        const a = x - Q[3 * j], b = y - Q[3 * j + 1], c = z - Q[3 * j + 2];
        if ((a * a + b * b) + c * c <= eps2) out.push(j);
      }
    }
    nb[i] = out;
  }
  const labels = new Int32Array(n).fill(-1);
  let label = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] !== -1 || nb[i].length < minSamples) continue;
    labels[i] = label;
    const stack = [i];
    while (stack.length) {
      const j = stack.pop();
      if (nb[j].length < minSamples) continue;       // border point: labelled, does not expand
      for (const v of nb[j]) if (labels[v] === -1) { labels[v] = label; stack.push(v); }
    }
    label++;
  }
  return labels;
}

function isRecedingSheet(Q, wallDepth) {            // Q: flat points of one cluster
  const n = Q.length / 3;
  const z = sortedCopy(Float64Array.from({length: n}, (_, i) => Q[3 * i + 2]));
  const zr = percentileSorted(z, 98) - percentileSorted(z, 2);
  if (wallDepth != null && zr > RECEDING_FRACTION * wallDepth) return {sheet: true, why: 'receding'};
  const mu = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) mu[a] += Q[3 * i + a];
  for (let a = 0; a < 3; a++) mu[a] /= n;
  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    const d = [Q[3 * i] - mu[0], Q[3 * i + 1] - mu[1], Q[3 * i + 2] - mu[2]];
    for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) cov[r][s] += d[r] * d[s];
  }
  for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) cov[r][s] /= n;
  const {w, v} = eigh3(cov);
  const flat = w[0] < 0.02 * w[2] && Math.abs(v[1][0]) > 0.9;
  return {sheet: flat, why: flat ? 'horizontal' : '', ratio: w[0] / (0.02 * w[2]), ny: Math.abs(v[1][0])};
}

function segmentObject(p, sensor) {                 // indices into p of the target object (ascending)
  const m = p.length / 3;
  const depth = new Float64Array(m);
  let fg = [];
  if (sensor) {
    for (let i = 0; i < m; i++) depth[i] = p[3 * i + 2] - sensor.oz;
    for (let i = 0; i < m; i++) if (depth[i] < WALL_FRACTION * sensor.wall_depth) fg.push(i);
  } else {
    let zmin = Infinity;
    for (let i = 0; i < m; i++) if (p[3 * i + 2] < zmin) zmin = p[3 * i + 2];
    for (let i = 0; i < m; i++) depth[i] = p[3 * i + 2] - zmin;
    fg = Array.from({length: m}, (_, i) => i);
  }
  if (fg.length < MIN_OBJECT_POINTS) return {idx: [], info: {fg: fg.length}};
  const Q = new Float64Array(fg.length * 3);
  fg.forEach((i, r) => { Q[3 * r] = p[3 * i]; Q[3 * r + 1] = p[3 * i + 1]; Q[3 * r + 2] = p[3 * i + 2]; });
  const eps = knnEps(Q);
  const labels = dbscan(Q, eps, 8);
  const ids = [...new Set(labels.filter(l => l >= 0))].sort((a, b) => a - b);
  const info = {fg: fg.length, eps, clusters: []};
  if (!ids.length) return {idx: [], info};
  const members = new Map(ids.map(k => [k, []]));
  labels.forEach((l, r) => { if (l >= 0) members.get(l).push(r); });
  const maxCount = Math.max(...ids.map(k => members.get(k).length));
  const cands = [];
  for (const k of ids) {
    const rows = members.get(k);
    if (rows.length >= Math.max(MIN_OBJECT_POINTS, 0.2 * maxCount)) {
      const med = median(Float64Array.from(rows, r => depth[fg[r]]));
      const flat = new Float64Array(rows.length * 3);
      rows.forEach((r, j) => { flat[3 * j] = Q[3 * r]; flat[3 * j + 1] = Q[3 * r + 1]; flat[3 * j + 2] = Q[3 * r + 2]; });
      const sheet = isRecedingSheet(flat, sensor ? sensor.wall_depth : null);
      cands.push({med, k, sheet: sheet.sheet});
      info.clusters.push({k, n: rows.length, med, sheet: sheet.sheet, why: sheet.why});
    }
  }
  if (!cands.length) return {idx: [], info};
  const good = cands.filter(c => !c.sheet);
  const pool = good.length ? good : cands;
  pool.sort((a, b) => a.med - b.med || a.k - b.k);
  const chosen = pool[0].k;
  info.chosen = chosen;
  const rows = members.get(chosen), cluster = rows.map(r => fg[r]);
  const P = new Float64Array(cluster.length * 3);
  cluster.forEach((i, r) => { P[3 * r] = p[3 * i]; P[3 * r + 1] = p[3 * i + 1]; P[3 * r + 2] = p[3 * i + 2]; });
  const keep = removeOutliers(P);
  return {idx: cluster.filter((_, r) => keep[r]), info};
}

// ---------- canonical frame ----------
export function toCanonical(points, transform) {      // canonical = (p - center) / scale, float32
  const n = points.length / 3, out = new Float32Array(points.length), c = transform.center, s = transform.scale;
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) out[3 * i + a] = (points[3 * i + a] - c[a]) / s;
  return out;
}

export function fromCanonical(points, transform) {    // exact inverse (float64 internally), float32 result
  const n = points.length / 3, out = new Float32Array(points.length), c = transform.center, s = transform.scale;
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) out[3 * i + a] = points[3 * i + a] * s + c[a];
  return out;
}

function farthestPointSample(P, n) {                  // deterministic FPS; cyclic tiling when fewer points exist
  const m = P.length / 3;
  if (m === 0) throw new Error('no points to sample');
  const k = Math.min(n, m);
  const mu = [0, 0, 0];
  for (let i = 0; i < m; i++) for (let a = 0; a < 3; a++) mu[a] += P[3 * i + a];
  for (let a = 0; a < 3; a++) mu[a] /= m;
  let start = 0, best = Infinity;
  for (let i = 0; i < m; i++) {
    const dx = P[3 * i] - mu[0], dy = P[3 * i + 1] - mu[1], dz = P[3 * i + 2] - mu[2];
    const d = Math.sqrt((dx * dx + dy * dy) + dz * dz);
    if (d < best) { best = d; start = i; }
  }
  const idx = new Int32Array(k), d = new Float64Array(m);
  idx[0] = start;
  const upd = (j, first) => {
    for (let i = 0; i < m; i++) {
      const dx = P[3 * i] - P[3 * j], dy = P[3 * i + 1] - P[3 * j + 1], dz = P[3 * i + 2] - P[3 * j + 2];
      const v = Math.sqrt((dx * dx + dy * dy) + dz * dz);
      d[i] = first || v < d[i] ? v : d[i];
    }
  };
  upd(start, true);
  for (let t = 1; t < k; t++) {
    let j = 0;
    for (let i = 1; i < m; i++) if (d[i] > d[j]) j = i;
    idx[t] = j;
    upd(j, false);
  }
  return Int32Array.from({length: n}, (_, i) => idx[i % k]);
}

function angularSizeDeg(P, idx, o) {
  const toDeg = 180 / Math.PI;
  const az = new Float64Array(idx.length), el = new Float64Array(idx.length);
  idx.forEach((i, r) => {
    const x = P[3 * i] - o[0], y = P[3 * i + 1] - o[1], z = Math.max(P[3 * i + 2] - o[2], 1e-6);
    az[r] = Math.atan2(x, z) * toDeg;
    el[r] = Math.atan2(y, z) * toDeg;
  });
  const a = sortedCopy(az), e = sortedCopy(el);
  return Math.max(percentileSorted(a, 99) - percentileSorted(a, 1), percentileSorted(e, 99) - percentileSorted(e, 1));
}

// One exported file -> {partial (Float32Array 2048*3, canonical), transform {center, scale}, meta, unique, objectIdx}
export function preprocess(points, nOut = N_IN) {
  const flat = points instanceof Float64Array ? points : Float64Array.from(Array.isArray(points[0]) ? points.flat() : points);
  const p = sanitize(flat);
  const m = p.length / 3;
  if (m < 3) throw new Error('scan has fewer than 3 finite points');
  let sensor = locateSensor(p);
  let seg = segmentObject(p, sensor);
  let mode = sensor ? sensor.source : 'scale_free';
  if (seg.idx.length < MIN_OBJECT_POINTS && sensor) {  // stripe fit found, but no clean foreground object
    sensor = null; mode = 'scale_free';
    seg = segmentObject(p, null);
  }
  let idx = seg.idx;
  if (idx.length < MIN_OBJECT_POINTS) {                // nothing compact found: whole cloud
    idx = Array.from({length: m}, (_, i) => i);
    mode = 'scale_free'; sensor = null;
  }
  const n = idx.length, obj = new Float64Array(n * 3);
  idx.forEach((i, r) => { obj[3 * r] = p[3 * i]; obj[3 * r + 1] = p[3 * i + 1]; obj[3 * r + 2] = p[3 * i + 2]; });
  const centre = [0, 0, 0];
  for (let r = 0; r < n; r++) for (let a = 0; a < 3; a++) centre[a] += obj[3 * r + a];
  for (let a = 0; a < 3; a++) centre[a] /= n;
  let scale, clamped = false, depthRatio = NaN, ang = NaN, depthZ = null, radial = null, sensorMeta = null;
  if (sensor) {
    const o = [sensor.ox, sensor.oy, sensor.oz];
    depthZ = centre[2] - sensor.oz;
    const dist = new Float64Array(n);
    for (let r = 0; r < n; r++) {
      const dx = obj[3 * r] - o[0], dy = obj[3 * r + 1] - centre[1], dz = obj[3 * r + 2] - o[2];
      dist[r] = Math.sqrt((dx * dx + dy * dy) + dz * dz);
    }
    radial = median(dist);
    scale = radial;
    if (scale < MIN_SCALE) { scale = MIN_SCALE; clamped = true; }
    depthRatio = scale / sensor.wall_depth;
    ang = angularSizeDeg(p, idx, o);
    sensorMeta = {source: sensor.source, o, wall_depth: sensor.wall_depth, wall_z: sensor.wall_z,
      quality: sensor.quality, rel_rms: sensor.est.rel_rms, n_stripes: sensor.est.n_stripes,
      roll_deg: sensor.est.roll_deg, stripe_confident: sensor.est.confident};
  } else {
    let mx = 0;
    for (let r = 0; r < n; r++) for (let a = 0; a < 3; a++) mx = Math.max(mx, Math.abs(obj[3 * r + a] - centre[a]));
    scale = Math.max(mx, 1e-3);
  }
  const transform = {center: centre, scale};
  const canon = toCanonical(obj, transform);
  const sel = farthestPointSample(Float64Array.from(canon), nOut);
  const partial = new Float32Array(nOut * 3);
  for (let i = 0; i < nOut; i++) for (let a = 0; a < 3; a++) partial[3 * i + a] = canon[3 * sel[i] + a];
  const meta = {sensor_estimate: sensorMeta, object_depth_ratio: depthRatio, angular_size_deg: ang,
    n_object_points: n, n_input_unique: m, mode, scale_clamped: clamped, depth_z: depthZ, depth_radial: radial,
    confident: !!(sensor && !clamped && n >= 100 && sensor.quality > 0.0 && depthRatio > 0.0 && depthRatio < WALL_FRACTION),
    segmentation: seg.info};
  return {partial, transform, meta, unique: p, objectIdx: idx};
}

export const partialChecksum = partial => fnv1a(new Uint8Array(partial.buffer, partial.byteOffset, partial.byteLength));

// ---------- model ----------
export async function loadModel(baseUrl, onProgress) {
  const manifest = await (await fetch(`${baseUrl}manifest.json`)).json();
  const buf = new Uint8Array(manifest.size);
  const cacheName = `adapointr-${manifest.version || 'unversioned'}`;
  let done = 0;
  for (const chunk of manifest.chunks) {
    const bytes = await fetchWithCache(cacheName, `${baseUrl}${chunk}`, n => onProgress?.((done + n) / manifest.size));
    buf.set(bytes, done);
    done += bytes.length;
    onProgress?.(done / manifest.size);
  }
  const session = await ort.InferenceSession.create(buf, {executionProviders: ['wasm'], graphOptimizationLevel: 'basic'});
  return {session, manifest};
}

async function fetchWithCache(cacheName, url, onBytes) {
  let cache = null;
  const native = window.Capacitor?.isNativePlatform?.();  // the Android app bundles the model: no cache copy
  try {
    if (!native) {
      cache = await caches.open(cacheName);
      for (const k of await caches.keys()) if (k.startsWith('adapointr-') && k !== cacheName) caches.delete(k);  // stale models
    }
  } catch (e) { /* Cache API unavailable */ }
  const hit = cache && await cache.match(url).catch(() => null);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not download ${url} (${res.status})`);
  const reader = res.body.getReader();
  const parts = [];
  let n = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    parts.push(value); n += value.length; onBytes?.(n);
  }
  const bytes = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  try { await cache?.put(url, new Response(bytes)); } catch (e) { /* quota: fine, just not cached */ }
  return bytes;
}

const triples = (flat, f = v => v) => {
  const out = new Array(flat.length / 3);
  for (let i = 0; i < out.length; i++) out[i] = [f(flat[3 * i]), f(flat[3 * i + 1]), f(flat[3 * i + 2])];
  return out;
};

// scan (flat Float64Array from parsePointFile) -> preprocess -> AdaPoinTr -> inverse transform
export async function complete(session, pts, ortLib = globalThis.ort) {
  const pre = preprocess(pts);
  const checksum = partialChecksum(pre.partial);
  // the proxied (web worker) runtime transfers the input buffer and detaches it, so hand it a copy
  const out = await session.run({partial: new ortLib.Tensor('float32', pre.partial.slice(), [1, N_IN, 3])});
  const canonical = Float32Array.from(out.complete.data);         // 16384 x 3 in the canonical frame
  const completed = fromCanonical(canonical, pre.transform);
  const isObj = new Uint8Array(pre.unique.length / 3);
  for (const i of pre.objectIdx) isObj[i] = 1;
  const observed = [], background = [];
  for (let i = 0; i < isObj.length; i++) {
    (isObj[i] ? observed : background).push([pre.unique[3 * i], pre.unique[3 * i + 1], pre.unique[3 * i + 2]]);
  }
  return {observed, background, completed: triples(completed), canonical, partial: pre.partial,
    transform: pre.transform, center: pre.transform.center, scale: pre.transform.scale, meta: pre.meta,
    checksum};
}

// ---------- self-test helpers (used by index.html?selftest and the Node parity harness) ----------
export const SELFTEST_TOL = {pre: 1e-9, cd: 2e-3};   // preprocessing numbers; model one-sided Chamfer-L1 (canonical units)

export function oneSidedChamfer(a, flat) {            // mean over triples in a of the distance to the nearest point of flat
  let acc = 0;
  const n = flat.length / 3;
  for (const p of a) {
    let best = Infinity;
    for (let j = 0; j < n; j++) {
      const dx = p[0] - flat[3 * j], dy = p[1] - flat[3 * j + 1], dz = p[2] - flat[3 * j + 2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < best) best = d;
    }
    acc += Math.sqrt(best);
  }
  return acc / a.length;
}

const relDiff = (a, b) => (a == null || Number.isNaN(a)) ? ((b == null || Number.isNaN(b)) ? 0 : Infinity)
  : Math.abs(a - b) / Math.max(1e-12, Math.abs(b));

// res = complete(...) result, e = expected.files[name] written by scripts/build_web.py from majproj.preprocess + PyTorch
export function compareWithExpected(res, e, tol = SELFTEST_TOL) {
  const m = res.meta, c = res.transform.center;
  const dCentre = Math.max(...[0, 1, 2].map(k => Math.abs(c[k] - e.center[k])));
  const dScale = relDiff(res.scale, e.scale);
  const dRatio = relDiff(m.object_depth_ratio, e.depth_ratio), dAng = relDiff(m.angular_size_deg, e.angular_size_deg);
  let dHead = 0;
  e.partial_head.forEach((p, i) => { for (let k = 0; k < 3; k++) dHead = Math.max(dHead, Math.abs(res.partial[3 * 16 * i + k] - p[k])); });
  const exact = res.checksum === e.partial_fnv1a;
  const cd = oneSidedChamfer(e.completed_canonical, res.canonical);
  const pre = m.mode === e.mode && m.confident === e.confident && m.n_object_points === e.n_object_points
    && m.n_input_unique === e.n_input_unique && dCentre <= tol.pre && dScale <= tol.pre && dRatio <= tol.pre
    && dAng <= tol.pre && exact && dHead <= 1e-6;
  return {pass: pre && cd < tol.cd, pre, exact, cd, dCentre, dScale, dRatio, dAng, dHead, mode: m.mode,
    nObj: m.n_object_points, line: `mode ${m.mode}/${e.mode}, object points ${m.n_object_points}/${e.n_object_points}, `
      + `partial bit-exact ${exact}, |d centre| ${dCentre.toExponential(1)}, rel d scale ${dScale.toExponential(1)}, `
      + `ratio ${Number.isFinite(m.object_depth_ratio) ? m.object_depth_ratio.toFixed(4) : 'n/a'}, model CD ${cd.toExponential(2)}`};
}
