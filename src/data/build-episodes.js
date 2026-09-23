// Build training episodes with a decision point INSIDE each 5-min window.
//
// Each 5-min market becomes several episodes — one per decision offset — because
// live we poll at whatever moment we happen to poll. Training across offsets with
// elapsed time as a feature lets the model handle any of them.
//
//   npm run episodes
import fs from 'node:fs';
import path from 'node:path';
import { FEATURES, DIM, WARM, FEATURE_VERSION, extract } from '../core/features.js';

const OUT = 'data';
const OFFSETS = [60, 120, 180, 240];         // seconds into the window at decision time

function loadCsv(dir) {
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.csv')).sort();
  if (!files.length) { console.error(`No CSVs in ${dir}/`); process.exit(1); }
  const rows = [];
  for (const f of files) {
    for (const line of fs.readFileSync(path.join(dir, f), 'utf8').split('\n')) {
      if (!line || line[0] === 'o') continue;
      const c = line.split(',');
      const t = Number(c[0]);
      if (!Number.isFinite(t)) continue;
      rows.push({ t: t > 1e15 ? t / 1000 : t, o: +c[1], h: +c[2], l: +c[3], c: +c[4], v: +c[5] });
    }
  }
  rows.sort((a, b) => a.t - b.t);
  const out = [rows[0]];
  for (let i = 1; i < rows.length; i++) if (rows[i].t !== out[out.length - 1].t) out.push(rows[i]);
  return out;
}

const k5 = loadCsv('data/klines');
console.log(`  5m candles: ${k5.length.toLocaleString()}`);
const k1 = loadCsv('data/klines1m');
console.log(`  1m candles: ${k1.length.toLocaleString()}`);

// index 1m candles by open time for O(1) intra-window lookup
const by1m = new Map();
for (const c of k1) by1m.set(c.t, c);

const maxEp = (k5.length - WARM) * OFFSETS.length;
const X = new Float32Array(maxEp * DIM);
const Y = new Uint8Array(maxEp);
const T = new Float64Array(maxEp);
let n = 0, skippedGap = 0;

for (let i = WARM; i < k5.length; i++) {
  const w = k5[i];
  if (!(w.o > 0) || !(w.c > 0)) continue;
  // All 97 prior bars must be contiguous, otherwise ret/vol horizons change.
  if (w.t - k5[i - WARM].t !== WARM * 300_000) { skippedGap += OFFSETS.length; continue; }
  const label = w.c >= w.o ? 1 : 0;

  for (const off of OFFSETS) {
    // price at t0+off = close of the 1m candle covering [t0+off-60, t0+off)
    const c1 = by1m.get(w.t + (off - 60) * 1000);
    if (!c1 || !(c1.c > 0)) { skippedGap++; continue; }
    const row = extract(k5, i, w.o, c1.c, off);
    let ok = true;
    for (const v of row) if (!Number.isFinite(v)) { ok = false; break; }
    if (!ok) { skippedGap++; continue; }
    X.set(row, n * DIM); Y[n] = label; T[n] = w.t; n++;
  }
}

console.log(`  episodes:   ${n.toLocaleString()}  dim=${DIM}  (${skippedGap.toLocaleString()} skipped: 1m gaps)`);

fs.writeFileSync(path.join(OUT, 'X.bin'), Buffer.from(X.buffer, 0, n * DIM * 4));
fs.writeFileSync(path.join(OUT, 'Y.bin'), Buffer.from(Y.buffer, 0, n));
fs.writeFileSync(path.join(OUT, 'T.bin'), Buffer.from(T.buffer, 0, n * 8));
let ups = 0; for (let i = 0; i < n; i++) ups += Y[i];
fs.writeFileSync(path.join(OUT, 'meta.json'), JSON.stringify({
  featureVersion: FEATURE_VERSION, dim: DIM, features: FEATURES, n, offsets: OFFSETS,
  from: new Date(T[0]).toISOString(), to: new Date(T[n - 1]).toISOString(), baseRate: ups / n,
}, null, 2));

console.log(`  span:       ${new Date(T[0]).toISOString().slice(0, 10)} -> ${new Date(T[n - 1]).toISOString().slice(0, 10)}`);
console.log(`  base rate (Up): ${(ups / n * 100).toFixed(2)}%`);
