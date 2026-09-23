// New-run observations only. Multiple offsets in one market are correlated.
import fs from 'node:fs';
import path from 'node:path';
import { config } from '../config.js';
import { mulberry32 } from './core/mlp.js';

const dir = config.live.resultDir;
function lines(name) {
  const file = path.join(dir, name);
  if (!fs.existsSync(file)) return [];
  // Ignore only the in-progress final append; complete malformed records fail loudly.
  const text = fs.readFileSync(file, 'utf8');
  return text.slice(0, text.lastIndexOf('\n') + 1).split('\n').filter(Boolean).map(JSON.parse);
}
const modelId = fs.existsSync(path.join(dir, 'live-state.json'))
  ? JSON.parse(fs.readFileSync(path.join(dir, 'live-state.json'), 'utf8')).modelId : null;
const outcomes = new Map(lines('resolutions.jsonl').filter((r) => ['Up', 'Down'].includes(r.outcome))
  .map((r) => [r.slug, Number(r.outcome === 'Up')]));
const unique = new Map();
for (const o of lines('events.jsonl')) {
  if (o.type === 'observation') unique.set(o.key, o);
}
const rows = [...unique.values()].filter((r) => outcomes.has(r.slug))
  .map((r) => ({ ...r, y: outcomes.get(r.slug) }));
console.log(`Current model ${modelId ?? 'none'} · all ${new Set([...unique.values()].map((r) => r.modelId)).size} deployed versions, scored as originally predicted`);
console.log(`${unique.size} observations · ${rows.length} scored · ${new Set(rows.map((r) => r.slug)).size} resolved markets`);
if (!rows.length) { console.log('Waiting for new-run markets to resolve.'); process.exit(0); }
function metrics(rows, field) {
  let brier = 0, logloss = 0, correct = 0;
  for (const r of rows) {
    const p = Math.max(1e-9, Math.min(1 - 1e-9, r[field]));
    brier += (p - r.y) ** 2;
    logloss -= r.y * Math.log(p) + (1 - r.y) * Math.log(1 - p);
    correct += Number((p >= 0.5 ? 1 : 0) === r.y);
  }
  return { brier: brier / rows.length, logloss: logloss / rows.length, acc: correct / rows.length };
}
console.log('offset    n    model Brier  market Brier  model acc  market acc');
for (const off of ['all', ...config.live.offsets]) {
  const group = off === 'all' ? rows : rows.filter((r) => r.offset === off);
  if (!group.length) continue;
  const m = metrics(group, 'pUp'), q = metrics(group, 'qUp');
  console.log(`${String(off).padEnd(6)} ${String(group.length).padStart(4)}      ${m.brier.toFixed(5)}       ${q.brier.toFixed(5)}      ${(m.acc * 100).toFixed(1)}%      ${(q.acc * 100).toFixed(1)}%`);
}
// First average offsets within each market, then resample whole UTC days.
const markets = new Map();
for (const r of rows) {
  if (!markets.has(r.slug)) markets.set(r.slug, { day: new Date(r.at).toISOString().slice(0, 10), d: [] });
  markets.get(r.slug).d.push((r.qUp - r.y) ** 2 - (r.pUp - r.y) ** 2);
}
const days = new Map();
for (const m of markets.values()) {
  if (!days.has(m.day)) days.set(m.day, []);
  days.get(m.day).push(m.d.reduce((s, x) => s + x, 0) / m.d.length);
}
if (days.size < 7 || markets.size < 100) {
  console.log(`Descriptive only: ${days.size} UTC day(s). Wait for at least 7 days and 100 markets before a block-bootstrap interval.`);
} else {
  const blocks = [...days.values()].map((d) => ({ sum: d.reduce((s, x) => s + x, 0), n: d.length }));
  const rng = mulberry32(1337), estimates = [];
  for (let b = 0; b < 2000; b++) {
    let sum = 0, n = 0;
    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[Math.floor(rng() * blocks.length)]; sum += block.sum; n += block.n;
    }
    estimates.push(sum / n);
  }
  estimates.sort((a, b) => a - b);
  console.log(`Daily block-bootstrap 95% interval for market Brier minus model Brier: [${estimates[50].toFixed(5)}, ${estimates[1949].toFixed(5)}]`);
}
console.log('Positive Brier differences favor the model; forecasting scores alone do not establish after-fee trading profitability.');
console.log('Outcomes mix collector-derived Chainlink labels and confirmed public-API recovery; paper returns are not live execution results.');
