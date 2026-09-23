// Warm-start from recorded, actually simulated fills only. This is reward-model
// regression, NOT a backtest of unrecorded actions or a claim of executable P/L.
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { config } from '../config.js';
import { atomicJSON } from './core/live-data.js';
import { decisionFeatures, initialPolicy, actionValues, replayUpdate, policyId } from './core/bandit.js';
import { marketEnd } from './core/resolution.js';

const dir = config.lab.resultDir;
if (fs.existsSync(path.join(dir, 'checkpoint.json')) || fs.existsSync(path.join(dir, 'events.jsonl'))) {
  throw new Error('An experiment already exists; never overwrite its seed or reset its paper account.');
}
function lines(file) {
  if (!fs.existsSync(file)) return [];
  const text = fs.readFileSync(file, 'utf8');
  return text.slice(0, text.lastIndexOf('\n') + 1).split('\n').filter(Boolean).map(JSON.parse);
}
const sources = [config.live.bootstrapDir, config.live.resultDir];
const resolutions = new Map(sources.flatMap((p) => lines(path.join(p, 'resolutions.jsonl'))).map((r) => [r.slug, r]));
const grouped = new Map();
let rejected = 0;
for (const source of sources) {
  const events = lines(path.join(source, 'events.jsonl'));
  const observations = new Map(events.filter((e) => e.type === 'observation').map((e) => [e.key, e]));
  for (const e of events.filter((e) => e.type === 'entry')) {
    const o = observations.get(`${e.slug}:${e.offset}`), r = resolutions.get(e.slug), end = marketEnd(e.slug);
    if (!o || !r || r.periodEnd !== end || !['Up', 'Down'].includes(r.outcome) || !['Up', 'Down'].includes(e.side) ||
        !Number.isFinite(e.shares) || e.shares <= 0 || !Number.isFinite(e.cost) || e.cost <= 0 ||
        o.at > e.at || e.at >= end * 1000 || o.at < (end - 300) * 1000 || e.cost / e.shares >= 1) { rejected++; continue; }
    const row = { action: e.side === 'Up' ? 1 : 2, x: decisionFeatures(o, config.live.feeRate),
      reward: Number(e.side === r.outcome) - e.cost / e.shares };
    if (!grouped.has(e.slug)) grouped.set(e.slug, { slug: e.slug, end, rows: [] });
    grouped.get(e.slug).rows.push(row);
  }
}
const groups = [...grouped.values()].sort((a, b) => a.end - b.end);
if (groups.length < 100) throw new Error('Insufficient recorded filled markets for chronological replay');
const nTrain = Math.floor(groups.length * 0.70), nVal = Math.floor(groups.length * 0.15);
const train = groups.slice(0, nTrain), val = groups.slice(nTrain, nTrain + nVal), test = groups.slice(nTrain + nVal);
function mse(weights, subset) {
  return subset.reduce((sum, g) => sum + g.rows.reduce((s, r) => s + (actionValues(weights, r.x)[r.action] - r.reward) ** 2, 0) / g.rows.length, 0) / subset.length;
}
const start = performance.now(), baseline = initialPolicy();
let best = structuredClone(baseline), bestVal = mse(best, val), bestEpoch = 0;
const learner = { weights: initialPolicy(), replay: [], updates: 0, experiences: 0 }, epochs = [];
for (let epoch = 1; epoch <= 6; epoch++) {
  for (const g of train) replayUpdate(learner, g, config.lab);
  const validationMSE = mse(learner.weights, val);
  epochs.push({ epoch, validationMSE });
  if (validationMSE < bestVal) { bestVal = validationMSE; best = structuredClone(learner.weights); bestEpoch = epoch; }
}
const report = { generatedAt: new Date().toISOString(), paperOnly: true,
  kind: 'recorded-action reward regression; not a policy-profit backtest', sources, rejected,
  markets: groups.length, trainMarkets: train.length, validationMarkets: val.length, testMarkets: test.length,
  trainThrough: new Date(train.at(-1).end * 1000).toISOString(), testFrom: new Date(test[0].end * 1000).toISOString(),
  selectedEpoch: bestEpoch, epochs, baselineTestMSE: mse(baseline, test), candidateTestMSE: mse(best, test),
  elapsedMs: performance.now() - start, policyId: policyId(best) };
fs.mkdirSync(dir, { recursive: true });
atomicJSON(path.join(dir, 'bootstrap.json'), { weights: best, report });
console.log(JSON.stringify(report, null, 2));
