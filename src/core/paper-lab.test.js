import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { config } from '../../config.js';
import { PaperLab, openPaperLab } from './paper-lab.js';
import { initialPolicy, actionValues, decisionFeatures, policyId, replayUpdate } from './bandit.js';
import { atomicJSON, readJournal } from './live-data.js';

const cfg = { ...config.lab, workers: 6, profiles: [{ epsilon: 0, margin: 0.02, stake: 0.40 }] };
const start = 1_789_002_000;
function frame(i = 0, outcome = 'Up') {
  const t = start + 300 * i, slug = `btc-updown-5m-${t}`, at = (t + 60) * 1000;
  const observation = { slug, key: `${slug}:60`, offset: 60, at, pUp: 0.8,
    askUp: 0.40, askDown: 0.61, elapsedSec: 60, features: new Array(25).fill(0.01) };
  const books = ['Up', 'Down'].map((side) => ({ slug, outcome: side, recvTs: at + 1100, exchTs: at + 1100,
    bids: [{ price: side === 'Up' ? 0.38 : 0.59, size: 10 }], asks: [{ price: side === 'Up' ? 0.40 : 0.61, size: 10 }] }));
  const resolution = { slug, periodEnd: t + 300, outcome, source: 'synthetic test' };
  return { observation, books, resolution, at, fillAt: at + 2200, end: (t + 301) * 1000, slug };
}
function run(lab, f) {
  lab.observe(f.observation, f.at); lab.fill(f.books, f.fillAt);
  lab.resolve(new Map([[f.slug, f.resolution]]), f.end);
}
function temporary(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'botsimrl-lab-'));
  t.after(() => fs.rmSync(dir, { recursive: true }));
  atomicJSON(path.join(dir, 'bootstrap.json'), { weights: initialPolicy(), report: { trainMarkets: 0 } });
  return dir;
}

test('base reward policy reproduces after-fee forecast edge and keeps abstention at zero', () => {
  const f = frame(), x = decisionFeatures(f.observation, config.live.feeRate);
  const values = actionValues(initialPolicy(), x);
  assert.equal(values[0], 0);
  assert.ok(Math.abs(values[1] - (0.8 - 0.4 - 0.07 * 0.4 * 0.6)) < 1e-12);
  assert.ok(values[2] < 0);
  assert.throws(() => decisionFeatures({ ...f.observation, pUp: NaN }, 0.07), /Invalid/);
});

test('accounts are independent, fills use later books and central P/L never pools worker money', () => {
  const lab = new PaperLab(cfg, config.live), f = frame();
  assert.equal(lab.observe(f.observation, f.at), true);
  assert.equal(lab.observe(f.observation, f.at), false);
  assert.equal(lab.s.accounts.brain.bankroll, 80);
  lab.fill(f.books, f.at + 1000);
  assert.equal(Object.keys(lab.s.accounts.brain.open).length, 0);
  lab.fill(f.books, f.fillAt);
  assert.equal(lab.s.accounts.brain.bankroll, 79.6);
  assert.equal(lab.s.accounts['worker-0'].bankroll, 79.6);
  const pos = lab.s.accounts.brain.open[f.slug];
  assert.ok(pos.fees > 0); assert.ok(pos.cost <= 0.4);
  lab.resolve(new Map([[f.slug, f.resolution]]), f.end);
  assert.ok(Math.abs(lab.s.accounts.brain.pnl - (pos.shares - pos.cost)) < 1e-12);
  assert.ok(Math.abs(lab.s.accounts.brain.bankroll - (80 + lab.s.accounts.brain.pnl)) < 1e-12);
  assert.equal(lab.s.learning.updates, 1);
  assert.equal(lab.s.learning.experiences, 1); // Six copies are one filled-action example.
});

test('32 versus 128 identical workers do not multiply evidence or change the learner', () => {
  const a = new PaperLab({ ...cfg, workers: 32 }, config.live), b = new PaperLab({ ...cfg, workers: 128 }, config.live);
  run(a, frame()); run(b, frame());
  assert.equal(a.s.learning.experiences, b.s.learning.experiences);
  assert.deepEqual(a.s.learning.weights, b.s.learning.weights);
  const s = { weights: initialPolicy(), replay: [], updates: 0, experiences: 0 };
  replayUpdate(s, { slug: 'm', rows: [
    { action: 1, x: new Array(8).fill(0.1), reward: 0.3 },
    { action: 1, x: new Array(8).fill(0.1), reward: 0.30000000000000004 },
  ] }, cfg);
  assert.equal(s.experiences, 1);
});

test('workers continue learning after central paper loss limits stop entries', () => {
  const lab = new PaperLab(cfg, config.live);
  const f = frame();
  lab.s.accounts.brain.lossDay = new Date(f.at).toISOString().slice(0, 10);
  lab.s.accounts.brain.dailyLosses = 1.60;
  run(lab, f);
  assert.equal(lab.s.accounts.brain.settled, 0);
  assert.equal(lab.s.accounts.brain.bankroll, 80);
  assert.equal(lab.s.accounts['worker-0'].settled, 1);
  assert.equal(lab.s.learning.updates, 1);
});

test('bankrupt worker episodes retain losses and deposits; the central account never resets', () => {
  const lab = new PaperLab(cfg, config.live);
  for (const id of ['brain', 'worker-0']) { lab.s.accounts[id].bankroll = 0.05; lab.s.accounts[id].pnl = -79.95; }
  lab.resolve(new Map(), frame().at);
  const worker = lab.s.accounts['worker-0'];
  assert.equal(worker.bankroll, 80); assert.equal(worker.pnl, -79.95); assert.equal(worker.resets, 1);
  assert.equal(worker.deposited, 159.95);
  assert.equal(lab.s.accounts.brain.bankroll, 0.05); assert.equal(lab.s.accounts.brain.resets, 0);
  assert.throws(() => lab.emit('episode_reset', { id: 'brain' }, frame().at), /Invalid/);
});

test('no future labels, repeated settlement, stale observations or stale fills', () => {
  const lab = new PaperLab(cfg, config.live), f = frame();
  assert.equal(lab.observe(f.observation, f.at + 11_000), false);
  lab.observe(f.observation, f.at);
  lab.resolve(new Map([[f.slug, f.resolution]]), f.at);
  assert.equal(lab.s.learning.updates, 0);
  lab.fill(f.books, f.at + 30_000);
  assert.equal(Object.keys(lab.s.accounts.brain.pending).length, 0);
  lab.resolve(new Map([[f.slug, f.resolution]]), f.end);
  const seq = lab.s.seq;
  lab.resolve(new Map([[f.slug, f.resolution]]), f.end + 10_000);
  assert.equal(lab.s.seq, seq); assert.equal(lab.s.accounts.brain.bankroll, 80);
});

test('checkpoint plus unapplied journal tail restores pending/fills/learning without duplication', (t) => {
  const c = { ...cfg, resultDir: temporary(t) }, a = openPaperLab(c, config.live), f = frame();
  a.observe(f.observation, f.at); a.checkpoint();
  a.fill(f.books, f.fillAt); // Simulate a crash before checkpointing this fill batch.
  const b = openPaperLab(c, config.live);
  assert.deepEqual(b.s, a.s);
  b.resolve(new Map([[f.slug, f.resolution]]), f.end);
  const d = openPaperLab(c, config.live);
  assert.deepEqual(d.s, b.s);
  assert.equal(d.s.learning.updates, 1);
  assert.equal(d.observe(f.observation, f.at), false);
  assert.throws(() => openPaperLab({ ...c, workers: 7 }, config.live), /configuration/);
});

test('torn journal tail after a checkpoint preserves the earlier prefix', (t) => {
  const c = { ...cfg, resultDir: temporary(t) }, a = openPaperLab(c, config.live), f = frame();
  a.observe(f.observation, f.at); a.checkpoint();
  const file = path.join(c.resultDir, 'events.jsonl'), before = fs.readFileSync(file, 'utf8');
  fs.appendFileSync(file, '{"seq":');
  const b = openPaperLab(c, config.live);
  assert.equal(fs.readFileSync(file, 'utf8'), before); assert.deepEqual(b.s, a.s);
  assert.equal(readJournal(file).length, 1);
});

test('three simulated days survive repeated serialized restores with bounded replay', () => {
  const c = { ...cfg, replayMarkets: 32 };
  let a = new PaperLab(c, config.live);
  const b = new PaperLab(c, config.live);
  for (let i = 0; i < 864; i++) {
    const f = frame(i, i % 3 ? 'Up' : 'Down'); run(a, f); run(b, f);
    if (i % 97 === 0) a = new PaperLab(c, config.live, undefined, JSON.parse(JSON.stringify(a.s)));
  }
  assert.deepEqual(a.s, b.s);
  assert.equal(a.s.learning.replay.length, 32);
  assert.equal(a.s.resolved.length, 864);
  assert.ok(a.s.learning.updates > a.s.accounts.brain.settled);
});

test('a candidate cannot replace the central policy before enough future time and samples', () => {
  const c = { ...cfg, evaluationMarkets: 2, evaluationMs: 300_000, minimumEvaluationTrades: 1 };
  const lab = new PaperLab(c, config.live);
  lab.s.candidate[0][0] = 0.1;
  const candidateId = policyId(lab.s.candidate), original = policyId(lab.s.champion);
  run(lab, frame());
  assert.equal(policyId(lab.s.champion), original);
  lab.s.accounts.candidate.pnl = 3; lab.s.accounts.reference.pnl = 0;
  run(lab, frame(1));
  assert.equal(lab.s.promotions, 1); assert.equal(policyId(lab.s.champion), candidateId);
  assert.equal(lab.s.accounts.brain.deposited, 80);
  assert.equal(lab.s.lastEvaluation.markets, 2);
});
