import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { MLP } from './mlp.js';
import { OnlineLearner } from './online.js';
import { paperRisk, paperBudget } from './paper-risk.js';
import { confirmedResolution } from './resolution.js';
import { readJournal, appendJSON, paperFill } from './live-data.js';
import { config } from '../../config.js';

const start = 1_789_000_200; // aligned to a five-minute boundary
const base = { ...new MLP(2, { hidden: 3 }).toJSON(), platt: { a: 0.8, b: 0.1 } };
const cfg = { ...config.live.learning, replayMarkets: 16, batchMarkets: 4 };
function temporary(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'botsimrl-online-'));
  t.after(() => fs.rmSync(dir, { recursive: true }));
  return dir;
}
function market(i, learner, y = 1) {
  const t = start + i * 300, slug = `btc-updown-5m-${t}`;
  const observation = { type: 'observation', slug, key: `${slug}:60`, at: (t + 60) * 1000,
    features: [0.2, -0.1], qUp: 0.5, ...learner.predictions([0.2, -0.1]) };
  return { slug, rows: [observation], res: { slug, periodEnd: t + 300, outcome: y ? 'Up' : 'Down' }, now: (t + 301) * 1000 };
}
const feed = (l, m, evaluate = true) => l.learn(m.slug, m.rows, m.res, m.now, evaluate);

test('updates weights only after a matching resolved outcome; duplicate outcomes are harmless', (t) => {
  const l = new OnlineLearner(path.join(temporary(t), 'learning.json'), base, cfg);
  const before = l.status(), m = market(0, l);
  assert.equal(l.learn(m.slug, m.rows, m.res, m.now - 2000), false);
  assert.equal(l.learn(m.slug, m.rows, { ...m.res, slug: 'wrong' }, m.now), false);
  assert.equal(l.learn(m.slug, [{ ...m.rows[0], at: m.now }], m.res, m.now), false);
  assert.equal(l.status().learnerId, before.learnerId);
  assert.equal(feed(l, m), true);
  assert.notEqual(l.status().learnerId, before.learnerId);
  assert.equal(l.modelId, before.modelId); // training is not automatic deployment
  assert.equal(l.candidateId, before.candidateId);
  assert.equal(feed(l, m), false);
  assert.equal(l.updates, 1);
});

test('three days, delayed labels and repeated restarts preserve exactly-once deterministic learning', (t) => {
  const dir = temporary(t), file = path.join(dir, 'learning.json');
  let l = new OnlineLearner(file, base, cfg);
  const continuous = new OnlineLearner(path.join(dir, 'continuous.json'), base, cfg);
  for (let i = 0; i < 3 * 288; i++) {
    const m = market(i, l, i % 3 !== 0);
    assert.equal(feed(l, m), true); feed(continuous, m);
    if (i % 97 === 0) {
      l.save(); l = new OnlineLearner(file, base, cfg);
      assert.equal(feed(l, m), false);
      assert.equal(l.status().learnerId, continuous.status().learnerId);
    }
  }
  l.save(); l = new OnlineLearner(file, base, cfg);
  assert.equal(l.updates, 864);
  assert.equal(l.replay.length, cfg.replayMarkets);
  assert.equal(l.status().learnerId, continuous.status().learnerId);
  assert.equal(l.lastMarketEnd, (start + 864 * 300) * 1000);
});

test('candidate gate uses forward logged predictions, market-level weighting and a minimum time span', (t) => {
  const l = new OnlineLearner(path.join(temporary(t), 'learning.json'), base,
    { ...cfg, evaluationMarkets: 3, evaluationMs: 600_000 });
  feed(l, market(0, l), false); l.rotateCandidate();
  const candidate = l.candidateId.split(':')[1];
  for (let i = 1; i <= 3; i++) {
    const m = market(i, l);
    m.rows[0].pUp = 0.4; m.rows[0].candidatePUp = 0.9;
    m.rows.push({ ...m.rows[0], key: `${m.slug}:120`, at: m.rows[0].at + 60_000 });
    feed(l, m);
    if (i < 3) assert.equal(l.promotions, 0);
  }
  assert.equal(l.promotions, 1);
  assert.equal(l.modelId, candidate);
  assert.equal(l.lastEvaluation.n, 3); // six offsets are not six independent markets
  assert.equal(l.evaluation.n, 0);
  assert.notEqual(l.modelId, l.status().learnerId); // promoted frozen candidate, not newly trained weights
});

test('a worse candidate stays in shadow; stale-cycle predictions cannot pass the next gate', (t) => {
  const l = new OnlineLearner(path.join(temporary(t), 'learning.json'), base,
    { ...cfg, evaluationMarkets: 2, evaluationMs: 300_000 });
  const before = l.modelId, delayed = market(0, l);
  for (let i = 1; i <= 2; i++) {
    const m = market(i, l); m.rows[0].pUp = 0.9; m.rows[0].candidatePUp = 0.1; feed(l, m);
  }
  assert.equal(l.modelId, before); assert.equal(l.promotions, 0);
  assert.equal(l.lastEvaluation.promoted, false);
  feed(l, delayed);
  assert.equal(l.updates, 3); assert.equal(l.evaluation.n, 0);
  assert.equal(l.lastMarketEnd, (start + 900) * 1000);
});

test('checkpoint corruption or changed base model never silently resets training', (t) => {
  const file = path.join(temporary(t), 'learning.json');
  const l = new OnlineLearner(file, base, cfg); feed(l, market(0, l)); l.save();
  assert.throws(() => new OnlineLearner(file, { ...base, b2: 20 }, cfg), /mismatch/);
  fs.writeFileSync(file, '{"version":');
  assert.throws(() => new OnlineLearner(file, base, cfg), SyntaxError);
});

function account() {
  return { bankroll: 80, peakEquity: 80, open: {}, pending: {}, lossDay: null, dailyLosses: 0, drawdownHalted: false };
}
test('paper limits cap confidence sizing and include fees, open positions and pending commitments', () => {
  const s = account(), now = start * 1000, c = { cost: 0.5, edge: 0.4 };
  assert.equal(paperBudget(s, c, config.live, now), 0.4);
  const fill = paperFill({ asks: [{ price: 0.5, size: 10 }] }, 0.4, 0.9, config.live);
  assert.ok(fill.cost <= 0.4 + 1e-12); assert.ok(fill.fees > 0);
  s.bankroll -= 0.4; s.open.one = { cost: 0.4, periodEnd: start + 300 };
  s.pending.two = { budget: 0.4 };
  assert.equal(paperRisk(s, config.live, now).capacity, 0);
  assert.ok(paperBudget(s, c, config.live, now, 'two') <= 0.4);
});

test('daily gross loss budget reserves unsettled losses, resets next UTC day, but drawdown does not', () => {
  const s = account(), now = start * 1000;
  s.lossDay = new Date(now).toISOString().slice(0, 10); s.dailyLosses = 1.3;
  s.open.one = { cost: 0.3, periodEnd: start + 300 }; s.bankroll -= 0.3;
  assert.equal(paperRisk(s, config.live, now).reason, 'daily_loss_limit');
  s.open = {}; s.bankroll = 78.4;
  assert.equal(paperRisk(s, config.live, now + 86_400_000).reason, null);
  s.bankroll = 72; s.drawdownHalted = true;
  assert.equal(paperRisk(s, config.live, now + 3 * 86_400_000).reason, 'drawdown_halt');
});

test('overdue settlements block new paper exposure; no invented release of funds', () => {
  const s = account(); s.open.one = { cost: 0.4, periodEnd: start };
  assert.equal(paperRisk(s, config.live, (start + 601) * 1000).reason, 'waiting_overdue_settlement');
});

test('ten unattended losing paper days cannot bypass the persistent drawdown budget', () => {
  const s = account(), c = { cost: 0.5, edge: 0.4 };
  let totalEntries = 0;
  for (let day = 0; day < 10; day++) {
    const now = (start + day * 86_400) * 1000;
    s.lossDay = new Date(now).toISOString().slice(0, 10); s.dailyLosses = 0;
    for (let i = 0; i < 288; i++) {
      const budget = paperBudget(s, c, config.live, now);
      if (budget < config.live.minPaperSpend) continue;
      s.bankroll -= budget; s.dailyLosses += budget; totalEntries++;
      assert.ok(s.dailyLosses <= config.live.dailyLossLimit + 1e-9);
      assert.ok(s.bankroll >= 72 - 1e-9);
    }
  }
  assert.ok(totalEntries > 10);
  assert.equal(paperRisk(s, config.live, (start + 11 * 86_400) * 1000).reason, 'drawdown_halt');
});

test('recovery only accepts an exact, final, matching official outcome', () => {
  const slug = `btc-updown-5m-${start}`, now = (start + 400) * 1000;
  const body = { slug, closed: true, umaResolutionStatus: 'resolved', endDate: new Date((start + 300) * 1000).toISOString(),
    outcomes: '["Down","Up"]', outcomePrices: '["1","0"]' };
  assert.equal(confirmedResolution(body, slug, now).outcome, 'Down');
  for (const patch of [{ closed: false }, { umaResolutionStatus: 'proposed' }, { slug: 'wrong' },
    { outcomePrices: '["0.999","0.001"]' }, { outcomes: '["Up","Up"]' }, { endDate: new Date(now).toISOString() }]) {
    assert.equal(confirmedResolution({ ...body, ...patch }, slug, now), null);
  }
  assert.equal(confirmedResolution(body, slug, now - 500_000), null);
});

test('torn final journal records are backed up; complete malformed records fail loudly', (t) => {
  const dir = temporary(t), file = path.join(dir, 'events.jsonl');
  appendJSON(file, { n: 1 }); fs.appendFileSync(file, '{"n":');
  assert.deepEqual(readJournal(file), [{ n: 1 }]);
  assert.ok(fs.readdirSync(dir).some((name) => name.startsWith('events.jsonl.partial-')));
  appendJSON(file, { n: 2 }); assert.deepEqual(readJournal(file), [{ n: 1 }, { n: 2 }]);
  fs.appendFileSync(file, '{"n":3}'); assert.equal(readJournal(file).length, 3);
  fs.appendFileSync(file, 'broken\n'); assert.throws(() => readJournal(file), SyntaxError);
});
