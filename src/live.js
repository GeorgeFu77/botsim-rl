// PAPER ONLY. Reads public data and local collector files; never submits orders.
import fs from 'node:fs';
import path from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { config } from '../config.js';
import { OnlineLearner } from './core/online.js';
import { marketEnd, fetchResolution } from './core/resolution.js';
import { paperRisk, paperBudget } from './core/paper-risk.js';
import { openPaperLab } from './core/paper-lab.js';
import { extract, WARM, DIM, FEATURE_VERSION, FEATURES } from './core/features.js';
import { JsonlCursor, atomicJSON, appendJSON, readJournal, priceAt, bookAt, paperFill, feePerShare } from './core/live-data.js';

const cfg = config.live;
const serviceStartedAt = Date.now();
const DIR = cfg.resultDir;
const log = (event, data = {}) => console.log(`${new Date().toISOString()} event=${event} ${JSON.stringify(data)}`);
fs.mkdirSync(DIR, { recursive: true });
const saved = JSON.parse(fs.readFileSync('results/model.json', 'utf8'));
if (saved.d !== DIM || saved.featureVersion !== FEATURE_VERSION || JSON.stringify(saved.features) !== JSON.stringify(FEATURES)) {
  throw new Error('Model/feature version mismatch; rebuild episodes and retrain.');
}

// One process owns the append-only account journal. Snapshot is just a view.
const pidFile = path.join(DIR, 'live.pid');
if (fs.existsSync(pidFile)) {
  const old = Number(fs.readFileSync(pidFile, 'utf8'));
  if (!Number.isInteger(old) || old < 1) throw new Error('Invalid live.pid; inspect before starting');
  try { process.kill(old, 0); throw new Error(`Paper agent already running: PID ${old}`); }
  catch (e) { if (e.code !== 'ESRCH') throw e; }
  fs.unlinkSync(pidFile);
}
fs.writeFileSync(pidFile, String(process.pid), { flag: 'wx' });
process.on('exit', () => {
  if (fs.existsSync(pidFile) && fs.readFileSync(pidFile, 'utf8') === String(process.pid)) fs.unlinkSync(pidFile);
});
let stopping = false;
for (const signal of ['SIGTERM', 'SIGINT']) process.on(signal, () => { stopping = true; });

const journal = path.join(DIR, 'events.jsonl');
const resFile = path.join(DIR, 'resolutions.jsonl');
const lab = config.lab.enabled ? openPaperLab(config.lab, cfg) : null;
let labCheckpointAt = Date.now(), labCheckpointSeq = lab?.s.seq ?? 0;
const learner = new OnlineLearner(path.join(DIR, 'learning.json'), saved, cfg.learning);
const observations = new Map();
function remember(e) {
  if (learner.seen.has(e.slug)) return;
  if (!observations.has(e.slug)) observations.set(e.slug, []);
  observations.get(e.slug).push(e);
}
const state = { version: 4, initialBankroll: cfg.bankroll, bankroll: cfg.bankroll,
  open: {}, pending: {}, settled: 0, wins: 0, peakEquity: cfg.bankroll,
  dailyLosses: 0, lossDay: null, drawdownHalted: false,
  pnl: 0, seen: 0, skipped: 0, startedAt: Date.now() };
const observed = new Set(), traded = new Set();
function applyEvent(e) {
  if (e.type === 'account_created') {
    if (e.bankroll !== cfg.bankroll) throw new Error('Paper account starting balance changed; preserve this run and start a separate experiment');
  } else if (e.type === 'observation') {
    observed.add(e.key); state.seen++;
    remember(e);
    if (e.order) state.pending[e.slug] = e.order; else state.skipped++;
  } else if (e.type === 'entry') {
    state.bankroll -= e.cost; state.open[e.slug] = e; traded.add(e.slug); delete state.pending[e.slug];
  } else if (e.type === 'cancel') {
    delete state.pending[e.slug]; state.skipped++;
  } else if (e.type === 'settled') {
    state.bankroll += e.payout; state.pnl += e.pnl; state.settled++; state.wins += Number(e.won);
    delete state.open[e.slug];
    const day = new Date(e.at).toISOString().slice(0, 10);
    if (state.lossDay !== day) { state.lossDay = day; state.dailyLosses = 0; }
    state.dailyLosses += Math.max(0, -e.pnl);
    const equity = state.bankroll + Object.values(state.open).reduce((s, p) => s + p.cost, 0);
    state.peakEquity = Math.max(state.peakEquity, equity);
    if (state.peakEquity - equity >= cfg.maxDrawdown - 1e-8) state.drawdownHalted = true;
  }
}
const events = readJournal(journal);
for (const e of events) { state.startedAt = Math.min(state.startedAt, e.at); applyEvent(e); }
function record(e) {
  e = { modelId: learner.modelId, ...e, at: Date.now() };
  appendJSON(journal, e);
  applyEvent(e);
  if (lab && e.type === 'observation') lab.observe(e, Date.now());
  if (e.type !== 'observation' || e.order) log(e.type, e.type === 'observation'
    ? { slug: e.slug, offset: e.offset, pUp: e.pUp, qUp: e.qUp, side: e.order.side } : e);
}
if (!events.length) record({ type: 'account_created', bankroll: cfg.bankroll, paperOnly: true });
if (lab) for (const e of events) if (e.type === 'observation' && Date.now() - e.at <= cfg.maxFeedAgeMs) lab.observe(e, Date.now());

const cursors = {
  prices: new JsonlCursor(path.join(cfg.botSimDir, 'data/chainlink/prices.jsonl'), 2 * 1024 * 1024),
  books: new JsonlCursor(path.join(cfg.botSimDir, 'data/polymarket/books.jsonl'), 2 * 1024 * 1024),
  resolutions: new JsonlCursor(path.join(cfg.botSimDir, 'data/polymarket/resolutions.jsonl')),
};
let prices = [], books = [];
const outcomes = new Map(readJournal(resFile).map((r) => [r.slug, r]));
// Read the previous run without rewriting its account or claiming it as fresh evaluation.
const historical = (name) => {
  const file = path.join(cfg.bootstrapDir, name);
  if (!fs.existsSync(file)) return [];
  const text = fs.readFileSync(file, 'utf8');
  return text.slice(0, text.lastIndexOf('\n') + 1).split('\n').filter(Boolean).map(JSON.parse);
};
for (const r of historical('resolutions.jsonl')) if (!outcomes.has(r.slug)) outcomes.set(r.slug, r);
for (const e of historical('events.jsonl')) if (e.type === 'observation') remember(e);
if (!learner.bootstrapped) {
  for (const [slug, rows] of [...observations].sort(([a], [b]) => marketEnd(a) - marketEnd(b))) {
    if (learner.learn(slug, rows, outcomes.get(slug), Date.now(), false)) {
      observations.delete(slug);
      if (learner.updates % 256 === 0) { learner.save(); log('learning_bootstrap', { markets: learner.updates }); }
    }
  }
  learner.bootstrapped = true; learner.rotateCandidate(); learner.save();
  log('learning_bootstrap_complete', learner.status());
}
function acceptResolution(r) {
  if (!r || outcomes.has(r.slug)) return;
  appendJSON(resFile, r); outcomes.set(r.slug, r);
}
function updateFeeds(now) {
  prices.push(...cursors.prices.read().filter((r) => r.src === 'chainlink'));
  books.push(...cursors.books.read().filter((r) => r.type === 'book'));
  prices = prices.filter((r) => r.exchTs >= now - 20 * 60_000);
  books = books.filter((r) => r.recvTs >= now - 60_000);
  for (const r of cursors.resolutions.read()) {
    if (r.type !== 'resolution' || !['Up', 'Down'].includes(r.outcome) ||
        !r.source?.startsWith('chainlink ') || marketEnd(r.slug) !== r.periodEnd || r.periodEnd * 1000 > now) continue;
    acceptResolution({ ...r, receivedAt: now });
  }
}

let recovery = null, recoveryError = null, recoverAfter = 0;
const retries = new Map();
function recoverMissing(now) {
  if (recovery || now < recoverAfter || stopping) return;
  const slug = [...new Set([...Object.keys(state.open), ...observations.keys()])].find((s) => {
    const end = marketEnd(s);
    return end && now > end * 1000 + 120_000 && !outcomes.has(s) && (retries.get(s)?.at ?? 0) <= now;
  });
  if (!slug) return;
  recoverAfter = now + 10_000;
  // Bounded read-only request in parallel with decisions; retries persist indefinitely
  // with per-market backoff, so a network outage never freezes learning or fills.
  recovery = fetchResolution(slug).then((r) => {
    if (r) { acceptResolution(r); retries.delete(slug); recoveryError = null; log('resolution_recovered', { slug, outcome: r.outcome }); }
    else retry();
  }).catch((e) => {
    if (['ENOSPC', 'EACCES', 'EROFS', 'EIO'].includes(e.code)) { console.error(e); process.exit(1); }
    recoveryError = e.message; log('resolution_retry', { slug, error: e.message }); retry();
  }).finally(() => { recovery = null; });
  function retry() {
    const attempts = (retries.get(slug)?.attempts ?? 0) + 1;
    retries.set(slug, { attempts, at: Date.now() + Math.min(3_600_000, 60_000 * 2 ** Math.min(attempts, 6)) });
  }
}

function learnResolved(now) {
  let changed = false;
  for (const [slug, rows] of observations) {
    if (!learner.learn(slug, rows, outcomes.get(slug), now)) continue;
    observations.delete(slug); changed = true;
  }
  if (changed) { learner.save(); log('learning_updated', learner.status()); }
}

let cached = { periodStart: 0, k: [] }, retryHistoryAt = 0;
async function history(periodStart) {
  if (cached.periodStart === periodStart) return cached.k;
  if (Date.now() < retryHistoryAt) return null;
  const u = new URL('https://api.exchange.coinbase.com/products/BTC-USD/candles');
  u.searchParams.set('granularity', '300');
  u.searchParams.set('start', new Date(periodStart - WARM * 300_000).toISOString());
  u.searchParams.set('end', new Date(periodStart).toISOString());
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch(u, { signal: AbortSignal.timeout(8000) });
      if (!r.ok) throw new Error(`Coinbase HTTP ${r.status}`);
      const body = await r.json();
      if (!Array.isArray(body)) throw new Error('Invalid Coinbase candle response');
      const k = body.map(([t, l, h, o, c, v]) => ({ t: t * 1000, o, h, l, c, v }))
        .filter((c) => c.t >= periodStart - WARM * 300_000 && c.t < periodStart).sort((a, b) => a.t - b.t);
      if (k.length !== WARM || k.some((c, i) => c.t !== periodStart - (WARM - i) * 300_000 ||
          ![c.o, c.h, c.l, c.c].every((v) => Number.isFinite(v) && v > 0) || !Number.isFinite(c.v) || c.v < 0)) {
        throw new Error('Coinbase history is incomplete or invalid');
      }
      cached = { periodStart, k }; return k;
    } catch (e) {
      if (attempt === 2 || stopping) { retryHistoryAt = Date.now() + 30_000; throw e; }
      await sleep(500 * (attempt + 1));
    }
  }
}

let status = 'starting', errors = 0, lastError = null, lastReport = 0;
async function tick() {
  let now = Date.now();
  updateFeeds(now);
  recoverMissing(now);
  learnResolved(now);
  if (lab) { lab.resolve(outcomes, now); lab.fill(books, now); }
  for (const [slug, pos] of Object.entries(state.open)) {
    const res = outcomes.get(slug);
    if (!res || res.periodEnd * 1000 > now) continue;
    const won = pos.side === res.outcome, payout = won ? pos.shares : 0;
    record({ type: 'settled', slug, side: pos.side, won, payout, pnl: payout - pos.cost,
      outcome: res.outcome, resolutionSource: res.source, entryModelId: pos.modelId });
  }
  for (const [slug, order] of Object.entries(state.pending)) {
    if (now < order.dueAt) continue;
    const book = bookAt(books, slug, order.side, now, cfg);
    if (now > order.dueAt + cfg.maxFeedAgeMs || now >= order.periodEnd * 1000) {
      record({ type: 'cancel', slug, reason: 'expired_pending' }); continue;
    }
    if (!book || book.recvTs < order.dueAt || book.exchTs < order.dueAt) continue;
    const q = book.asks[0].price, cost = q + feePerShare(q, cfg.feeRate);
    const budget = Math.min(order.budget, paperBudget(state, { cost, edge: order.p - cost }, cfg, now, slug));
    const fill = paperFill(book, budget, order.p, cfg);
    if (!fill) { record({ type: 'cancel', slug, reason: 'edge_or_depth_gone' }); continue; }
    record({ type: 'entry', slug, side: order.side, p: order.p, ...fill, periodEnd: order.periodEnd,
      offset: order.offset, decisionAt: order.decisionAt, bookExchTs: book.exchTs, bookRecvTs: book.recvTs,
      modelId: order.modelId, partial: fill.cost < order.budget - 1e-6 });
  }

  const periodStart = Math.floor(now / 300_000) * 300_000;
  const slug = `btc-updown-5m-${periodStart / 1000}`;
  const k = await history(periodStart);
  now = Date.now();
  if (!k || Math.floor(now / 300_000) * 300_000 !== periodStart) { status = 'waiting_history'; return; }
  updateFeeds(now);
  const btc = priceAt(prices, now, now - cfg.feedDelayMs, cfg.maxFeedAgeMs);
  const open = priceAt(prices, periodStart, now - cfg.feedDelayMs, cfg.maxOpenAgeMs);
  const up = bookAt(books, slug, 'Up', now, cfg), down = bookAt(books, slug, 'Down', now, cfg);
  if (!btc || !open || !up || !down) { status = 'waiting_fresh_feeds'; return; }
  if (Math.max(btc.exchTs, up.exchTs, down.exchTs) - Math.min(btc.exchTs, up.exchTs, down.exchTs) > cfg.maxSourceSkewMs) {
    status = 'waiting_aligned_feeds'; return;
  }
  const elapsedSec = (btc.exchTs - periodStart) / 1000;
  const offset = cfg.offsets.find((off) => elapsedSec >= off && elapsedSec <= off + cfg.offsetToleranceSec);
  status = 'ready';
  if (offset === undefined || observed.has(`${slug}:${offset}`)) return;
  const x = Float32Array.from(extract([...k, { t: periodStart }], k.length, open.price, btc.price, elapsedSec));
  if (!x.every(Number.isFinite)) throw new Error('Non-finite live features');
  const prediction = learner.predictions(x), { pUp } = prediction;
  const choices = [['Up', pUp, up], ['Down', 1 - pUp, down]].map(([side, p, book]) => {
    const q = book.asks[0].price, cost = q + feePerShare(q, cfg.feeRate);
    return { side, p, q, cost, edge: p - cost };
  }).filter((c) => c.q >= cfg.priceMin && c.q <= cfg.priceMax).sort((a, b) => b.edge - a.edge);
  let order = null;
  const choice = choices[0];
  if (cfg.paperEntries && choice?.edge > cfg.margin && !traded.has(slug) && !state.pending[slug]) {
    const budget = paperBudget(state, choice, cfg, now);
    if (budget >= cfg.minPaperSpend) order = { side: choice.side, p: choice.p, budget, offset, modelId: learner.modelId,
      decisionAt: now, dueAt: now + cfg.fillLatencyMs, periodEnd: periodStart / 1000 + 300 };
  }
  record({ type: 'observation', key: `${slug}:${offset}`, slug, offset, ...prediction, qUp: up.mid,
    qDown: down.mid, askUp: up.asks[0].price, askDown: down.asks[0].price,
    open5: open.price, btcNow: btc.price, elapsedSec, features: [...x], order,
    priceExchTs: btc.exchTs, priceRecvTs: btc.recvTs, openExchTs: open.exchTs,
    upExchTs: up.exchTs, upRecvTs: up.recvTs, downExchTs: down.exchTs, downRecvTs: down.recvTs });
}

log('started', { pid: process.pid, modelId: learner.modelId, paperOnly: true, resultDir: DIR });
while (!stopping) {
  try { await tick(); }
  catch (e) {
    errors++; lastError = e.message; status = 'error'; log('tick_error', { error: e.message });
    if (['ENOSPC', 'EACCES', 'EROFS', 'EIO'].includes(e.code) || /online|gradient|model|lab|bandit|reward|journal/i.test(e.message)) throw e;
  }
  const now = Date.now();
  const overdue = Object.values(state.open).filter((p) => now - p.periodEnd * 1000 > 600_000).map((p) => p.slug);
  if (lab) {
    atomicJSON(path.join(config.lab.resultDir, 'state.json'), { ...lab.summary(now), pid: process.pid,
      heartbeatAt: now, serviceStartedAt, feedStatus: status, sourceErrors: errors, lastError });
    if (lab.s.seq !== labCheckpointSeq && now - labCheckpointAt >= 60_000) {
      lab.checkpoint(); labCheckpointAt = now; labCheckpointSeq = lab.s.seq;
    }
  }
  atomicJSON(path.join(DIR, 'live-state.json'), { ...state, pid: process.pid, modelId: learner.modelId, heartbeatAt: now,
    status, errors, lastError, overdue, paperOnly: true, risk: paperRisk(state, cfg, now),
    learning: { ...learner.status(), waitingMarkets: observations.size,
      stale: !learner.lastMarketEnd || now - learner.lastMarketEnd > 1_800_000 }, recoveryError });
  if (now - lastReport >= 60_000) {
    if (lab) log('lab_status', { cash: lab.s.accounts.brain.bankroll, pnl: lab.s.accounts.brain.pnl,
      workers: config.lab.workers, updates: lab.s.learning.updates, experiences: lab.s.learning.experiences,
      evaluatedMarkets: lab.s.evaluatedMarkets, promotions: lab.s.promotions });
    log('status', { status, bankroll: state.bankroll, pnl: state.pnl, settled: state.settled,
      seen: state.seen, open: Object.keys(state.open).length, overdue, errors }); lastReport = now;
  }
  if (!stopping) await sleep(cfg.tickMs);
}
if (recovery) await recovery;
learnResolved(Date.now());
if (lab) { lab.resolve(outcomes, Date.now()); lab.checkpoint(); }
log('stopped');
