import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { appendJSON, atomicJSON, readJournal, bookAt, fillAsks } from './live-data.js';
import { paperRisk } from './paper-risk.js';
import { marketEnd } from './resolution.js';
import { mulberry32 } from './mlp.js';
import { initialPolicy, validatePolicy, decisionFeatures, actionValues, chooseAction, replayUpdate, policyId } from './bandit.js';

const copy = (x) => structuredClone(x);
const account = (id, balance) => ({ id, bankroll: balance, deposited: balance, pnl: 0, fees: 0, settled: 0,
  wins: 0, resets: 0, peakEquity: balance, maxDrawdown: 0, dailyLosses: 0, lossDay: null,
  drawdownHalted: false, pending: {}, open: {}, lastEnteredSlug: null });
const empty = (a) => !Object.keys(a.open).length && !Object.keys(a.pending).length;
const seedFor = (s) => { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; };

export class PaperLab {
  constructor(cfg, execution, seed, restored = null, persist = () => {}) {
    if (!Number.isInteger(cfg.workers) || cfg.workers < 1 || cfg.workers > 512 || cfg.bankroll !== 80) throw new Error('Invalid paper lab configuration');
    if (![cfg.lr, cfg.maxSlippage, cfg.evaluationMs, cfg.minimumEvaluationGain, cfg.allowedDrawdownDifference].every((n) => Number.isFinite(n) && n >= 0) ||
        ![cfg.replayMarkets, cfg.batchMarkets, cfg.evaluationMarkets, cfg.minimumEvaluationTrades].every((n) => Number.isInteger(n) && n > 0) ||
        !cfg.profiles?.length || cfg.profiles.some((p) => ![p.epsilon, p.margin, p.stake].every(Number.isFinite) ||
          p.epsilon < 0 || p.epsilon > 1 || p.margin < 0 || p.stake <= 0 || p.stake > cfg.bankroll)) throw new Error('Invalid paper lab learning/profile settings');
    this.cfg = cfg; this.execution = execution; this.persist = persist;
    if (restored) this.s = restored;
    else {
      const accounts = Object.fromEntries(['brain', 'candidate', 'reference', ...Array.from({ length: cfg.workers }, (_, i) => `worker-${i}`)]
        .map((id) => [id, account(id, cfg.bankroll)]));
      const weights = copy(seed?.weights ?? initialPolicy());
      this.s = { version: 1, seq: 0, startedAt: null, accounts, learning: { weights, replay: [], updates: 0, experiences: 0 },
        champion: initialPolicy(), candidate: copy(weights), seed: seed?.report ?? null, cycle: 1, promotions: 0,
        observed: [], resolved: [], markets: {}, evaluatedMarkets: 0, evaluationStart: null, lastEvaluation: null,
        lastObservationAt: null, lastLearningAt: null, lastMarketEnd: null };
    }
    for (const w of [this.s.learning.weights, this.s.champion, this.s.candidate]) validatePolicy(w);
    if (this.s.version !== 1 || !['brain', 'candidate', 'reference'].every((id) => this.s.accounts[id]?.id === id) ||
        Object.keys(this.s.accounts).length !== cfg.workers + 3 || this.s.learning.replay.length > cfg.replayMarkets ||
        !Number.isSafeInteger(this.s.seq) || this.s.seq < 0) throw new Error('Invalid lab checkpoint');
    for (const a of Object.values(this.s.accounts)) {
      if (![a.bankroll, a.pnl, a.deposited, a.fees, a.peakEquity].every(Number.isFinite) || a.bankroll < -1e-8) throw new Error('Invalid paper account checkpoint');
    }
    this.observed = new Set(this.s.observed); this.resolved = new Set(this.s.resolved);
  }

  emit(type, data, at) {
    const e = { seq: this.s.seq + 1, type, at, ...data };
    this.persist(e); this.apply(e);
  }

  observe(o, now) {
    const end = marketEnd(o.slug);
    if (this.observed.has(o.key) || this.resolved.has(o.slug) || !end || !Number.isFinite(o.at) ||
        o.key !== `${o.slug}:${o.offset}` || !this.execution.offsets.includes(o.offset) ||
        o.at < (end - 300) * 1000 || o.at >= end * 1000 || !Number.isFinite(now) || now >= end * 1000 ||
        now < o.at || now - o.at > this.execution.maxFeedAgeMs) return false;
    const x = decisionFeatures(o, this.execution.feeRate), orders = [], policies = new Map();
    for (const a of Object.values(this.s.accounts)) {
      if (!empty(a) || a.lastEnteredSlug === o.slug) continue;
      const worker = a.id.startsWith('worker-'), index = worker ? Number(a.id.slice(7)) : 0;
      const profile = this.cfg.profiles[index % this.cfg.profiles.length];
      const weights = worker ? this.s.learning.weights : a.id === 'candidate' ? this.s.candidate : this.s.champion;
      if (!policies.has(weights)) policies.set(weights, { values: actionValues(weights, x), id: policyId(weights) });
      const policy = policies.get(weights);
      const random = mulberry32(seedFor(`${a.id}:${o.key}:${a.resets}`));
      const action = chooseAction(policy.values, random, worker ? profile.epsilon : 0, worker ? profile.margin : this.execution.margin);
      if (!action) continue;
      const budget = a.id === 'brain'
        ? Math.min(paperRisk(a, this.execution, now).capacity, a.bankroll * this.execution.maxStake, this.execution.maxStakeDollars)
        : Math.min(a.bankroll, worker ? profile.stake : this.execution.maxStakeDollars);
      if (budget < this.execution.minPaperSpend) continue;
      orders.push({ id: a.id, slug: o.slug, key: o.key, action, side: action === 1 ? 'Up' : 'Down', budget,
        periodEnd: end, dueAt: now + this.execution.fillLatencyMs, decisionAt: now, x,
        unitCost: x[action + 1], score: policy.values[action], exploratory: worker,
        policyId: policy.id, cycle: this.s.cycle });
    }
    this.emit('observation', { key: o.key, slug: o.slug, periodEnd: end, sourceAt: o.at, orders }, now);
    return true;
  }

  fill(books, now) {
    const fills = [], cancels = [], cache = new Map();
    for (const a of Object.values(this.s.accounts)) for (const order of Object.values(a.pending)) {
      if (now < order.dueAt) continue;
      if (now > order.dueAt + this.execution.maxFeedAgeMs || now >= order.periodEnd * 1000) {
        cancels.push({ id: a.id, slug: order.slug, reason: 'expired' }); continue;
      }
      const key = `${order.slug}:${order.side}`;
      if (!cache.has(key)) cache.set(key, bookAt(books, order.slug, order.side, now, this.execution));
      const book = cache.get(key);
      if (!book || book.recvTs < order.dueAt || book.exchTs < order.dueAt) continue;
      const budget = Math.min(order.budget, a.bankroll, a.id === 'brain'
        ? paperRisk(a, this.execution, now, order.slug).capacity : Infinity);
      const maxCost = order.unitCost + Math.min(this.cfg.maxSlippage,
        order.exploratory ? this.cfg.maxSlippage : Math.max(0, order.score - this.execution.margin));
      const fill = fillAsks(book, budget, this.execution, (p, fee) => p + fee <= maxCost + 1e-12);
      if (!fill) { cancels.push({ id: a.id, slug: order.slug, reason: 'risk_price_or_depth' }); continue; }
      fills.push({ id: a.id, slug: order.slug, ...fill, bookExchTs: book.exchTs, bookRecvTs: book.recvTs });
    }
    if (fills.length || cancels.length) this.emit('execution', { fills, cancels }, now);
  }

  resolve(outcomes, now) {
    for (const slug of Object.keys(this.s.markets)) {
      const r = outcomes.get(slug), end = marketEnd(slug);
      if (!r || r.slug !== slug || r.periodEnd !== end || end * 1000 > now || !['Up', 'Down'].includes(r.outcome)) continue;
      this.emit('resolution', { slug, outcome: r.outcome, periodEnd: end, source: r.source }, now);
    }
    const s = this.s;
    if (s.evaluatedMarkets >= this.cfg.evaluationMarkets && now - s.evaluationStart >= this.cfg.evaluationMs &&
        empty(s.accounts.reference) && empty(s.accounts.candidate)) this.emit('evaluation', {}, now);
    for (const a of Object.values(s.accounts)) {
      if (a.id.startsWith('worker-') && empty(a) && a.bankroll < this.execution.minPaperSpend) this.emit('episode_reset', { id: a.id }, now);
    }
  }

  apply(e) {
    if (e.seq !== this.s.seq + 1 || !Number.isFinite(e.at)) throw new Error('Invalid lab journal sequence');
    const s = this.s; s.seq = e.seq; s.startedAt ??= e.at;
    if (e.type === 'observation') {
      if (this.observed.has(e.key) || this.resolved.has(e.slug)) throw new Error('Duplicate lab observation');
      this.observed.add(e.key); s.observed.push(e.key); s.lastObservationAt = e.at;
      s.markets[e.slug] ??= { cycle: s.cycle, firstAt: e.at };
      s.evaluationStart ??= e.at;
      for (const order of e.orders) s.accounts[order.id].pending[order.slug] = order;
    } else if (e.type === 'execution') {
      for (const f of e.fills) {
        const a = s.accounts[f.id], order = a?.pending[f.slug];
        if (!order || a.open[f.slug] || ![f.cost, f.shares, f.fees].every(Number.isFinite) ||
            f.cost <= 0 || f.shares <= 0 || f.fees < 0 || f.cost > Math.min(order.budget, a.bankroll) + 1e-8) throw new Error('Invalid paper fill journal');
        a.bankroll -= f.cost; a.fees += f.fees; a.open[f.slug] = { ...order, ...f };
        a.lastEnteredSlug = f.slug; delete a.pending[f.slug];
      }
      for (const c of e.cancels) delete s.accounts[c.id].pending[c.slug];
    } else if (e.type === 'resolution') {
      if (this.resolved.has(e.slug) || !s.markets[e.slug] || marketEnd(e.slug) !== e.periodEnd ||
          e.periodEnd * 1000 > e.at || !['Up', 'Down'].includes(e.outcome)) throw new Error('Invalid lab resolution journal');
      const rows = [];
      for (const a of Object.values(s.accounts)) {
        delete a.pending[e.slug];
        const pos = a.open[e.slug];
        if (!pos) continue;
        const won = pos.side === e.outcome, payout = won ? pos.shares : 0, pnl = payout - pos.cost;
        a.bankroll += payout; a.pnl += pnl; a.settled++; a.wins += Number(won); delete a.open[e.slug];
        const day = new Date(e.at).toISOString().slice(0, 10);
        if (a.lossDay !== day) { a.lossDay = day; a.dailyLosses = 0; }
        a.dailyLosses += Math.max(0, -pnl);
        const equity = a.bankroll + Object.values(a.open).reduce((n, p) => n + p.cost, 0);
        a.peakEquity = Math.max(a.peakEquity, equity);
        a.maxDrawdown = Math.max(a.maxDrawdown, a.peakEquity - equity);
        if (a.id === 'brain' && a.maxDrawdown >= this.execution.maxDrawdown - 1e-8) a.drawdownHalted = true;
        if (a.id.startsWith('worker-')) rows.push({ action: pos.action, x: pos.x, reward: pnl / pos.shares });
      }
      replayUpdate(s.learning, { slug: e.slug, rows }, this.cfg);
      if (rows.length) s.lastLearningAt = e.at;
      if (s.markets[e.slug].cycle === s.cycle) s.evaluatedMarkets++;
      s.lastMarketEnd = Math.max(s.lastMarketEnd ?? 0, e.periodEnd * 1000);
      this.resolved.add(e.slug); s.resolved.push(e.slug); delete s.markets[e.slug];
    } else if (e.type === 'evaluation') {
      const candidate = s.accounts.candidate, reference = s.accounts.reference;
      const promoted = candidate.settled >= this.cfg.minimumEvaluationTrades && reference.settled >= this.cfg.minimumEvaluationTrades &&
        candidate.pnl > 0 && candidate.pnl - reference.pnl >= this.cfg.minimumEvaluationGain &&
        candidate.maxDrawdown <= reference.maxDrawdown + this.cfg.allowedDrawdownDifference;
      s.lastEvaluation = { at: e.at, markets: s.evaluatedMarkets, promoted,
        candidate: { pnl: candidate.pnl, trades: candidate.settled, drawdown: candidate.maxDrawdown },
        reference: { pnl: reference.pnl, trades: reference.settled, drawdown: reference.maxDrawdown } };
      if (promoted) { s.champion = copy(s.candidate); s.promotions++; }
      s.candidate = copy(s.learning.weights); s.cycle++; s.evaluatedMarkets = 0; s.evaluationStart = null;
      s.accounts.candidate = account('candidate', this.cfg.bankroll); s.accounts.reference = account('reference', this.cfg.bankroll);
    } else if (e.type === 'episode_reset') {
      const a = s.accounts[e.id];
      if (!a?.id.startsWith('worker-') || !empty(a) || a.bankroll >= this.execution.minPaperSpend) throw new Error('Invalid paper episode reset');
      const added = this.cfg.bankroll - a.bankroll;
      a.bankroll += added; a.deposited += added; a.resets++; a.peakEquity = this.cfg.bankroll;
    } else throw new Error(`Unknown lab event: ${e.type}`);
  }

  summary(now) {
    const s = this.s, brain = s.accounts.brain;
    return { paperOnly: true, startedAt: s.startedAt, brain: copy(brain), risk: paperRisk(brain, this.execution, now),
      workers: this.cfg.workers, workerResets: Object.values(s.accounts).reduce((n, a) => n + a.resets, 0),
      observations: s.observed.length, resolvedMarkets: s.resolved.length, waitingMarkets: Object.keys(s.markets).length,
      learning: { updates: s.learning.updates, uniqueExperiences: s.learning.experiences, replayMarkets: s.learning.replay.length,
        learnerId: policyId(s.learning.weights), activeId: policyId(s.champion), candidateId: policyId(s.candidate),
        lastLearningAt: s.lastLearningAt, lastMarketEnd: s.lastMarketEnd, promotions: s.promotions,
        evaluatedMarkets: s.evaluatedMarkets, requiredMarkets: this.cfg.evaluationMarkets, lastEvaluation: s.lastEvaluation,
        seed: s.seed, stale: !s.lastMarketEnd || now - s.lastMarketEnd > 1_800_000 } };
  }
}

export function openPaperLab(cfg, execution) {
  fs.mkdirSync(cfg.resultDir, { recursive: true });
  const file = (name) => path.join(cfg.resultDir, name);
  const signature = createHash('sha256').update(JSON.stringify({ cfg, execution })).digest('hex');
  const seed = JSON.parse(fs.readFileSync(file('bootstrap.json'), 'utf8'));
  const saved = fs.existsSync(file('checkpoint.json')) ? JSON.parse(fs.readFileSync(file('checkpoint.json'), 'utf8')) : null;
  if (saved && (saved.signature !== signature || saved.seedHash !== policyId(seed.weights))) throw new Error('Lab configuration/seed changed; preserve this experiment before starting another');
  if (saved?.offset && !fs.existsSync(file('events.jsonl'))) throw new Error('Missing lab journal');
  const lab = new PaperLab(cfg, execution, seed, saved?.state, (e) => appendJSON(file('events.jsonl'), e));
  for (const e of readJournal(file('events.jsonl'), saved?.offset ?? 0)) lab.apply(e);
  lab.checkpoint = () => atomicJSON(file('checkpoint.json'), { signature, seedHash: policyId(seed.weights),
    offset: fs.existsSync(file('events.jsonl')) ? fs.statSync(file('events.jsonl')).size : 0, state: lab.s });
  lab.checkpoint();
  return lab;
}
