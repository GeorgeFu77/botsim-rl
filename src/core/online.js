import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { MLP, mulberry32 } from './mlp.js';
import { Platt } from './platt.js';
import { atomicJSON, validProbability } from './live-data.js';
import { marketEnd } from './resolution.js';

const id = (value) => createHash('sha256').update(JSON.stringify(value)).digest('hex').slice(0, 16);
function restore(saved, cfg) {
  if (!Number.isInteger(saved.d) || saved.d < 1 || !Number.isInteger(saved.h) || saved.h < 1) throw new Error('Invalid model dimensions');
  for (const [key, n] of Object.entries({ W1: saved.d * saved.h, b1: saved.h, W2: saved.h, mean: saved.d, std: saved.d })) {
    if (saved[key]?.length !== n || !saved[key].every(Number.isFinite)) throw new Error(`Invalid model ${key}`);
  }
  if (!saved.std.every((v) => v > 0) || ![saved.b2, saved.platt?.a, saved.platt?.b].every(Number.isFinite)) throw new Error('Invalid model calibration');
  const model = new MLP(saved.d, { hidden: saved.h, lr: cfg.lr, l2: saved.l2, gradientClip: 1 });
  for (const key of ['W1', 'b1', 'W2', 'mean', 'std']) model[key].set(saved[key]);
  model.b2 = saved.b2;
  return { model, platt: Platt.fromJSON(saved.platt) };
}
const snapshot = ({ model, platt }) => ({ ...model.toJSON(), platt: platt.toJSON() });
const predict = ({ model, platt }, x) => {
  const p = platt.apply(model.predict(x, 0));
  if (!validProbability(p)) throw new Error('Non-finite or saturated online prediction');
  return p;
};
const emptyEvaluation = () => ({ n: 0, firstAt: null, lastAt: null, champion: 0, candidate: 0,
  market: 0, championLog: 0, candidateLog: 0 });
const logLoss = (p, y) => -Math.log(Math.max(1e-9, y ? p : 1 - p));

export class OnlineLearner {
  constructor(file, base, cfg) {
    this.file = file; this.cfg = cfg; this.baseId = id(base);
    const s = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null;
    if (s && (s.version !== 1 || s.baseId !== this.baseId)) throw new Error('Online checkpoint/base mismatch; do not silently reset learning');
    this.learner = restore(s?.learner ?? base, cfg);
    if (!s) {
      // Fold the old fixed calibrator into the output layer. Online BCE now trains
      // the actual reported probability, rather than an uncalibrated intermediate.
      const { model, platt } = this.learner;
      model.W2 = model.W2.map((w) => w * platt.a);
      model.b2 = model.b2 * platt.a + platt.b;
      this.learner.platt = new Platt();
    }
    this.champion = restore(s?.champion ?? base, cfg);
    this.candidate = restore(s?.candidate ?? base, cfg);
    this.seen = new Set(s?.seen ?? []); this.replay = s?.replay ?? [];
    this.updates = s?.updates ?? 0; this.lastUpdateAt = s?.lastUpdateAt ?? null;
    this.lastMarketEnd = s?.lastMarketEnd ?? null; this.promotions = s?.promotions ?? 0;
    this.cycle = s?.cycle ?? 0; this.evaluation = s?.evaluation ?? emptyEvaluation();
    this.lastEvaluation = s?.lastEvaluation ?? null;
    this.bootstrapped = s?.bootstrapped ?? false;
    this.modelId = id(snapshot(this.champion)); this.candidateId = `${this.cycle}:${id(snapshot(this.candidate))}`;
    if ([this.learner, this.champion, this.candidate].some((m) => m.model.d !== base.d || m.model.h !== base.h) ||
        !Number.isSafeInteger(this.updates) || this.updates < 0 || this.seen.size !== this.updates || this.replay.length > cfg.replayMarkets ||
        this.replay.some((m) => ![0, 1].includes(m.y) || !m.rows.length ||
          m.rows.some((x) => x.length !== base.d || !x.every(Number.isFinite)))) throw new Error('Invalid online replay/checkpoint');
  }

  predictions(x) {
    return { pUp: predict(this.champion, x), candidatePUp: predict(this.candidate, x),
      candidateId: this.candidateId, modelId: this.modelId };
  }

  learn(slug, observations, resolution, now = Date.now(), evaluate = true) {
    const end = marketEnd(slug);
    if (this.seen.has(slug) || !end || end * 1000 > now || resolution?.slug !== slug ||
        resolution.periodEnd !== end || !['Up', 'Down'].includes(resolution.outcome)) return false;
    const rows = [...new Map(observations.map((o) => [o.key, o])).values()].filter((o) => o.slug === slug &&
      o.at >= (end - 300) * 1000 && o.at < end * 1000 && o.features?.length === this.learner.model.d &&
      o.features.every(Number.isFinite));
    if (!rows.length) return false;
    const y = Number(resolution.outcome === 'Up');
    // Only immutable, logged predictions made BEFORE resolution enter the gate.
    // Late labels from a previous candidate cycle still train, but cannot score it.
    const scored = rows.filter((o) => o.candidateId === this.candidateId && o.modelId === this.modelId &&
      [o.pUp, o.candidatePUp, o.qUp].every(validProbability));
    if (evaluate && scored.length) {
      const e = this.evaluation; e.n++;
      const at = Math.min(...scored.map((o) => o.at));
      e.firstAt = Math.min(e.firstAt ?? at, at); e.lastAt = Math.max(e.lastAt ?? at, at);
      for (const o of scored) {
        e.champion += (o.pUp - y) ** 2 / scored.length;
        e.candidate += (o.candidatePUp - y) ** 2 / scored.length;
        e.market += (o.qUp - y) ** 2 / scored.length;
        e.championLog += logLoss(o.pUp, y) / scored.length;
        e.candidateLog += logLoss(o.candidatePUp, y) / scored.length;
      }
    }
    const current = { slug, y, rows: rows.map((o) => o.features) };
    const rng = mulberry32(1337 + this.updates), batch = [current];
    for (let i = 1; i < this.cfg.batchMarkets && this.replay.length; i++) batch.push(this.replay[Math.floor(rng() * this.replay.length)]);
    const X = [], Y = [], weights = [];
    for (const market of batch) for (const x of market.rows) {
      X.push(...x); Y.push(market.y); weights.push(1 / market.rows.length);
    }
    // Each market has equal gradient weight regardless of missing decision offsets.
    this.learner.model.trainBatch(X, Y, Y.map((_, i) => i), weights.map((w) => w * Y.length / batch.length));
    restore(snapshot(this.learner), this.cfg); // Reject corrupt weights before persistence.
    this.replay.push(current);
    if (this.replay.length > this.cfg.replayMarkets) this.replay.shift();
    this.seen.add(slug); this.updates++; this.lastUpdateAt = now;
    this.lastMarketEnd = Math.max(this.lastMarketEnd ?? 0, end * 1000);
    const e = this.evaluation;
    if (e.n >= this.cfg.evaluationMarkets && e.lastAt - e.firstAt >= this.cfg.evaluationMs) {
      const promoted = (e.champion - e.candidate) / e.n >= this.cfg.minBrierGain &&
        e.candidateLog <= e.championLog && e.candidate <= e.market;
      this.lastEvaluation = { ...e, promoted, at: now };
      if (promoted) {
        this.champion = restore(snapshot(this.candidate), this.cfg);
        this.modelId = id(snapshot(this.champion)); this.promotions++;
      }
      this.rotateCandidate();
    }
    return true;
  }

  rotateCandidate() {
    this.candidate = restore(snapshot(this.learner), this.cfg); this.cycle++;
    this.candidateId = `${this.cycle}:${id(snapshot(this.candidate))}`; this.evaluation = emptyEvaluation();
  }

  save() {
    atomicJSON(this.file, { version: 1, baseId: this.baseId, learner: snapshot(this.learner),
      champion: snapshot(this.champion), candidate: snapshot(this.candidate),
      seen: [...this.seen], replay: this.replay, ...this.status(), evaluation: this.evaluation,
      bootstrapped: this.bootstrapped });
  }

  status() {
    return { updates: this.updates, lastUpdateAt: this.lastUpdateAt, lastMarketEnd: this.lastMarketEnd,
      promotions: this.promotions, cycle: this.cycle, modelId: this.modelId,
      learnerId: id(snapshot(this.learner)), candidateId: this.candidateId,
      evaluatedMarkets: this.evaluation.n, requiredMarkets: this.cfg.evaluationMarkets,
      lastEvaluation: this.lastEvaluation };
  }
}
